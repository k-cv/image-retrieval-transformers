import logging
import math
import os
import sys
from typing import Union

import torch
import torch.nn.functional as F
from tqdm import tqdm

from metric import recall
from xbm import XBM, momentum_update_key_encoder

def train(
        encoder: torch.nn.Module,
        projector: torch.nn.Module,  # 特徴次元数を変換するプロジェクター
        encoder_k: Union[torch.nn.Module, None],
        criterion: torch.nn.Module,
        xbm: XBM,
        regularization: torch.nn.Module,
        data_loader: torch.utils.data.DataLoader,
        optimizer: torch.optim.Optimizer,
        device: torch.device,
        loss_scaler, max_norm,
        log_writer=None,
        args=None
):
    encoder.train(True)
    optimizer.zero_grad()
    iteration = 0
    _train_loader = iter(data_loader)

    while iteration < args.max_iter:
        try:
            # 特徴を取得 (形状: [バッチサイズ, 130, 256])
            features, targets = _train_loader.next()
        except StopIteration:
            _train_loader = iter(data_loader)
            features, targets = _train_loader.next()

        # 特徴とターゲットをデバイスに送る
        features = features.to(device)
        targets = targets.to(device)

        # 特徴次元数を256から384に変換
        features = projector(features)  # [バッチサイズ, 256, 384] に変換
        f_tmp = features.clone()

        # 特徴をSiamese Networkで処理
        features = encoder(features)
        if isinstance(features, tuple):
            features = features[0]

        # # 特徴をシーケンス方向 (dim=2) で正規化
        # features = F.normalize(features, dim=2)
        # クラストークンの特徴を取得
        # クラストークンの特徴を取得
        features = features[:, 0, :]  # [バッチサイズ, 384]
        features = F.normalize(features, dim=1)

        if encoder_k is not None:
            with torch.no_grad(), torch.cuda.amp.autocast():
                features_k = encoder_k(f_tmp)
                if isinstance(features_k, tuple):
                    features_k = features_k[0]
                features_k = features_k[:, 0, :]
                features_k = F.normalize(features_k, dim=1)
        else:
            features_k = features


        # シーケンス方向 (dim=1) で平均を取り、2次元に変換
        features_k_avg = features_k  # [バッチサイズ, 384]
        features_avg = features      # [バッチサイズ, 384]
        
        # TODO: ここの特徴量を抽出してconcatするといいのではないか

        # XBMを用いてキューに特徴を登録
        xbm.enqueue_dequeue(features_k_avg.detach(), targets.detach())

        # Contrastive Loss 計算
        loss_contr = criterion(features, targets)
        loss_koleo = regularization(features)
        xbm_features, xbm_targets = xbm.get()
        loss_contr += criterion(features, targets, ref_emb=xbm_features, ref_labels=xbm_targets)

        # 総損失の計算
        loss = loss_contr + loss_koleo * args.lambda_reg

        loss_contr_value = loss_contr.item()
        loss_koleo_value = loss_koleo.item()
        loss_value = loss.item()

        # 勾配の計算とパラメータ更新
        if not math.isfinite(loss_value):
            logging.warning("Loss is {}, stopping training".format(loss_value))
            sys.exit(1)

        optimizer.zero_grad()
        is_second_order = hasattr(optimizer, 'is_second_order') and optimizer.is_second_order
        loss_scaler(loss, optimizer, clip_grad=max_norm, parameters=encoder.parameters(), create_graph=is_second_order)

        iteration += 1

        # モーメンタムエンコーダの更新
        if encoder_k is not None:
            momentum_update_key_encoder(encoder, encoder_k, args.encoder_momentum)

        # ロギングと進捗の出力
        if (iteration > 0 and iteration % args.logging_freq == 0) or iteration == args.max_iter:
            logging.info(
                f"Iteration [{iteration:5,}/{args.max_iter:5,}] "
                f"contrastive: {loss_contr.item():.4f}  "
                f"regularization : {loss_koleo.item():.4f}(x {args.lambda_reg}) "
                f"total_loss: {loss_value:.4f}  "
            )
            if log_writer is not None:
                log_writer.add_scalar("loss/contrastive", loss_contr_value, iteration)
                log_writer.add_scalar("loss/regularization", loss_koleo_value, iteration)
                log_writer.add_scalar("loss/total", loss_value, iteration)

    # モデルの保存
    save_path = os.path.join(args.output_dir, "model_checkpoint.pth")
    torch.save({
        'encoder_state_dict': encoder.state_dict(),
        'projector_state_dict': projector.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        # エポック数や他の情報を保存する場合
        # 'epoch': current_epoch,
    }, save_path)



@torch.no_grad()
def evaluate(
        data_loader_query,
        data_loader_gallery,
        encoder,
        device,
        projector,          # projector を追加
        log_writer=None,
        rank=[1, 5, 10]):
    encoder.eval()
    recall_list = []

    query_features = []
    query_labels = []

    # クエリデータの処理
    for (features, targets) in tqdm(data_loader_query, total=len(data_loader_query), desc="query"):
        features = features.to(device)
        features = projector(features)  # 特徴次元数を変換
        output = encoder(features)
        if isinstance(output, tuple):
            output = output[0]
        # クラストークンの特徴を取得
        output = output[:, 0, :]  # [バッチサイズ, 384]
        output = F.normalize(output, dim=1)
        query_features.append(output.detach().cpu())
        query_labels += targets.tolist()

    query_features = torch.cat(query_features, dim=0)
    query_labels = torch.LongTensor(query_labels)

    if data_loader_gallery is None:
        recall_list = recall(query_features, query_labels, rank=rank)

    else:
        gallery_features = []
        gallery_labels = []
        for (features, targets) in tqdm(data_loader_gallery, total=len(data_loader_gallery), desc="gallery"):
            features = features.to(device)
            features = projector(features)  # 特徴次元数を変換

            with torch.cuda.amp.autocast():
                output = encoder(features)
                if isinstance(output, tuple):
                    output = output[0]
                output = output[:, 0, :]  # クラストークンの特徴を取得
                output = F.normalize(output, dim=1)
                gallery_features.append(output.detach().cpu())
                gallery_labels += targets.tolist()

        gallery_features = torch.cat(gallery_features, dim=0)
        gallery_labels = torch.LongTensor(gallery_labels)
        recall_list = recall(query_features, query_labels, rank=rank, gallery_features=gallery_features, gallery_labels=gallery_labels)

    for (k, _recall) in zip(rank, recall_list):
        logging.info(f"Recall@{k} : {_recall:.2%}")
        if log_writer is not None:
            log_writer.add_scalar(f"metric/Recall", _recall, k)

    return recall_list