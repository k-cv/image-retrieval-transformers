import logging
import math
import os
import sys
from typing import Union

import torch
import torch.nn.functional as F
from tqdm import tqdm
from torch import nn

from metric import recall
from xbm import XBM, momentum_update_key_encoder

from PIL import Image, ImageDraw, ImageFont

from combine import CombinedModel


def train_combined_model(
        model: CombinedModel,
        criterion: nn.Module,
        xbm: XBM,
        regularization: nn.Module,
        data_loader: torch.utils.data.DataLoader,
        optimizer: torch.optim.Optimizer,
        device: torch.device,
        loss_scaler, max_norm,
        log_writer=None,
        args=None,
        geo_features_path=None
):
    model.train(True)
    optimizer.zero_grad()
    iteration = 0
    _train_loader = iter(data_loader)
    
    # 幾何特徴のロード
    geo_features_data = torch.load(geo_features_path)  # .pthファイルからロード

    while iteration < args.max_iter:
        try:
            images, targets, indices = _train_loader.next()  # 画像, ラベル, インデックスを取得
        except StopIteration:
            _train_loader = iter(data_loader)
            images, targets, indices = _train_loader.next()
        
        images = images.to(device)
        targets = targets.to(device)
        # geo_features_data は辞書型 {'features': Tensor, 'labels': Tensor} を想定
        geo_features_data = torch.load(geo_features_path)  # .pthファイルからロード

        # indices のデータ型を確認し、必要なら変換
        if isinstance(indices, list):
            indices = [int(idx) for idx in indices]  # 文字列を整数に変換
        indices = torch.tensor(indices, dtype=torch.long)  # 整数型テンソルに変換

        # 'features' テンソルからインデックスを指定して取得
        geo_features = geo_features_data['features'][indices]  # 対応するインデックスの特徴を取得

        # 必要に応じてデバイスに移動
        geo_features = geo_features.to(device)

        
        # モデルのフォワードパス
        features = model(images, geo_features)
        features = nn.functional.normalize(features, dim=1)
        
        # Contrastive Lossの計算
        loss_contr = criterion(features, targets)
        
        # XBMの処理
        xbm.enqueue_dequeue(features.detach(), targets.detach())
        xbm_features, xbm_targets = xbm.get()
        loss_contr += criterion(features, targets, ref_emb=xbm_features, ref_labels=xbm_targets)
        
        # 正則化項の計算
        loss_koleo = regularization(features)
        loss = loss_contr + loss_koleo * args.lambda_reg
        
        # 勾配計算と更新
        optimizer.zero_grad()
        loss_scaler(loss, optimizer, clip_grad=max_norm, parameters=model.parameters())
        
        iteration += 1
        if (iteration > 0 and iteration % args.logging_freq == 0) or iteration == args.max_iter:
            logging.info(
                f"Iteration [{iteration:5,}/{args.max_iter:5,}] "
                f"contrastive: {loss_contr.item():.4f}  "
                f"regularization: {loss_koleo.item():.4f}(x {args.lambda_reg}) "
                f"total_loss: {loss.item():.4f}"
            )
            if log_writer is not None:
                log_writer.add_scalar("loss/contrastive", loss_contr.item(), iteration)
                log_writer.add_scalar("loss/regularization", loss_koleo.item(), iteration)
                log_writer.add_scalar("loss/total", loss.item(), iteration)

@torch.no_grad()
def evaluate(data_loader_query, data_loader_gallery, encoder, device, geo_features_path, output_dir="output", log_writer=None, rank=[1, 5, 10]):
    # 幾何特徴のロード
    geo_features_data = torch.load(geo_features_path)
    geo_features_tensor = geo_features_data['features']

    query_features = []
    query_labels = []
    query_paths = []

    for images, targets, indices in data_loader_query:
        images = images.to(device)
        geo_features = geo_features_tensor[indices].to(device)

        output = encoder(images, geo_features)
        if isinstance(output, tuple):
            output = output[0]
        output = F.normalize(output, dim=1)
        query_features.append(output.detach().cpu())
        query_labels += targets.tolist()
        query_paths.extend([data_loader_query.dataset.index_to_path[idx.item()] for idx in indices])

    query_features = torch.cat(query_features, dim=0)
    query_labels = torch.LongTensor(query_labels)
    
    """debugs"""
    # クエリとギャラリーのクラスラベルを出力
    print(f"Query Labels: {query_labels.unique()}")
    # print(f"Gallery Labels: {gallery_labels.unique()}")
    from collections import Counter

    # クラスラベルの分布を確認
    query_label_counts = Counter(query_labels.tolist())
    # gallery_label_counts = Counter(gallery_labels.tolist())
    print(f"Query Label Counts: {query_label_counts}")
    # print(f"Gallery Label Counts: {gallery_label_counts}")
    import pdb;pdb.set_trace()

    # ギャラリーデータの特徴量を取得
    if data_loader_gallery is None:
        gallery_features = query_features
        gallery_labels = query_labels
        gallery_paths = query_paths
    else:
        gallery_features = []
        gallery_labels = []
        gallery_paths = []
        for images, targets, indices in tqdm(data_loader_gallery, total=len(data_loader_gallery), desc="gallery"):
            images = images.to(device)
            geo_features = geo_features_tensor[indices].to(device)

            output = encoder(images, geo_features)
            if isinstance(output, tuple):
                output = output[0]
            output = F.normalize(output, dim=1)
            gallery_features.append(output.detach().cpu())
            gallery_labels += targets.tolist()
            gallery_paths.extend([data_loader_gallery.dataset.index_to_path[idx.item()] for idx in indices])

        gallery_features = torch.cat(gallery_features, dim=0)
        gallery_labels = torch.LongTensor(gallery_labels)

    # 修正: rank の調整
    max_rank = min(len(gallery_features), len(query_features))
    rank = [r for r in rank if r <= max_rank]
    if not rank:
        raise ValueError(f"No valid ranks! The rank values {args.rank} exceed the dataset size.")

    # Recall 計算
    recall_list, top_k_indices = recall(query_features, query_labels, rank=rank, gallery_features=gallery_features, gallery_labels=gallery_labels)

    # ...
    for (k, _recall) in zip(rank, recall_list):
        logging.info(f"Recall@{k} : {_recall:.2%}")
        if log_writer is not None:
            log_writer.add_scalar(f"metric/Recall", _recall, k)

    return recall_list

    # # クエリ画像と `top 4` 画像を保存
    # for i, query_idx in enumerate(top_k_indices):
    #     query_label = query_labels[i].item()
    #     top_k = query_idx[:4]  # `top 4` 画像
    #     top_k_labels = [gallery_labels[idx].item() for idx in top_k]
        
    #     # ラベルに誤りがある場合に画像を保存
    #     if any(label != query_label for label in top_k_labels):
    #         query_image = Image.open(query_paths[i]).convert("RGB")
    #         query_image = add_text_to_image(query_image, query_paths[i])  # パス名を追加
    #         query_image.save(os.path.join(output_dir, f"query_{i}_label_{query_label}.png"))
            
    #         for j, idx in enumerate(top_k):
    #             top_image = Image.open(gallery_paths[idx]).convert("RGB")
    #             top_image = add_text_to_image(top_image, gallery_paths[idx])  # パス名を追加
    #             top_image.save(os.path.join(output_dir, f"query_{i}_top{j+1}_label_{top_k_labels[j]}.png"))
    for (k, _recall) in zip(rank, recall_list):
        logging.info(f"Recall@{k} : {_recall:.2%}")
        if log_writer is not None:
            log_writer.add_scalar(f"metric/Recall", _recall, k)

    return recall_list


def add_text_to_image(image, text):
    """
    画像の上部にテキストを追加する関数
    """
    # 描画オブジェクトを作成
    draw = ImageDraw.Draw(image)
    
    # Unicode対応のフォントを指定
    font_path = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"  # フォントパスを環境に合わせて調整
    try:
        font = ImageFont.truetype(font_path, 16)
    except IOError:
        font = ImageFont.load_default()  # フォントが見つからない場合はデフォルトフォントにフォールバック

    # テキスト位置とカラーの設定
    text_position = (10, 10)  # 画像の上部に位置
    text_color = (255, 255, 255)  # 白色のテキスト
    text_background = (0, 0, 0)  # 黒の背景

    # テキストのバックグラウンド用の矩形を描画
    text_size = draw.textsize(text, font=font)
    background_position = (text_position[0], text_position[1], text_position[0] + text_size[0], text_position[1] + text_size[1])
    draw.rectangle(background_position, fill=text_background)

    # テキストを描画
    draw.text(text_position, text, fill=text_color, font=font)

    return image