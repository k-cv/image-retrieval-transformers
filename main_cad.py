import argparse
import datetime
import logging
import os
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.backends.cudnn as cudnn

from timm.models import create_model  # timmのcreate_modelをインポート
from timm.optim import create_optimizer
from timm.utils import NativeScaler
from torch.utils.tensorboard import SummaryWriter

from torch.utils.data import RandomSampler
from pytorch_metric_learning.samplers import MPerClassSampler
from pytorch_metric_learning.distances import CosineSimilarity
from pytorch_metric_learning.losses import ContrastiveLoss

from xbm import XBM
from datasets.cad_from_features import CADFeatureDataset  # CADFeatureDatasetをインポート
from engine_cad import train, evaluate  # engine_cad.py に合わせて変更
from regularizer import DifferentialEntropyRegularization


def get_args_parser():
    parser = argparse.ArgumentParser('Training CAD Features for Retrieval', add_help=False)

    # Model parameters
    parser.add_argument('--model', default='deit_small_distilled_patch16_224', type=str, help='Name of model to train')
    parser.add_argument('--input-size', default=130, type=int, help='Input sequence length (e.g., 130)')
    parser.add_argument('--embed-dim', default=256, type=int, help='Embedding dimension (e.g., 256)')
    parser.add_argument('--drop', type=float, default=0.0, help='Dropout rate (default: 0.)')
    parser.add_argument('--drop-path', type=float, default=0.1, metavar='PCT', help='Drop path rate (default: 0.1)')

    # Optimizer parameters
    parser.add_argument('--max-iter', default=2000, type=int)
    parser.add_argument('--batch-size', default=64, type=int)
    parser.add_argument('--lr', type=float, default=3e-5, help='Learning rate (3e-5 for category level)')
    parser.add_argument('--opt', default='adamw', type=str, help='Optimizer (default: "adamw")')
    parser.add_argument('--opt-eps', default=1e-8, type=float, help='Optimizer Epsilon (default: 1e-8)')
    parser.add_argument('--opt-betas', default=None, type=float, nargs='+', help='Optimizer Betas (default: None, use opt default)')
    parser.add_argument('--clip-grad', type=float, default=None, help='Clip gradient norm (default: None, no clipping)')
    parser.add_argument('--momentum', type=float, default=0.9, help='SGD momentum (default: 0.9)')
    parser.add_argument('--weight-decay', type=float, default=5e-4, help='Weight decay (default: 5e-4)')

    # Dataset parameters
    parser.add_argument('--dataset', default='cad_dataset', type=str, help='Dataset name')
    parser.add_argument('--data-dir', 
        default='/path/to/features',  
        type=str, 
        help='Directory containing feature .npy files')
    parser.add_argument('--label-file', 
        default='/path/to/labels.txt', 
        type=str, 
        help='Path to label file')
    parser.add_argument('--m', default=0, type=int, help="Sample m features per class")
    parser.add_argument('--rank', default=[1, 5, 10], nargs="+", type=int, help="Compute recall@r")
    parser.add_argument('--num-workers', default=16, type=int)
    parser.add_argument('--pin-mem', action='store_true')
    parser.add_argument('--no-pin-mem', action='store_false', dest='pin_mem')
    parser.set_defaults(pin_mem=True)

    # Loss parameters
    parser.add_argument('--lambda-reg', type=float, default=0.7, help="Regularization strength")
    parser.add_argument('--margin', type=float, default=0.5, help="Negative margin of contrastive loss (beta)")

    # XBM parameters
    parser.add_argument('--memory-ratio', type=float, default=1.0, help="Size of the XBM queue")
    parser.add_argument('--encoder-momentum', type=float, default=0.999, help="Momentum for the key encoder") 

    # MISC
    parser.add_argument('--logging-freq', type=int, default=50)
    parser.add_argument('--output-dir', default='./outputs', help='Path where to save, empty for no saving')
    parser.add_argument('--log-dir', default='./logs', help='Path where to save TensorBoard logs')
    parser.add_argument('--device', default='cuda:0', help='Device to use for training/testing')
    parser.add_argument('--seed', default=1127, type=int)

    return parser


def get_dataset(args):
    """
    CADFeatureDataset を使用してトレーニング、クエリ、ギャラリーのデータセットを取得します。
    """
    # トレーニングデータセット
    dataset_train = CADFeatureDataset(
        data_dir=args.data_dir,
        label_file=args.label_file,
        split='train'
    )

    # クエリデータセット
    dataset_query = CADFeatureDataset(
        data_dir=args.data_dir,
        label_file=args.label_file,
        split='query'
    )

    # ギャラリーデータセット
    dataset_gallery = CADFeatureDataset(
        data_dir=args.data_dir,
        label_file=args.label_file,
        split='gallery'
    )

    return dataset_train, dataset_query, dataset_gallery


def main(args):

    logging.info("=" * 20 + " training arguments " + "=" * 20)
    for k, v in vars(args).items():
        logging.info(f"{k}: {v}")
    logging.info("=" * 60)

    # Fix random seed
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(args.seed)
    random.seed(args.seed)

    device = torch.device(args.device)

    # Get training/query/gallery dataset
    dataset_train, dataset_query, dataset_gallery = get_dataset(args)
    logging.info(f"Number of training examples: {len(dataset_train)}")
    logging.info(f"Number of query examples: {len(dataset_query)}")
    if dataset_gallery is not None:
        logging.info(f"Number of gallery examples: {len(dataset_gallery)}")

    # サンプラーの設定
    sampler_train = RandomSampler(dataset_train)
    if args.m > 0:
        sampler_train = MPerClassSampler([label for _, label in dataset_train], m=args.m, batch_size=args.batch_size)

    data_loader_train = torch.utils.data.DataLoader(
        dataset_train,
        sampler=sampler_train,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,
        drop_last=True,
    )

    data_loader_query = torch.utils.data.DataLoader(
        dataset_query,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,
        drop_last=False,
        shuffle=False
    )

    data_loader_gallery = None
    if dataset_gallery is not None:
        data_loader_gallery = torch.utils.data.DataLoader(
            dataset_gallery,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            pin_memory=args.pin_mem,
            drop_last=False,
            shuffle=False
        )

    # モデルの作成 (timmを使用)
    original_model = create_model(
        args.model,
        pretrained=False,  # CADデータでは事前学習済みモデルは不要
        num_classes=0,
        drop_rate=args.drop,
        drop_path_rate=args.drop_path
    )

    class CustomViTModel(torch.nn.Module):
        def __init__(self, original_model, input_dim=256, target_dim=384):
            super(CustomViTModel, self).__init__()
            self.original_model = original_model
            # 次元を256 -> 384に変換するためのLinear層を追加
            self.fc = torch.nn.Linear(input_dim, target_dim)
            # ポジションエンベディングをリサイズして、ターゲット次元に合わせる
            self.pos_embed = torch.nn.Parameter(
                original_model.pos_embed[:, :target_dim, :].clone()
            )
            self.pos_drop = original_model.pos_drop
            self.blocks = original_model.blocks
            self.norm = original_model.norm

        def forward(self, x):
            # 入力特徴量の次元を256 -> 384に変換
            x = self.fc(x)  # 次元を合わせる
            # ポジションエンベディングを特徴量に加算
            x = x + self.pos_embed[:, :x.size(1), :]
            x = self.pos_drop(x)
            for blk in self.blocks:
                x = blk(x)
            x = self.norm(x)
            return x


    # カスタムモデルを作成
    model = CustomViTModel(original_model, input_dim=256)
    model.to(device)

    # モデルの構造を出力
    print(model)
    import pdb;pdb.set_trace()

    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logging.info(f'Number of params: {round(n_parameters / 1_000_000, 2):.2f} M')

    # Optimizer の作成
    optimizer = create_optimizer(args, model)

    # 損失関数と正則化
    criterion = ContrastiveLoss(
        pos_margin=1.0,
        neg_margin=args.margin,
        distance=CosineSimilarity(),
    )
    regularization = DifferentialEntropyRegularization()
    xbm = XBM(
        memory_size=int(len(dataset_train) * args.memory_ratio),
        embedding_dim=args.embed_dim,
        device=device
    )
    loss_scaler = NativeScaler()

    log_writer = None
    if args.log_dir is not None:
        os.makedirs(args.log_dir, exist_ok=True)
        log_writer = SummaryWriter(log_dir=args.log_dir)

    start_time = time.time()

    # モデルのトレーニング
    train(
        model,
        None, #momentum_encoder,
        criterion,
        xbm,
        regularization,
        data_loader_train,
        optimizer,
        device,
        loss_scaler,
        args.clip_grad,
        log_writer,
        args=args
    )

    logging.info("Start evaluation job")

    # モデルの評価
    evaluate(
        data_loader_query,
        data_loader_gallery,
        model,
        device,
        log_writer,
        rank=sorted(args.rank)
    )

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    logging.info('Training time {}'.format(total_time_str))


if __name__ == "__main__":

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(message)s',
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    parser = argparse.ArgumentParser('Training CAD Features for Retrieval', parents=[get_args_parser()])
    args = parser.parse_args()

    if args.log_dir:
        args.log_dir = os.path.join(args.log_dir, args.dataset)
        Path(args.log_dir).mkdir(parents=True, exist_ok=True)

    if args.output_dir:
        args.output_dir = os.path.join(args.output_dir, args.dataset)
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    main(args)
