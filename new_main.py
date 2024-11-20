import argparse
import logging
import os
import random
import time
import datetime
from pathlib import Path

import numpy as np
import torch
from timm.models import create_model
from timm.optim import create_optimizer
from timm.utils import NativeScaler
from torch.utils.data import DataLoader, RandomSampler
from pytorch_metric_learning.samplers import MPerClassSampler
from pytorch_metric_learning.distances import CosineSimilarity
from pytorch_metric_learning.losses import ContrastiveLoss

from datasets import get_dataset
from enginec import train, evaluate
from regularizer import DifferentialEntropyRegularization
from xbm import XBM
from torch.utils.tensorboard import SummaryWriter


class FeatureProjector(torch.nn.Module):
    def __init__(self, input_dim=256, output_dim=384):
        super().__init__()
        self.linear = torch.nn.Linear(input_dim, output_dim)

    def forward(self, x):
        return self.linear(x)


class CustomViTModel(torch.nn.Module):
    def __init__(self, original_model, seq_len):
        super().__init__()
        self.original_model = original_model
        self.cls_token = original_model.cls_token
        self.pos_drop = original_model.pos_drop
        self.blocks = original_model.blocks
        self.norm = original_model.norm

        num_tokens = 1 + (1 if hasattr(original_model, 'dist_token') else 0)
        self.pos_embed = torch.nn.Parameter(
            torch.zeros(1, seq_len + num_tokens, original_model.embed_dim)
        )
        torch.nn.init.trunc_normal_(self.pos_embed, std=.02)

    def forward(self, x):
        B = x.shape[0]
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1) + self.pos_embed
        x = self.pos_drop(x)
        for blk in self.blocks:
            x = blk(x)
        return self.norm(x)


def setup_logger(log_dir, dataset_name):
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(message)s',
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    if log_dir:
        os.makedirs(log_dir, exist_ok=True)
        return SummaryWriter(log_dir=os.path.join(log_dir, dataset_name))
    return None


def set_random_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def create_data_loaders(args, dataset_train, dataset_query, dataset_gallery):
    sampler_train = MPerClassSampler(
        [label for _, label in dataset_train],
        m=args.m,
        batch_size=args.batch_size
    ) if args.m > 0 else RandomSampler(dataset_train)

    train_loader = DataLoader(
        dataset_train,
        sampler=sampler_train,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,
        drop_last=True,
    )
    query_loader = DataLoader(
        dataset_query,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,
        drop_last=False,
        shuffle=False,
    )
    gallery_loader = None if dataset_gallery is None else DataLoader(
        dataset_gallery,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,
        drop_last=False,
        shuffle=False,
    )
    return train_loader, query_loader, gallery_loader


def main(args):
    set_random_seed(args.seed)
    device = torch.device(args.device)
    log_writer = setup_logger(args.log_dir, args.dataset)

    dataset_train, dataset_query, dataset_gallery = get_dataset(args)
    train_loader, query_loader, gallery_loader = create_data_loaders(
        args, dataset_train, dataset_query, dataset_gallery
    )

    original_model = create_model(
        args.model, pretrained=False, num_classes=0,
        drop_rate=args.drop, drop_path_rate=args.drop_path
    )
    model = CustomViTModel(original_model, args.input_size).to(device)
    projector = FeatureProjector(256, args.embed_dim).to(device)

    momentum_encoder = None
    if args.encoder_momentum is not None:
        momentum_encoder = CustomViTModel(
            create_model(
                args.model, pretrained=False, num_classes=0,
                drop_rate=args.drop, drop_path_rate=args.drop_path
            ), args.input_size
        ).to(device)

    optimizer = create_optimizer(args, model)
    criterion = ContrastiveLoss(
        pos_margin=1.0, neg_margin=args.margin, distance=CosineSimilarity()
    )
    regularization = DifferentialEntropyRegularization()
    xbm = XBM(
        memory_size=int(len(dataset_train) * args.memory_ratio),
        embedding_dim=args.embed_dim,
        device=device
    )
    loss_scaler = NativeScaler()

    train(
        model, projector, momentum_encoder, criterion, xbm, regularization,
        train_loader, optimizer, device, loss_scaler, args.clip_grad, log_writer, args
    )
    evaluate(query_loader, gallery_loader, model, device, projector, log_writer, sorted(args.rank))

    logging.info(f'Training time: {str(datetime.timedelta(seconds=int(time.time() - start_time)))}')


if __name__ == "__main__":
    parser = argparse.ArgumentParser('Training CAD Features for Retrieval')
    args = parser.parse_args()

    if args.log_dir:
        args.log_dir = os.path.join(args.log_dir, args.dataset)
    if args.output_dir:
        args.output_dir = os.path.join(args.output_dir, args.dataset)
    main(args)

