import os
import torch
import numpy as np
from torchvision import transforms
from PIL import Image

# BaseDatasetをインポート
from datasets.base_dataset import BaseDataset

class CADImageDataset(BaseDataset):
    def __init__(self, label_file, data_dir=None, input_size=224, split="train", ignore_label=-1, index_file='/home/kfujii/vitruvion/encoder_index2.pth'):
        """
        :param label_file: クラスラベルが記載されたファイル
        :param split: "train" or "test"
        :param ignore_label: ラベルが指定されていない場合の値（デフォルトは -1）
        :param index_file: index.pth ファイルのパス
        :param input_size: 画像の入力サイズ (デフォルトは224)
        """
        self.label_file = label_file
        self.ignore_label = ignore_label
        self.input_size = input_size  # input_size を初期化
        self.index_file = index_file
        self.split = split  # split を保持
        super(CADImageDataset, self).__init__(data_dir=data_dir, input_size=input_size, split=split)
        
        # index.pth ファイルの読み込み
        if self.index_file:
            self.encoder_index = torch.load(self.index_file)
        
        self.labels, self.paths = self.load_labels_and_paths(label_file)  # ラベルとパスをロード

        assert len(self.paths) == len(self.labels), "有効な特徴量とラベルの数が一致しません"

    def load_labels_and_paths(self, label_file):
        # ラベルファイルをロードし、各サンプルにクラスラベルを付与する
        labels = {}
        paths = []
        with open(label_file, 'r') as f:
            for class_label, line in enumerate(f):
                class_samples = list(map(int, line.strip().split()))  # スペース区切りのラベルをリストに
                for sample in class_samples:
                    labels[sample] = class_label  # 各 feature_index に対してクラスラベルを付与

        # train_test_splitに基づき、データを分割
        max_samples = max(labels.keys()) + 1
        labels_with_ignore = [self.ignore_label] * max_samples
        paths_with_ignore = [''] * max_samples
        for i in range(max_samples):
            if i in labels:
                labels_with_ignore[i] = labels[i]  # feature_indexに基づいてクラスラベルを付与
                paths_with_ignore[i] = self.get_image_path(i)

        # トレーニングまたはテストセット用にフィルタリング
        if self.split == "train":
            split_filter = 1  # トレーニングセットは1
        else:
            split_filter = 0  # テストセットは0

        # `label.txt`の情報を基にsplit_filterを使用してフィルタリング
        valid_labels = []
        valid_paths = []
        with open('data/CAD/train_test_split.txt', 'r') as split_file:
            for line in split_file:
                image_id, is_training_image = map(int, line.strip().split())
                if is_training_image == split_filter:
                    valid_labels.append(labels_with_ignore[image_id])
                    valid_paths.append(paths_with_ignore[image_id])

        return valid_labels, valid_paths

    def get_image_path(self, idx):
        # index.pthを使って、idxを変換してから画像のパスを生成する
        actual_idx = self.encoder_index[idx].item()  # index.pthに対応するファイルインデックスを取得
        return f'/home/kfujii/vitruvion/outputs/2024-09-05/12-54-06_all_images/output_sketch_{actual_idx}.png'

    def __getitem__(self, index):
        # インデックスに基づいて画像をロード
        image_path = self.paths[index]
        label = self.labels[index]

        # 画像を開き、transformを適用
        image = Image.open(image_path).convert('RGB')
        transform = transforms.Compose([
            transforms.Resize((self.input_size, self.input_size)),  # input_sizeを使用
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        image = transform(image)

        return image, label
