import os
import numpy as np
from torchvision import transforms
from PIL import Image

# BaseDatasetをインポート
from datasets.base_dataset import BaseDataset

class CADImageDataset(BaseDataset):
    def __init__(self, label_file, data_dir=None, input_size=224, split="train", ignore_label=-1):
        """
        :param label_file: クラスラベルが記載されたファイル
        :param split: "train" or "test"
        :param ignore_label: ラベルが指定されていない場合の値（デフォルトは -1）
        """
        self.label_file = label_file
        self.ignore_label = ignore_label
        super(CADImageDataset, self).__init__(data_dir=data_dir, input_size=input_size, split=split)
        self.labels = self.load_labels(label_file)  # ラベルをロード

        # ラベルが有効（ignore_label でないもの）のデータのみを保持
        self.valid_indices = [i for i, label in enumerate(self.labels) if label != self.ignore_label]
        self.paths = [self.get_image_path(i) for i in self.valid_indices]
        self.labels = [self.labels[i] for i in self.valid_indices]

        assert len(self.paths) == len(self.labels), "有効な特徴量とラベルの数が一致しません"

    def load_labels(self, label_file):
        # ラベルファイルをロードし、各サンプルにクラスラベルを付与する
        labels = {}
        with open(label_file, 'r') as f:
            for class_label, line in enumerate(f):
                class_samples = list(map(int, line.strip().split()))  # スペース区切りのラベルをリストに
                for sample in class_samples:
                    labels[sample] = class_label  # 各 feature_index に対してクラスラベルを付与
        # ラベルリストを特徴量の順序に従って作成
        max_samples = max(labels.keys()) + 1  # インデックスが0から始まるため、+1
        labels_with_ignore = [self.ignore_label] * max_samples
        for i in range(max_samples):
            if i in labels:
                labels_with_ignore[i] = labels[i]  # feature_indexに基づいてクラスラベルを付与
        return labels_with_ignore

    def get_image_path(self, idx):
        # idxに基づいて画像のパスを生成する
        return f'/home/kfujii/vitruvion/outputs/2024-09-05/12-54-06_all_images/output_sketch_{idx}.png'

    def __getitem__(self, index):
        # インデックスに基づいて画像をロード
        image_path = self.paths[index]
        label = self.labels[index]

        # 画像を開き、transformを適用
        image = Image.open(image_path).convert('RGB')
        transform = transforms.Compose([
            transforms.Resize((self.input_size, self.input_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        image = transform(image)

        return image, label
