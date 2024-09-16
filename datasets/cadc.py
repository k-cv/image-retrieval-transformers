import os
import torch
import numpy as np
from torch.utils.data import Dataset
import matplotlib.pyplot as plt


class CADFeatureDataset(Dataset):
    def __init__(self, data_dir, label_file, split="train", ignore_label=-1):
        """
        :param data_dir: 特徴量ファイルが保存されているディレクトリ
        :param label_file: クラスラベルが記載されたファイル
        :param split: "train" or "test"
        :param ignore_label: ラベルが指定されていない場合の値（デフォルトは -1）
        """
        self.data_dir = data_dir
        self.split = split
        self.ignore_label = ignore_label
        self.features = self.load_features()  # 特徴量をロード
        self.labels = self.load_labels(label_file)  # ラベルをロード
        
        # ラベルが有効（ignore_label でないもの）のデータのみを保持
        self.valid_indices = [i for i, label in enumerate(self.labels) if label != self.ignore_label]
        self.features = self.features[self.valid_indices]
        self.labels = [self.labels[i] for i in self.valid_indices]

        assert len(self.features) == len(self.labels), "有効な特徴量とラベルの数が一致しません"

    def load_features(self):
        # 特徴量ファイル (.pthファイル) をロードする
        feature_file = os.path.join(self.data_dir)
        print("特徴量読み込み中...")
        features = torch.load(feature_file)  # [データ数, シーケンス長, 256]
        print("特徴量読み込み完了しました")
        return features

    def load_labels(self, label_file):
        # ラベルファイルをロードし、各サンプルにクラスラベルを付与する
        labels = {}
        with open(label_file, 'r') as f:
            for class_label, line in enumerate(f):
                class_samples = list(map(int, line.strip().split()))  # スペース区切りのラベルをリストに
                for sample in class_samples:
                    labels[sample] = class_label  # 各 feature_index に対してクラスラベルを付与
        # ラベルリストを特徴量の順序に従って作成
        max_samples = len(self.features)
        labels_with_ignore = [self.ignore_label] * max_samples
        for i in range(max_samples):
            if i in labels:
                labels_with_ignore[i] = labels[i]  # feature_indexに基づいてクラスラベルを付与
        return labels_with_ignore

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        # 特徴量とラベルを取得する
        feature = self.features[idx]  # [シーケンス長, 256]
        label = self.labels[idx]
        return torch.tensor(feature, dtype=torch.float32), torch.tensor(label, dtype=torch.long)

    def __repr__(self):
        return f"{self.__class__.__name__}(split={self.split}, num_samples={len(self)})"


if __name__ == "__main__":
    # データセットの準備
    data_dir = '/home/kfujii/vitruvion/encoder_features2.pth'  # CADデータセットのパスを指定
    label_file = '/home/kfujii/image-retrieval-transformers/data/CAD/label.txt'  # ラベルファイルのパスを指定
    dataset = CADFeatureDataset(data_dir=data_dir, label_file=label_file, split="train")

    # 一つのデータを取り出す
    idx = 0  # 取り出したいデータのインデックス
    feature, label = dataset[idx]

    # 特徴量を表示
    print(f"Label: {label}")
    print(f"Feature shape: {feature.shape}")

    import pdb;pdb.set_trace()

    # 特徴量のプロット (例として最初の次元をプロット)
    plt.plot(feature.numpy()[0])  # シーケンス長 0 の特徴量をプロット
    plt.title(f"Feature of data index {idx} with label {label}")
    plt.show()
