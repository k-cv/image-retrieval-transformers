import torch
from torch.utils.data import Dataset
import numpy as np
from sklearn.model_selection import train_test_split

class MisumiFeatureDataset(Dataset):
    def __init__(self, data_dir, split="train", ignore_label=-1, test_ratio=0.2):
        """
        :param data_dir: 特徴量とラベルが保存されている .pth ファイル
        :param split: "train" or "test"
        :param ignore_label: ラベルが指定されていない場合の値（デフォルトは -1）
        :param test_ratio: テストデータに使用する割合 (デフォルトは後半10%)
        """
        self.data_dir = data_dir
        self.split = split
        self.ignore_label = ignore_label
        self.test_ratio = test_ratio
        self.features, self.labels = self.load_features_and_labels()

        # データを層化サンプリングで分割
        self.train_data, self.test_data = self.split_data_stratified()

        # 指定されたsplitに従ってデータを選択
        if self.split == "train":
            self.data = self.train_data
        elif self.split == "test":
            self.data = self.test_data
        else:
            raise ValueError("splitは'train'または'test'である必要があります")

        # 有効なデータを取得
        self.features, self.labels = zip(*self.data)

        assert len(self.features) == len(self.labels), "有効な特徴量とラベルの数が一致しません"

    def load_features_and_labels(self):
        # 特徴量とラベルのペアが保存されている .pth ファイルをロードする
        print("特徴量とラベルのペアを読み込み中...")
        data = torch.load(self.data_dir)  # {'features': テンソル, 'labels': ラベル}
        features = data['features']  # [データ数, シーケンス長, 256]
        labels = data['labels']      # [データ数]
        print("特徴量とラベルのペアを読み込み完了しました")
        return features, labels

    def split_data_stratified(self):
        labels_np = np.array(self.labels)

        # 各クラスのサンプル数を確認
        unique_labels, label_counts = np.unique(labels_np, return_counts=True)
        
        train_data = []
        test_data = []

        # 各クラスごとにデータを分割
        for label in unique_labels:
            # 該当するラベルを持つインデックスを取得
            label_indices = np.where(labels_np == label)[0]

            if len(label_indices) < 2:
                # クラスのサンプル数が1つの場合は全てをトレーニングデータに
                print(f"Label {label} has only {len(label_indices)} sample(s), assigning to train set.")
                for idx in label_indices:
                    train_data.append((self.features[idx], self.labels[idx]))
            else:
                # サンプル数が2つ以上の場合、層化サンプリングで分割
                train_indices, test_indices = train_test_split(
                    label_indices,
                    test_size=self.test_ratio,
                    stratify=labels_np[label_indices]  # 層化サンプリング
                )
                for idx in train_indices:
                    train_data.append((self.features[idx], self.labels[idx]))
                for idx in test_indices:
                    test_data.append((self.features[idx], self.labels[idx]))

        print(f"Train data count: {len(train_data)}, Test data count: {len(test_data)}")
        return train_data, test_data

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        # 特徴量とラベルを取得する
        feature = self.features[idx]  # [シーケンス長, 256]
        label = self.labels[idx]

        # シーケンス長を256に調整
        if feature.shape[0] < 256:
            padding = torch.zeros(256 - feature.shape[0], feature.shape[1])
            feature = torch.cat([feature, padding], dim=0)
        elif feature.shape[0] > 256:
            feature = feature[:256, :]

        return feature.float(), label

    def __repr__(self):
        return f"{self.__class__.__name__}(split={self.split}, num_samples={len(self)})"


if __name__ == "__main__":
    # データセットの準備
    data_dir = '/home/kfujii/vitruvion/outputs/2024-10-21/14-40-41/features_labels.pth'  # 特徴量とラベルのペアが保存されているファイル
    
    # Trainデータセット
    train_dataset = MisumiFeatureDataset(data_dir=data_dir, split="train")
    print(f"Train dataset size: {len(train_dataset)}")
    
    # Testデータセット
    test_dataset = MisumiFeatureDataset(data_dir=data_dir, split="test")
    print(f"Test dataset size: {len(test_dataset)}")

    # 一つのデータを取り出す
    idx = 80  # 取り出したいデータのインデックス
    feature, label = train_dataset[idx]

    # 特徴量を表示
    print(f"Label: {label}")
    print(f"Feature shape: {feature.shape}")
