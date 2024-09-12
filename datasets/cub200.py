import os
import numpy as np
from torchvision import datasets
try:
    from .base_dataset import BaseDataset
except: 
    from base_dataset import BaseDataset


class Cub200Dataset(BaseDataset):

    def __init__(self, *args, **kwargs):
        super(Cub200Dataset, self).__init__(*args, **kwargs)
        assert self.split in {"train", "test"}

    def set_paths_and_labels(self):

        dataset = datasets.ImageFolder(os.path.join(self.data_dir, 'images'))
        paths = np.array([a for (a, b) in dataset.imgs])
        labels = np.array([b for (a, b) in dataset.imgs])
        sorted_lb = list(sorted(set(labels)))
        if self.split == "train":
            set_labels = set(sorted_lb[:len(sorted_lb) // 2])
        else:
            set_labels = set(sorted_lb[len(sorted_lb) // 2:])
        self.paths = []
        self.labels = []
        for lb, pth in zip(labels, paths):
            if lb in set_labels:
                self.paths.append(pth)
                self.labels.append(lb)

if __name__ == "__main__":
    # データセットの準備
    data_dir = 'data/CUB_200_2011'  # ここにCUB200データセットへのパスを指定
    input_size = 224  # 入力画像サイズ
    dataset = Cub200Dataset(data_dir=data_dir, input_size=input_size, split="train")

    # 一枚のデータを取り出す
    idx = 0  # 取り出したいデータのインデックスを指定
    image, label = dataset[idx]

    # 画像の表示
    img = np.transpose(image.numpy(), (1, 2, 0))  # (C, H, W) -> (H, W, C) に変換
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    img = std * img + mean  # 正規化を元に戻す
    img = np.clip(img, 0, 1)
    import pdb; pdb.set_trace()