import os
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image

class MisumiImageDataset(Dataset):
    def __init__(self, data_dir, input_size=224, split="train", test_ratio=0.2):
        """
        :param data_dir: すべてのデータディレクトリを含むルートディレクトリ
        :param input_size: 画像の入力サイズ
        :param split: "train" or "test"
        :param test_ratio: テストデータの割合
        """
        self.data_dir = data_dir
        self.input_size = input_size
        self.split = split
        self.test_ratio = test_ratio
        self.data = self.load_images_and_labels()

        # パスからインデックスへのマッピングを作成
        self.path_to_index = {path: idx for idx, (path, _) in enumerate(self.data)}
        self.index_to_path = {idx: path for path, idx in self.path_to_index.items()}

        # データをトレーニングとテストに分割
        self.train_data, self.test_data = self.split_data()

        if self.split == "train":
            self.data = self.train_data
        else:
            self.data = self.test_data

    def load_images_and_labels(self):
        data = []
        print("load_images_and_labels ...")
        for dir_name in os.listdir(self.data_dir):
            dir_path = os.path.join(self.data_dir, dir_name)
            if os.path.isdir(dir_path):
                class_dirs = os.listdir(dir_path)
                for class_idx, class_dir in enumerate(class_dirs):
                    class_path = os.path.join(dir_path, class_dir)
                    if os.path.isdir(class_path):
                        image_files = os.listdir(class_path)
                        for image_file in image_files:
                            image_path = os.path.join(class_path, image_file)
                            # 画像ファイルのみを処理する
                            if image_file.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif')):
                                data.append((image_path, class_idx))

        return data

    def split_data(self):
        # データを層化サンプリングで分割
        total_size = len(self.data)
        test_size = int(total_size * self.test_ratio)
        train_size = total_size - test_size
        return self.data[:train_size], self.data[train_size:]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        image_path, label = self.data[idx]
        image = Image.open(image_path).convert('RGB')

        transform = transforms.Compose([
            transforms.Resize((self.input_size, self.input_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        image = transform(image)
        unique_index = self.path_to_index[image_path]
        return image, label, unique_index  # 一意のインデックスを返す

if __name__ == "__main__":
    # データセットの準備
    data_dir = '/home/kfujii/drawing/data'  # 01-1および01-2ディレクトリを含むルートディレクトリ
    
    # Trainデータセット
    train_dataset = MisumiImageDataset(data_dir=data_dir, split="train")
    print(f"Train dataset size: {len(train_dataset)}")
    
    # Testデータセット
    test_dataset = MisumiImageDataset(data_dir=data_dir, split="test")
    print(f"Test dataset size: {len(test_dataset)}")

    # 一つのデータを取り出す
    idx = 80  # 取り出したいデータのインデックス
    feature, label, unique_index = train_dataset[idx]

    # 特徴量を表示
    print(f"Label: {label}")
    print(f"Feature shape: {feature.shape}")
    print(f"Unique index: {unique_index}")
