import os
from collections import defaultdict
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
from sklearn.model_selection import train_test_split


class MisumiImageDataset(Dataset):
    def __init__(self, data_dir, input_size=224, split="train", test_ratio=0.2):
        self.data_dir = data_dir
        self.input_size = input_size
        self.split = split
        self.test_ratio = test_ratio
        self.data = self.load_images_and_labels()

        # クラス名から数値ラベルへのマッピングを作成
        class_names = {os.path.basename(os.path.dirname(path)) for path, _ in self.data}
        self.class_to_index = {class_name: idx for idx, class_name in enumerate(sorted(class_names))}
        self.index_to_class = {idx: class_name for class_name, idx in self.class_to_index.items()}

        # クラスごとに分割
        self.train_data, self.test_data = self.split_data_per_class()

        # パスとインデックスのマッピングを作成
        if self.split == "train":
            self.data = self.train_data
        else:
            self.data = self.test_data
        self.index_to_path = {idx: path for idx, (path, _) in enumerate(self.data)}

    def load_images_and_labels(self):
        data = []
        for dir_name in os.listdir(self.data_dir):
            dir_path = os.path.join(self.data_dir, dir_name)
            if os.path.isdir(dir_path):
                image_files = os.listdir(dir_path)
                for image_file in image_files:
                    image_path = os.path.join(dir_path, image_file)
                    if image_file.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif')):
                        class_name = os.path.basename(dir_name)
                        data.append((image_path, class_name))
        return data

    def split_data_per_class(self):
        class_to_images = defaultdict(list)

        for image_path, class_name in self.data:
            class_to_images[class_name].append(image_path)

        train_data = []
        test_data = []

        for class_name, image_paths in class_to_images.items():
            train_paths, test_paths = train_test_split(image_paths, test_size=self.test_ratio, random_state=42)
            train_data.extend([(path, class_name) for path in train_paths])
            test_data.extend([(path, class_name) for path in test_paths])

        return train_data, test_data

    def __getitem__(self, idx):
        image_path, class_name = self.data[idx]
        image = Image.open(image_path).convert('RGB')

        transform = transforms.Compose([
            transforms.Resize((self.input_size, self.input_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        image = transform(image)
        class_label = self.class_to_index[class_name]  # クラス名を数値ラベルに変換
        relative_path = os.path.relpath(image_path, self.data_dir)  # 相対パスを取得
        return image, class_label, relative_path

    def __len__(self):
        return len(self.data)


if __name__ == "__main__":
    # データセットの準備
    data_dir = '/home/kfujii/drawing/data'  # 01-1および01-2ディレクトリを含むルートディレクトリ

    # Trainデータセット
    train_dataset = MisumiImageDataset(data_dir=data_dir, split="train")
    print(f"Train dataset size: {len(train_dataset)}")

    # Testデータセット
    test_dataset = MisumiImageDataset(data_dir=data_dir, split="test")
    print(f"Test dataset size: {len(test_dataset)}")

    # クラスごとの分割結果を確認
    from collections import Counter

    train_labels = [label for _, label, _ in train_dataset]
    test_labels = [label for _, label, _ in test_dataset]

    train_label_counts = Counter(train_labels)
    test_label_counts = Counter(test_labels)

    print("Train label counts:")
    for label, count in sorted(train_label_counts.items()):
        print(f"Class {label}: {count} images")

    print("\nTest label counts:")
    for label, count in sorted(test_label_counts.items()):
        print(f"Class {label}: {count} images")

    # 分割割合が8:2になっているか検証
    for label in train_label_counts.keys():
        train_count = train_label_counts[label]
        test_count = test_label_counts.get(label, 0)  # テストデータにないクラスがある場合を考慮
        total = train_count + test_count
        print(f"Class {label}: Train/Test = {train_count}/{test_count} (Total: {total}, Ratio: {train_count/total:.2f})")

    # インデックスの確認
    sample_idx = 10
    _, label, unique_index = train_dataset[sample_idx]
    print(f"Sample {sample_idx}: Label={label}, Unique Index={unique_index}")
