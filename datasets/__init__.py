from .cub200 import Cub200Dataset
from .sop import SOPDataset
from .inshop import InShopDataset
from .cad import CADImageDataset
from .cadc import CADFeatureDataset


def get_dataset(args):

    """returns train, query and gallery dataset"""

    train, query, gallery = None, None, None

    if args.dataset == 'cub200':
        train   = Cub200Dataset(args.data_path, split="train")
        query   = Cub200Dataset(args.data_path, split="test")

    if args.dataset == 'sop':
        train   = SOPDataset(args.data_path, split="train")
        query   = SOPDataset(args.data_path, split="test")

    if args.dataset == 'inshop':
        train   = InShopDataset(args.data_path, split="train")
        query   = InShopDataset(args.data_path, split="query")
        gallery = InShopDataset(args.data_path, split="gallery")

    # CADデータセットの追加
    if args.dataset == 'cad':
        train   = CADImageDataset(
            label_file='/home/kfujii/image-retrieval-transformers/data/CAD/label.txt',
            data_dir='/home/kfujii/vitruvion/outputs/2024-09-05/12-54-06_all_images',
            input_size=args.input_size,
            split="train"
        )
        query   = CADImageDataset(
            label_file='/home/kfujii/image-retrieval-transformers/data/CAD/label.txt',
            data_dir='/home/kfujii/vitruvion/outputs/2024-09-05/12-54-06_all_images',
            input_size=args.input_size,
            split="test"
        )
        gallery = None  # CADデータセットではギャラリーデータが不要な場合

    if args.dataset == 'cadc':
        train = CADFeatureDataset(
            label_file='/home/kfujii/image-retrieval-transformers/data/CAD/label.txt',
            data_dir ='/home/kfujii/vitruvion/encoder_features2.pth',
            
            split="train"
        )
        query = CADFeatureDataset(
            label_file='/home/kfujii/image-retrieval-transformers/data/CAD/label.txt',
            data_dir ='/home/kfujii/vitruvion/encoder_features2.pth',
            
            split="test"
        )

    return train, query, gallery
