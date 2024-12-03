import torch
import torch.nn as nn
from timm.models import create_model

class CombinedModel(nn.Module):
    def __init__(self, base_model, pretrained, input_size, geo_feature_dim, drop_rate=0.0, drop_path_rate=0.1):
        super(CombinedModel, self).__init__()
        
        # ベースモデル（Vision Transformer）
        self.base_model = create_model(
            base_model,
            pretrained=pretrained,
            num_classes=0,
            drop_rate=drop_rate,
            drop_path_rate=drop_path_rate,
        )

        # ベースモデルの出力次元を取得
        self.embed_dim = self.base_model.embed_dim  # 例: 384

        # 幾何特徴を処理するための全結合層（入力次元をgeo_feature_dimから修正）
        self.geo_fc = nn.Linear(geo_feature_dim, self.embed_dim)

        # 画像特徴と幾何特徴を統合するための層
        self.fusion_fc = nn.Linear(self.embed_dim + self.embed_dim, self.embed_dim)

    def forward(self, images, geo_features):
        # ベースモデルで画像特徴を抽出
        image_features = self.base_model(images)

        if isinstance(image_features, tuple):
            image_features = image_features[0]

        # geo_features のシーケンス次元を平均化して 2 次元に変換
        geo_features = geo_features.mean(dim=1)  # [batch_size, geo_feature_dim]

        # geo_fc を適用して幾何特徴を embed_dim に変換
        geo_features = self.geo_fc(geo_features)  # [batch_size, embed_dim]

        # 画像特徴と幾何特徴を統合
        combined_features = torch.cat((image_features, geo_features), dim=1)
        output = self.fusion_fc(combined_features)

        return output
