import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

class PerceptualLoss(nn.Module):
    """
    VGGベースの知覚損失
    
    事前学習済みVGGネットワークを用いて、生成画像と参照画像の
    高次元特徴量空間での類似性を測定する損失関数。
    
    Args:
        lambda_perceptual (float): 損失の重み係数
        layers (list): 特徴抽出を行うVGGのレイヤー名
        normalize (bool): ImageNet正規化を行うかどうか
    """
    def __init__(self, lambda_perceptual=0.1, 
                 layers=['relu1_2', 'relu2_2', 'relu3_3', 'relu4_3'],
                 normalize=True):
        super().__init__()
        self.lambda_perceptual = lambda_perceptual
        self.normalize = normalize
        
        # VGG19の事前学習済みモデル
        vgg = models.vgg19(pretrained=True).features.eval()
        
        # レイヤー名とインデックスのマッピング
        self.layer_mapping = {
            'relu1_1': 2, 'relu1_2': 4,
            'relu2_1': 7, 'relu2_2': 9,
            'relu3_1': 12, 'relu3_2': 14, 'relu3_3': 16, 'relu3_4': 18,
            'relu4_1': 21, 'relu4_2': 23, 'relu4_3': 25, 'relu4_4': 27,
            'relu5_1': 30, 'relu5_2': 32, 'relu5_3': 34, 'relu5_4': 36
        }
        
        # 指定されたレイヤーまでのスライスを作成
        self.slices = nn.ModuleDict()
        for name in layers:
            if name in self.layer_mapping:
                end_idx = self.layer_mapping[name]
                self.slices[name] = vgg[:end_idx + 1]
            else:
                raise ValueError(f"Unknown layer name: {name}")
        
        # VGGの重みは固定（学習しない）
        for param in self.parameters():
            param.requires_grad = False
        
        # ImageNet正規化用の平均・標準偏差
        self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))
    
    def _normalize(self, x):
        """ImageNet正規化を適用"""
        return (x - self.mean) / self.std
    
    def _preprocess(self, x):
        """前処理: [-1, 1] -> [0, 1] + 正規化（必要な場合）"""
        # [-1, 1] -> [0, 1]
        x = (x + 1) / 2
        # ImageNet正規化
        if self.normalize:
            x = self._normalize(x)
        return x
    
    def forward(self, x, y):
        """
        Args:
            x (torch.Tensor): 生成画像 [B, 3, H, W]
            y (torch.Tensor): 参照画像 [B, 3, H, W]
            
        Returns:
            torch.Tensor: 知覚損失値
        """
        # 前処理
        x = self._preprocess(x)
        y = self._preprocess(y.detach())  # 参照画像は勾配不要
        
        # 各レイヤーの特徴量抽出と損失計算
        loss = 0.0
        for name, slice_model in self.slices.items():
            x_feat = slice_model(x)
            y_feat = slice_model(y)
            
            # 特徴量の形状が異なる場合は調整（必要に応じて）
            if x_feat.shape[2:] != y_feat.shape[2:]:
                x_feat = F.interpolate(x_feat, size=y_feat.shape[2:], mode='bilinear', align_corners=False)
            
            # 特徴量のMSE損失
            loss += F.mse_loss(x_feat, y_feat)
        
        return self.lambda_perceptual * loss


def test_perceptual_loss():
    """単体テスト用関数"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    loss_fn = PerceptualLoss(lambda_perceptual=0.1).to(device)
    
    # ランダムな入力画像
    x = torch.randn(2, 3, 256, 256).to(device)  # 生成画像
    y = torch.randn(2, 3, 256, 256).to(device)  # 参照画像
    
    # 損失計算
    loss = loss_fn(x, y)
    
    print(f"Perceptual Loss: {loss.item()}")
    return loss


if __name__ == "__main__":
    test_perceptual_loss()