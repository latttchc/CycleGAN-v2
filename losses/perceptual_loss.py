import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models


class PerceptualLoss(nn.Module):
    """
    VGGベースの知覚損失
    
    高次元特徴空間での類似性を計測し、モネの筆致・質感を保持。
    
    Args:
        lambda_perceptual (float): 知覚損失の重み
        layers (list): 使用するVGGレイヤー名
    
    References:
        - Johnson et al., "Perceptual Losses for Real-Time Style Transfer" (ECCV 2016)
    """
    def __init__(self, lambda_perceptual=0.1, layers=['relu1_2', 'relu2_2', 'relu3_3', 'relu4_3']):
        super().__init__()
        self.lambda_perceptual = lambda_perceptual
        
        # VGG19の事前学習済みモデル
        vgg = models.vgg19(pretrained=True).features.eval()
        
        # 指定レイヤーまでの特徴抽出器
        self.slices = nn.ModuleDict()
        layer_map = {'relu1_2': 4, 'relu2_2': 9, 'relu3_3': 18, 'relu4_3': 27}
        
        for name in layers:
            self.slices[name] = vgg[:layer_map[name]]
        
        # 勾配計算を無効化
        for param in self.parameters():
            param.requires_grad = False
    
    def extract_features(self, x):
        """VGG特徴量を抽出"""
        features = {}
        for name, slice_model in self.slices.items():
            features[name] = slice_model(x)
        return features
    
    def forward(self, generated, target):
        """
        知覚損失を計算
        
        Args:
            generated (torch.Tensor): 生成画像 [B, 3, H, W]
            target (torch.Tensor): ターゲット画像 [B, 3, H, W]
        
        Returns:
            torch.Tensor: 知覚損失
        """
        # [-1, 1] -> [0, 1] に正規化
        generated = (generated + 1) / 2
        target = (target + 1) / 2
        
        # ImageNet正規化
        mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).to(generated.device)
        std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).to(generated.device)
        
        generated = (generated - mean) / std
        target = (target - mean) / std
        
        # 特徴量抽出
        gen_features = self.extract_features(generated)
        tgt_features = self.extract_features(target)
        
        # レイヤーごとの損失を計算
        loss = 0
        for name in self.slices.keys():
            loss += F.mse_loss(gen_features[name], tgt_features[name])
        
        return self.lambda_perceptual * loss