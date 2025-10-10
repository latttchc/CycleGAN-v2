import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import vgg19, VGG19_Weights

class PerceptualLoss(nn.Module):
    """
    VGG19ベースの知覚損失（単一forwardで複数層を回収）
    layers: 取得したい層のインデックス（featuresのインデックス）
    """
    def __init__(self, lambda_perceptual=0.1, layers=(4,9,18,27)):
        super().__init__()
        self.lambda_perceptual = lambda_perceptual
        weights = VGG19_Weights.IMAGENET1K_FEATURES
        vgg = vgg19(weights=weights).features.eval()
        for p in vgg.parameters():
            p.requires_grad = False
        self.vgg = vgg
        self.layers = set(layers)

        # 正規化のためのTransformパラメータ
        mean = torch.tensor(weights.meta["mean"]).view(1,3,1,1)
        std  = torch.tensor(weights.meta["std"]).view(1,3,1,1)
        self.register_buffer("mean", mean)
        self.register_buffer("std", std)

    def _preprocess(self, x):
        # [-1,1] -> [0,1] -> ImageNet 正規化
        x = (x + 1) * 0.5
        return (x - self.mean) / self.std

    def _features(self, x):
        feats = {}
        h = x
        for i, m in enumerate(self.vgg):
            h = m(h)
            if i in self.layers:
                feats[i] = h
        return feats

    def forward(self, generated, target):
        g = self._preprocess(generated)
        t = self._preprocess(target.detach())   # 参照は参照。勾配不要

        g_feats = self._features(g)
        t_feats = self._features(t)

        loss = 0.0
        for i in self.layers:
            loss = loss + F.mse_loss(g_feats[i], t_feats[i])
        return self.lambda_perceptual * loss
