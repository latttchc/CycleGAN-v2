import torch
import torch.nn as nn
import torch.nn.functional as F

class ContrastRegularization(nn.Module):
    """
    コントラスト正則化（サンプル毎の勾配標準偏差をターゲットへ）
    Args:
        lambda_contrast (float)
        target_std (float)
        reduction (str): 'mean' or 'none'
    """
    def __init__(self, lambda_contrast=0.05, target_std=0.15, reduction='mean', eps=1e-6):
        super().__init__()
        self.lambda_contrast = lambda_contrast
        self.target_std = target_std
        self.reduction = reduction
        self.eps = eps

    @staticmethod
    def _finite_diff(image):
        # [B,3,H,W] → x/y勾配
        gx = image[:, :, 1:, :] - image[:, :, :-1, :]
        gy = image[:, :, :, 1:] - image[:, :, :, :-1]
        return gx, gy

    def forward(self, image):
        gx, gy = self._finite_diff(image)
        # 各サンプルの勾配強度
        gmag = torch.sqrt(gx.pow(2).mean(dim=(1,2,3)) + gy.pow(2).mean(dim=(1,2,3)) + self.eps)  # [B]
        # 目標へ（Huber）
        loss = F.smooth_l1_loss(gmag, image.new_full(gmag.shape, self.target_std), reduction='none')
        loss = loss.mean() if self.reduction == 'mean' else loss
        return self.lambda_contrast * loss
