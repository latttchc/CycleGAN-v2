import torch
import torch.nn as nn


class ContrastRegularization(nn.Module):
    """
    コントラスト正則化
    
    白内障によるコントラスト低下を補正。
    
    Args:
        lambda_contrast (float): コントラスト正則化の重み
        target_std (float): 目標勾配標準偏差
    """
    def __init__(self, lambda_contrast=0.05, target_std=0.15):
        super().__init__()
        self.lambda_contrast = lambda_contrast
        self.target_std = target_std
    
    def compute_gradients(self, image):
        """Sobel filterで勾配を計算"""
        grad_x = image[:, :, :-1, :] - image[:, :, 1:, :]
        grad_y = image[:, :, :, :-1] - image[:, :, :, 1:]
        return grad_x, grad_y
    
    def forward(self, image):
        """
        コントラスト正則化損失を計算
        
        Args:
            image (torch.Tensor): [B, 3, H, W]
        
        Returns:
            torch.Tensor: コントラスト正則化損失
        """
        grad_x, grad_y = self.compute_gradients(image)
        grad_std = torch.sqrt(torch.var(grad_x) + torch.var(grad_y))
        loss = torch.abs(grad_std - self.target_std)
        
        return self.lambda_contrast * loss