import torch
import torch.nn as nn
import torch.nn.functional as F

class ContrastRegularization(nn.Module):
    """
    コントラスト正則化
    
    白内障による輪郭の曖昧化を補正するための正則化項。
    画像勾配の標準偏差を目標値に近づける。
    
    Args:
        lambda_contrast (float): 損失の重み係数
        target_std (float): 目標勾配標準偏差
        eps (float): 数値安定性のための小さな値
        use_reference (bool): 参照画像の標準偏差を目標とするか
    """
    def __init__(self, lambda_contrast=0.05, target_std=0.15, eps=1e-6, use_reference=True):
        super().__init__()
        self.lambda_contrast = lambda_contrast
        self.target_std = target_std
        self.eps = eps
        self.use_reference = use_reference
        
        # Sobelフィルタの定義
        sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32)
        sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=torch.float32)
        
        # フィルタをチャンネル数分拡張 [1, 1, 3, 3]
        self.register_buffer("sobel_x", sobel_x.view(1, 1, 3, 3))
        self.register_buffer("sobel_y", sobel_y.view(1, 1, 3, 3))
    
    def _compute_gradients(self, x):
        """
        Sobelフィルタを使用して勾配を計算
        
        Args:
            x (torch.Tensor): 入力画像 [B, C, H, W]
            
        Returns:
            tuple: x方向とy方向の勾配
        """
        # グレースケールに変換 (簡易版、精密な変換が必要な場合は重み付け)
        if x.size(1) == 3:
            x_gray = x.mean(dim=1, keepdim=True)  # [B, 1, H, W]
        else:
            x_gray = x  # 既にグレースケール
        
        # パディング（境界拡張）
        x_pad = F.pad(x_gray, (1, 1, 1, 1), mode='reflect')
        
        # Sobelフィルタを適用
        grad_x = F.conv2d(x_pad, self.sobel_x)
        grad_y = F.conv2d(x_pad, self.sobel_y)
        
        return grad_x, grad_y
    
    def forward(self, x, y=None):
        """
        Args:
            x (torch.Tensor): 生成画像 [B, 3, H, W]
            y (torch.Tensor, optional): 参照画像 [B, 3, H, W]
            
        Returns:
            torch.Tensor: コントラスト正則化損失値
        """
        # 画像を[0, 1]範囲に正規化（必要な場合）
        if x.min() < 0:
            x = (x + 1) / 2  # [-1, 1] -> [0, 1]
        
        # 勾配計算
        grad_x, grad_y = self._compute_gradients(x)
        
        # 勾配の強度
        grad_magnitude = torch.sqrt(grad_x**2 + grad_y**2 + self.eps)
        
        if self.use_reference and y is not None:
            # 参照画像の勾配標準偏差を目標値とする
            if y.min() < 0:
                y = (y.detach() + 1) / 2  # [-1, 1] -> [0, 1]
            
            ref_grad_x, ref_grad_y = self._compute_gradients(y)
            ref_magnitude = torch.sqrt(ref_grad_x**2 + ref_grad_y**2 + self.eps)
            ref_std = ref_magnitude.std()
            
            # 生成画像と参照画像の勾配標準偏差の差
            loss = F.l1_loss(grad_magnitude.std(), ref_std)
        else:
            # 固定目標値との差
            loss = torch.abs(grad_magnitude.std() - self.target_std)
        
        return self.lambda_contrast * loss


def test_contrast_regularization():
    """単体テスト用関数"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    loss_fn = ContrastRegularization(lambda_contrast=0.05).to(device)
    
    # ランダムな入力画像
    x = torch.randn(2, 3, 256, 256).to(device)  # 生成画像
    y = torch.randn(2, 3, 256, 256).to(device)  # 参照画像（オプション）
    
    # 損失計算
    loss1 = loss_fn(x)  # 参照なし
    loss2 = loss_fn(x, y)  # 参照あり
    
    print(f"Contrast Regularization (fixed target): {loss1.item()}")
    print(f"Contrast Regularization (reference): {loss2.item()}")
    return loss1, loss2


if __name__ == "__main__":
    test_contrast_regularization()