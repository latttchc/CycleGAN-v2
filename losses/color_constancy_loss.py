import torch
import torch.nn as nn
import torch.nn.functional as F

class ColorConstancyLoss(nn.Module):
    """
    色恒常性損失
    
    白内障によるモネの色彩認識変化を補正するための損失関数。
    RGBチャンネルの分布とヒストグラムを調整する。
    
    Args:
        lambda_cc (float): 損失の重み係数
        bins (int): 輝度ヒストグラムのビン数
        eps (float): 数値安定性のための小さな値
    """
    def __init__(self, lambda_cc=10.0, bins=64, eps=1e-6):
        super().__init__()
        self.lambda_cc = lambda_cc
        self.bins = bins
        self.eps = eps
        
        # 輝度変換用の重み（ITU-R BT.601）
        self.register_buffer("lum_weights", torch.tensor([0.299, 0.587, 0.114]).view(1, 3, 1, 1))
        
        # ヒストグラム用ビンの境界値 [0, 1]
        edges = torch.linspace(0., 1., bins + 1)
        self.register_buffer("bin_edges", edges)
    
    def forward(self, x, y):
        """
        Args:
            x (torch.Tensor): 生成画像 [B, 3, H, W]（通常は[-1, 1]範囲）
            y (torch.Tensor): 参照画像 [B, 3, H, W]（通常は[-1, 1]範囲）
            
        Returns:
            torch.Tensor: 色恒常性損失値
        """
        B = x.size(0)
        
        # [-1, 1] -> [0, 1]に正規化（tanhの出力範囲を想定）
        x01 = (x + 1) * 0.5
        y01 = (y.detach() + 1) * 0.5  # 参照画像は勾配不要
        
        # ===== RGBバランス損失 =====
        # チャンネル平均値の計算
        x_mean = x01.mean(dim=[2, 3], keepdim=True)  # [B, 3, 1, 1]
        y_mean = y01.mean(dim=[2, 3], keepdim=True)  # [B, 3, 1, 1]
        
        # 1. Grey-World仮説：各チャンネルの平均値が等しくなるべき
        x_r, x_g, x_b = x_mean[:, 0], x_mean[:, 1], x_mean[:, 2]
        grey_world_loss = (torch.abs(x_r - x_g) + torch.abs(x_g - x_b) + torch.abs(x_b - x_r)).mean()
        
        # 2. 参照画像のRGB比率に合わせる
        x_ratio = x_mean / (x_mean.sum(dim=1, keepdim=True) + self.eps)
        y_ratio = y_mean / (y_mean.sum(dim=1, keepdim=True) + self.eps)
        ratio_loss = F.l1_loss(x_ratio, y_ratio)
        
        rgb_balance_loss = grey_world_loss + ratio_loss
        
        # ===== 輝度ヒストグラム損失 =====
        # グレースケール輝度の計算
        x_lum = torch.sum(x01 * self.lum_weights, dim=1)  # [B, H, W]
        y_lum = torch.sum(y01 * self.lum_weights, dim=1)  # [B, H, W]
        
        # ヒストグラム計算のためにフラット化
        x_flat = x_lum.flatten(1)  # [B, H*W]
        y_flat = y_lum.flatten(1)  # [B, H*W]
        
        # binのインデックスを計算（bucketize）
        x_idx = torch.bucketize(x_flat, self.bin_edges) - 1  # [B, H*W]
        y_idx = torch.bucketize(y_flat, self.bin_edges) - 1  # [B, H*W]
        
        # 範囲外の値をクリップ
        x_idx = x_idx.clamp(0, self.bins - 1)
        y_idx = y_idx.clamp(0, self.bins - 1)
        
        # ヒストグラムの計算
        x_hist = torch.zeros(B, self.bins, device=x.device, dtype=x.dtype)
        y_hist = torch.zeros_like(x_hist)
        
        # scatter_add_で各binをカウント
        x_hist.scatter_add_(1, x_idx, torch.ones_like(x_idx, dtype=x.dtype))
        y_hist.scatter_add_(1, y_idx, torch.ones_like(y_idx, dtype=x.dtype))
        
        # 確率分布に正規化（スムージング付き）
        x_hist = (x_hist + self.eps) / (x_hist.sum(dim=1, keepdim=True) + self.eps * self.bins)
        y_hist = (y_hist + self.eps) / (y_hist.sum(dim=1, keepdim=True) + self.eps * self.bins)
        
        # KLダイバージェンス計算（F.kl_divは入力がlog_probの場合）
        histogram_loss = F.kl_div(x_hist.log(), y_hist, reduction="batchmean")
        
        # 最終的な損失（重み付き合計）
        return self.lambda_cc * (rgb_balance_loss + histogram_loss)


def test_color_constancy_loss():
    """単体テスト用関数"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    loss_fn = ColorConstancyLoss(lambda_cc=10.0).to(device)
    
    # ランダムな入力画像
    x = torch.randn(2, 3, 256, 256).to(device)  # 生成画像
    y = torch.randn(2, 3, 256, 256).to(device)  # 参照画像
    
    # 損失計算
    loss = loss_fn(x, y)
    
    print(f"Color Constancy Loss: {loss.item()}")
    return loss


if __name__ == "__main__":
    test_color_constancy_loss()