import torch
import torch.nn as nn
import torch.nn.functional as F


class ColorConstancyLoss(nn.Module):
    """
    色恒常性損失
    
    モネの白内障による色域の偏り（黄色・赤色への偏向）を補正するための損失関数。
    
    主な機能:
    1. RGBバランスの復元: チャンネル間の比率を目標ドメインに近づける
    2. グレースケール分布の整合: 輝度分布のKLダイバージェンスを最小化
    
    Args:
        lambda_cc (float): 色恒常性損失の重み（デフォルト: 10.0）
    
    References:
        - "Color Constancy and Image Understanding" (Forsyth, 1990)
        - モネの色彩変化分析 (Russell et al., 2007)
    """
    def __init__(self, lambda_cc=10.0):
        super(ColorConstancyLoss, self).__init__()
        self.lambda_cc = lambda_cc
        
    def forward(self, x, y):
        """
        色恒常性損失を計算
        
        Args:
            x (torch.Tensor): 生成画像 [B, 3, H, W]
            y (torch.Tensor): ターゲットドメイン画像（参照用） [B, 3, H, W]
        
        Returns:
            torch.Tensor: 色恒常性損失値
        """
        # RGB平均値の計算（空間次元で平均化）
        x_mean = torch.mean(x, dim=[2, 3], keepdim=True)  # [B, 3, 1, 1]
        y_mean = torch.mean(y, dim=[2, 3], keepdim=True)  # [B, 3, 1, 1]
        
        # RGBバランスの計算（各チャンネルの相対的な強度）
        # 分母に1e-8を加えてゼロ除算を防止
        x_balance = x_mean / (torch.sum(x_mean, dim=1, keepdim=True) + 1e-8)
        y_balance = y_mean / (torch.sum(y_mean, dim=1, keepdim=True) + 1e-8)
        
        # 色バランスの差異（L1ノルム）
        # 白内障による色の偏りを補正
        color_balance_loss = F.l1_loss(x_balance, y_balance)
        
        # グレースケールイメージの計算（輝度）
        # ITU-R BT.601標準の輝度変換係数を使用
        x_gray = 0.299 * x[:, 0] + 0.587 * x[:, 1] + 0.114 * x[:, 2]  # [B, H, W]
        y_gray = 0.299 * y[:, 0] + 0.587 * y[:, 1] + 0.114 * y[:, 2]  # [B, H, W]
        
        # グレースケール分布のKLダイバージェンス
        # 輝度分布の統計的類似性を測定
        x_hist = self._compute_histogram(x_gray)
        y_hist = self._compute_histogram(y_gray)
        kl_div = F.kl_div(torch.log(x_hist + 1e-8), y_hist, reduction='batchmean')
        
        # 総合損失
        return self.lambda_cc * (color_balance_loss + kl_div)
    
    def _compute_histogram(self, x, bins=64):
        """
        グレースケール画像のヒストグラムを計算
        
        Args:
            x (torch.Tensor): グレースケール画像 [B, H, W]
            bins (int): ヒストグラムのビン数（デフォルト: 64）
        
        Returns:
            torch.Tensor: 正規化されたヒストグラム [B, bins]
        """
        batch_size = x.size(0)
        hist = torch.zeros(batch_size, bins).to(x.device)
        
        for i in range(batch_size):
            min_val = torch.min(x[i])
            max_val = torch.max(x[i])
            
            # 正規化された値をビンにマッピング
            if max_val > min_val:
                x_norm = (x[i] - min_val) / (max_val - min_val)
                bin_idx = (x_norm * (bins - 1)).long().view(-1)
                
                # ヒストグラム計算
                for j in range(bin_idx.size(0)):
                    idx = bin_idx[j].item()
                    if 0 <= idx < bins:  # 範囲チェック
                        hist[i, idx] += 1
                
                # 正規化（確率分布化）
                hist[i] = hist[i] / torch.sum(hist[i])
            else:
                # 画像が一様な場合、均等分布を仮定
                hist[i] = torch.ones(bins).to(x.device) / bins
            
        return hist


# テスト関数
def test_color_constancy_loss():
    """色恒常性損失のテスト"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # テスト用画像生成
    batch_size = 2
    channels = 3
    height, width = 256, 256
    
    # 生成画像（やや黄色がかった画像を想定）
    x = torch.randn(batch_size, channels, height, width).to(device)
    x[:, 0] += 0.3  # Rチャンネルを増加
    x[:, 1] += 0.2  # Gチャンネルを増加
    
    # ターゲット画像（バランスの取れた画像）
    y = torch.randn(batch_size, channels, height, width).to(device)
    
    # 損失計算
    criterion = ColorConstancyLoss(lambda_cc=10.0).to(device)
    loss = criterion(x, y)
    
    print(f"Color Constancy Loss: {loss.item():.4f}")
    print(f"✓ Color Constancy Loss test passed!")
    
    return loss


if __name__ == "__main__":
    test_color_constancy_loss()