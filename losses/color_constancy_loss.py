import torch
import torch.nn as nn
import torch.nn.functional as F

class ColorConstancyLoss(nn.Module):
    """
    色恒常性損失（高速・安定版）
      - RGBバランス: Grey-World（チャネル間の平均差を中性へ）
      - 参照一致: 参照 y の RGB比率に寄せる（yはdetach）
      - 輝度分布: [0,1] 固定レンジでのヒストグラムKL

    Args:
        lambda_cc (float): 損失重み
        bins (int): ヒストグラムのビン数
        eps (float): 数値安定用
        use_ratio (bool): RGB比率項を使うか
    """
    def __init__(self, lambda_cc=10.0, bins=64, eps=1e-6, use_ratio=True):
        super().__init__()
        self.lambda_cc = lambda_cc
        self.bins = bins
        self.eps = eps
        self.use_ratio = use_ratio

        # ITU-R BT.601
        self.register_buffer("lum_w", torch.tensor([0.299, 0.587, 0.114]).view(1,3,1,1))

        # ヒストグラム用ビン境界 [0,1]
        edges = torch.linspace(0., 1., bins + 1)
        self.register_buffer("bin_edges", edges)

    def forward(self, x, y):
        """
        x: 生成画像 [B,3,H,W] （想定: tanh出力で[-1,1] → 下で[0,1]に正規化）
        y: 参照画像 [B,3,H,W] （勾配不要）
        """
        B = x.size(0)

        # [-1,1] -> [0,1]
        x01 = (x + 1) * 0.5
        y01 = (y.detach() + 1) * 0.5

        # --- RGBバランス（Grey-World + 参照比率） ---
        x_mean = x01.mean(dim=[2,3], keepdim=True)  # [B,3,1,1]
        y_mean = y01.mean(dim=[2,3], keepdim=True)  # [B,3,1,1]

        # Grey-World：チャネル間平均の差をゼロへ
        # ex) r-g, g-b, b-r の絶対差の合計
        rw, gw, bw = x_mean[:,0], x_mean[:,1], x_mean[:,2]
        grey_world = (rw - gw).abs() + (gw - bw).abs() + (bw - rw).abs()
        grey_world = grey_world.mean()  # scalar

        # 参照比率: チャネル比率を y に寄せる
        if self.use_ratio:
            x_ratio = x_mean / (x_mean.sum(dim=1, keepdim=True) + self.eps)
            y_ratio = y_mean / (y_mean.sum(dim=1, keepdim=True) + self.eps)
            ratio_loss = F.l1_loss(x_ratio, y_ratio)
        else:
            ratio_loss = x.new_zeros(())

        color_term = grey_world + ratio_loss

        # --- 輝度ヒストグラムのKL ---
        x_gray = (x01 * self.lum_w).sum(dim=1, keepdim=False)  # [B,H,W]
        y_gray = (y01 * self.lum_w).sum(dim=1, keepdim=False)  # [B,H,W]

        # ベクトル化ヒストグラム（bucketize）
        # [B,H*W], 値域は既に[0,1]
        x_flat = x_gray.flatten(1)
        y_flat = y_gray.flatten(1)

        # bin index in [0, bins-1]
        # bucketizeは右端含まない仕様に注意（edgesはbins+1個）
        x_idx = torch.bucketize(x_flat, self.bin_edges) - 1
        y_idx = torch.bucketize(y_flat, self.bin_edges) - 1
        x_idx.clamp_(0, self.bins-1)
        y_idx.clamp_(0, self.bins-1)

        # one-hot
        x_hist = torch.zeros(B, self.bins, device=x.device, dtype=x.dtype)
        y_hist = torch.zeros_like(x_hist)
        x_hist.scatter_add_(1, x_idx, torch.ones_like(x_idx, dtype=x.dtype))
        y_hist.scatter_add_(1, y_idx, torch.ones_like(y_idx, dtype=y.dtype))

        # 確率化 + スムージング
        x_hist.scatter_add_(1, x_idx, torch.ones_like(x_idx, dtype=x.dtype))
        y_hist.scatter_add_(1, y_idx, torch.ones_like(y_idx, dtype=x.dtype))

        # F.kl_div: 入力=log_prob, target=prob
        kl_div = F.kl_div(x_hist.log(), y_hist, reduction="batchmean")

        return self.lambda_cc * (color_term + kl_div)
