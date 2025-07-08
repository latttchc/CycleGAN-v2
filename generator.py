import torch
import torch.nn as nn


class ConvBlock(nn.Module):
    """
    畳み込みブロッククラス
    
    CycleGANのGenerator内で使用される基本的な畳み込みブロック。
    エンコーダ（ダウンサンプリング）とデコーダ（アップサンプリング）の両方に対応。
    
    Args:
        in_channels (int): 入力チャンネル数
        out_channels (int): 出力チャンネル数
        down (bool): Trueならダウンサンプリング（Conv2d）、Falseならアップサンプリング（ConvTranspose2d）
        use_act (bool): 活性化関数を使用するかどうか（ResidualBlockの最後の層では使用しない）
        **kwargs: 畳み込み層に渡される追加パラメータ（kernel_size, stride, padding等）
    """
    def __init__(self, in_channels, out_channels, down=True, use_act=True, **kwargs):
        super().__init__()
        self.conv = nn.Sequential(
            # downがTrueなら通常の畳み込み、Falseなら転置畳み込み
            nn.Conv2d(in_channels, out_channels, padding_mode="reflect", **kwargs)
            if down
            else nn.ConvTranspose2d(in_channels, out_channels, **kwargs),
            # Instance Normalization: バッチ内の各サンプルを個別に正規化
            # CycleGANではBatch Normalizationより効果的とされる
            nn.InstanceNorm2d(out_channels),
            # use_actがTrueならReLU、FalseならIdentity（何もしない）
            nn.ReLU(inplace=True) if use_act else nn.Identity(),
        )

    def forward(self, x):
        """順伝播"""
        return self.conv(x)


class ResidualBlock(nn.Module):
    """
    ResNet風の残差ブロック
    
    スキップ接続により勾配消失問題を軽減し、深いネットワークの学習を安定化。
    CycleGANのGeneratorの中間部分で特徴量の変換を行う。
    
    Args:
        channels (int): 入力・出力チャンネル数（残差ブロックなので同じ）
    """
    def __init__(self, channels):
        super().__init__()
        self.block = nn.Sequential(
            # 最初の畳み込み層（ReLU活性化あり）
            ConvBlock(channels, channels, kernel_size=3, padding=1),
            # 2番目の畳み込み層（ReLU活性化なし）
            # 活性化関数を使わないことで、残差学習がより効果的になる
            ConvBlock(channels, channels, use_act=False, kernel_size=3, padding=1),
        )

    def forward(self, x):
        """
        順伝播：入力xに変換結果を加算（残差接続）
        
        Args:
            x (torch.Tensor): 入力特徴量
            
        Returns:
            torch.Tensor: x + F(x) の形での出力
        """
        return x + self.block(x)


class Generator(nn.Module):
    """
    CycleGAN Generator
    
    U-Net風のエンコーダ-デコーダアーキテクチャにResidualBlockを組み込んだ構造。
    画像から画像への変換を学習する。
    
    ネットワーク構造:
    1. Initial Conv (7x7) - 初期特徴量抽出
    2. Down Blocks (3x3) - エンコーダ部分（特徴量圧縮）
    3. Residual Blocks - 中間変換部分（特徴量変換）
    4. Up Blocks (3x3) - デコーダ部分（特徴量復元）
    5. Final Conv (7x7) - 最終出力層
    
    Args:
        img_channels (int): 入力・出力画像のチャンネル数（RGB=3, グレースケール=1）
        num_features (int): 初期特徴量数（デフォルト64）
        num_residuals (int): ResidualBlockの数（デフォルト9）
    """
    def __init__(self, img_channels, num_features=64, num_residuals=9):
        super().__init__()
        
        # 1. 初期畳み込み層
        # 大きなカーネル（7x7）で広範囲の特徴を捉える
        # reflect paddingで境界のアーティファクトを軽減
        self.initial = nn.Sequential(
            nn.Conv2d(
                img_channels,           # 入力チャンネル数（例：RGB=3）
                num_features,           # 出力チャンネル数（例：64）
                kernel_size=7,          # 大きなカーネルで広範囲の特徴を捉える
                stride=1,               # ストライド1で解像度維持
                padding=3,              # 7x7カーネルでサイズ維持のためpadding=3
                padding_mode="reflect", # リフレクションパディング
            ),
            nn.InstanceNorm2d(num_features),
            nn.ReLU(inplace=True),
        )
        
        # 2. ダウンサンプリング層（エンコーダ）
        # 解像度を下げながら特徴量次元を増やす
        # 256x256 -> 128x128 -> 64x64
        self.down_blocks = nn.ModuleList(
            [
                # 第1ダウンサンプリング: 64 -> 128チャンネル
                ConvBlock(
                    num_features, num_features * 2, kernel_size=3, stride=2, padding=1
                ),
                # 第2ダウンサンプリング: 128 -> 256チャンネル
                ConvBlock(
                    num_features * 2,
                    num_features * 4,
                    kernel_size=3,
                    stride=2,
                    padding=1,
                ),
            ]
        )
        
        # 3. 残差ブロック群
        # 最も深い特徴量レベルで変換を行う
        # 解像度は維持したまま特徴量を変換
        self.res_blocks = nn.Sequential(
            *[ResidualBlock(num_features * 4) for _ in range(num_residuals)]
        )
        
        # 4. アップサンプリング層（デコーダ）
        # 解像度を復元しながら特徴量次元を減らす
        # 64x64 -> 128x128 -> 256x256
        self.up_blocks = nn.ModuleList(
            [
                # 第1アップサンプリング: 256 -> 128チャンネル
                ConvBlock(
                    num_features * 4,
                    num_features * 2,
                    down=False,             # アップサンプリングモード
                    kernel_size=3,
                    stride=2,
                    padding=1,
                    output_padding=1,       # 転置畳み込み用パディング
                ),
                # 第2アップサンプリング: 128 -> 64チャンネル
                ConvBlock(
                    num_features * 2,
                    num_features * 1,
                    down=False,             # アップサンプリングモード
                    kernel_size=3,
                    stride=2,
                    padding=1,
                    output_padding=1,       # 転置畳み込み用パディング
                ),
            ]
        )

        # 5. 最終出力層
        # 特徴量を元の画像チャンネル数に変換
        # 大きなカーネル（7x7）で最終的な詳細を調整
        self.last = nn.Conv2d(
            num_features * 1,       # 入力チャンネル数（64）
            img_channels,           # 出力チャンネル数（例：RGB=3）
            kernel_size=7,          # 大きなカーネルで詳細な出力
            stride=1,
            padding=3,
            padding_mode="reflect",
        )

    def forward(self, x):
        """
        順伝播
        
        Args:
            x (torch.Tensor): 入力画像 [batch_size, img_channels, height, width]
            
        Returns:
            torch.Tensor: 変換された画像 [batch_size, img_channels, height, width]
                         値域は[-1, 1]（tanhにより）
        """
        # 1. 初期特徴量抽出
        x = self.initial(x)
        
        # 2. エンコーダ：ダウンサンプリング
        for layer in self.down_blocks:
            x = layer(x)
        
        # 3. 中間変換：残差ブロック
        x = self.res_blocks(x)
        
        # 4. デコーダ：アップサンプリング
        for layer in self.up_blocks:
            x = layer(x)
        
        # 5. 最終出力：tanh活性化で[-1, 1]に正規化
        # 画像の値域を[-1, 1]にすることで学習の安定化
        return torch.tanh(self.last(x))


def test():
    """
    Generatorのテスト関数
    
    ランダムな入力に対してGeneratorが正常に動作し、
    期待される出力サイズを返すかを確認。
    """
    # テスト用パラメータ
    img_channels = 3        # RGBチャンネル
    img_size = 256         # 256x256の画像
    batch_size = 2         # バッチサイズ
    
    # ランダムな入力テンソル生成
    # 値域[-1, 1]を想定（実際の前処理と合わせる）
    x = torch.randn((batch_size, img_channels, img_size, img_size))
    
    # Generatorインスタンス作成
    # num_features=9は誤り（通常64等の2の累乗）、ここでは動作確認のため
    gen = Generator(img_channels, 9)
    
    # 順伝播実行
    output = gen(x)
    
    # 出力サイズ確認
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Output value range: [{output.min():.3f}, {output.max():.3f}]")
    
    # 期待される出力: [2, 3, 256, 256]
    assert output.shape == x.shape, f"Shape mismatch: expected {x.shape}, got {output.shape}"
    print("✓ Generator test passed!")


if __name__ == "__main__":
    test()