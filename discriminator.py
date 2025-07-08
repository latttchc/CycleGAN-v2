import torch
import torch.nn as nn


class Block(nn.Module):
    """
    Discriminatorの基本ブロック
    
    PatchGANアーキテクチャの構成要素として使用される畳み込みブロック。
    各ブロックは畳み込み、正規化、活性化関数から構成される。
    
    Args:
        in_channels (int): 入力チャンネル数
        out_channels (int): 出力チャンネル数
        stride (int): 畳み込みのストライド（2で解像度半分、1で解像度維持）
    """
    def __init__(self, in_channels, out_channels, stride):
        super().__init__()
        self.conv = nn.Sequential(
            # 4x4カーネルの畳み込み層
            # - kernel_size=4: PatchGANで一般的なカーネルサイズ
            # - bias=True: Instance Normの後でもバイアスを使用
            # - padding_mode="reflect": 境界でのアーティファクトを軽減
            nn.Conv2d(
                in_channels, 
                out_channels, 
                kernel_size=4,          # 4x4カーネル
                stride=stride,          # ストライド（1 or 2）
                padding=1,              # パディング1でサイズ調整
                bias=True,              # バイアス項を使用
                padding_mode="reflect"  # リフレクションパディング
            ),
            # Instance Normalization
            # - 各チャンネルを独立して正規化
            # - Batch Normより安定した学習が可能
            # - CycleGANでは標準的な選択
            nn.InstanceNorm2d(out_channels),
            # LeakyReLU活性化関数
            # - negative_slope=0.2: 負の値も少し通す
            # - 通常のReLUより勾配消失を防ぐ
            # - GANでは一般的な選択
            nn.LeakyReLU(0.2),
        )

    def forward(self, x):
        """
        順伝播
        
        Args:
            x (torch.Tensor): 入力特徴量
            
        Returns:
            torch.Tensor: 変換された特徴量
        """
        return self.conv(x)


class Discriminator(nn.Module):
    """
    CycleGAN Discriminator (PatchGAN)
    
    70×70パッチレベルでの画像判別を行うPatchGANアーキテクチャ。
    画像全体ではなく、局所的なパッチが本物か偽物かを判別する。
    これにより高周波の詳細な特徴を捉えることができる。
    
    ネットワーク構造:
    1. Initial Conv - 最初の畳み込み（正規化なし）
    2. Down Blocks - 解像度を下げながら特徴量を抽出
    3. Final Conv - 最終的な判別スコア出力
    
    出力:
    - 各位置での真偽判別スコア（0-1）
    - 最終的にはすべての位置の平均を取る
    
    Args:
        in_channels (int): 入力画像のチャンネル数（RGB=3, グレースケール=1）
        features (list): 各層の特徴量数のリスト [64, 128, 256, 512]
    """
    def __init__(self, in_channels=3, features=[64, 128, 256, 512]):
        super().__init__()
        
        # 初期畳み込み層
        # 最初の層のみInstance Normalizationを使用しない
        # これはPatchGANの標準的な設計パターン
        self.initial = nn.Sequential(
            nn.Conv2d(
                in_channels,            # 入力チャンネル数（例：RGB=3）
                features[0],            # 出力チャンネル数（例：64）
                kernel_size=4,          # 4x4カーネル
                stride=2,               # ストライド2で解像度半分
                padding=1,              # パディング1
                padding_mode="reflect", # リフレクションパディング
            ),
            # 最初の層では正規化なし
            # LeakyReLU活性化のみ
            nn.LeakyReLU(0.2),
        )
        
        # 中間層の構築
        # features[1:]に対して順次ブロックを追加
        layers = []
        in_channels = features[0]  # 現在のチャンネル数を追跡
        
        for feature in features[1:]:
            # 最後の層以外はstride=2（解像度半分）
            # 最後の層はstride=1（解像度維持）
            stride = 1 if feature == features[-1] else 2
            
            layers.append(Block(in_channels, feature, stride=stride))
            in_channels = feature  # 次の層の入力チャンネル数を更新
        
        # 最終出力層
        # 特徴量を1チャンネルの判別スコアに変換
        layers.append(
            nn.Conv2d(
                in_channels,            # 最後の特徴量数（例：512）
                1,                      # 出力1チャンネル（判別スコア）
                kernel_size=4,          # 4x4カーネル
                stride=1,               # ストライド1（解像度維持）
                padding=1,              # パディング1
                padding_mode="reflect"  # リフレクションパディング
            )
        )
        
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        """
        順伝播
        
        入力画像に対してパッチレベルでの真偽判別を行う。
        
        Args:
            x (torch.Tensor): 入力画像 [batch_size, channels, height, width]
                             例: [5, 3, 256, 256]
            
        Returns:
            torch.Tensor: 判別スコア [batch_size, 1, patch_h, patch_w]
                         例: [5, 1, 30, 30] (30x30パッチでの判別結果)
                         
        処理フロー:
        1. 256x256 -> 128x128 (initial, stride=2)
        2. 128x128 -> 64x64   (block1, stride=2) 
        3. 64x64 -> 32x32     (block2, stride=2)
        4. 32x32 -> 30x30     (block3, stride=1, final conv)
        
        最終的に30x30の各位置が70x70パッチの判別結果に対応
        """
        # 初期特徴量抽出
        x = self.initial(x)
        
        # 中間層での特徴量変換と最終判別
        x = self.model(x)
        
        # Sigmoid活性化で[0, 1]の確率値に変換
        # 1に近いほど本物、0に近いほど偽物と判別
        return torch.sigmoid(x)


def test():
    """
    Discriminatorのテスト関数
    
    ランダムな入力画像に対してDiscriminatorが正常に動作し、
    期待される出力サイズとスコア範囲を返すかを確認。
    """
    # テスト用パラメータ
    batch_size = 5          # バッチサイズ
    in_channels = 3         # RGBチャンネル
    img_height = 256        # 画像の高さ
    img_width = 256         # 画像の幅
    
    # ランダムな入力画像生成
    # 実際の使用時は[-1, 1]に正規化された画像を想定
    x = torch.randn((batch_size, in_channels, img_height, img_width))
    
    # Discriminatorインスタンス作成
    model = Discriminator(in_channels=in_channels)
    
    # 順伝播実行
    preds = model(x)
    
    # 結果の確認
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {preds.shape}")
    print(f"Output value range: [{preds.min():.3f}, {preds.max():.3f}]")
    
    # 期待される出力形状の確認
    # 256x256入力 -> 30x30出力 (70x70パッチに対応)
    expected_patch_size = 30
    expected_shape = (batch_size, 1, expected_patch_size, expected_patch_size)
    
    print(f"Expected output shape: {expected_shape}")
    print(f"Actual output shape: {preds.shape}")
    
    # スコアが[0, 1]範囲内かチェック
    assert 0 <= preds.min() and preds.max() <= 1, "Output should be in [0, 1] range"
    
    # 平均判別スコアの表示
    avg_score = preds.mean().item()
    print(f"Average discrimination score: {avg_score:.3f}")
    
    print("✓ Discriminator test passed!")


if __name__ == "__main__":
    test()
