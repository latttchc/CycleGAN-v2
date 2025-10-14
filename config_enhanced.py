import torch
import albumentations as A
from albumentations.pytorch import ToTensorV2

# デバイス設定
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# データセットのパス（モネの白内障前後作品）
TRAIN_DIR = "data/train"  # trainA: 白内障後, trainB: 白内障前
VAL_DIR = "data/val"      # valA: 白内障後, valB: 白内障前

# 学習パラメータ（5.2の表に従う）
BATCH_SIZE = 1            # 最小限のメモリ使用
LEARNING_RATE = 2e-4      # 論文のAdamデフォルト値
LAMBDA_IDENTITY = 5.0     # アイデンティティ損失の重み（高め）
LAMBDA_CYCLE = 20         # サイクル一貫性損失の重み（高め）
NUM_WORKERS = 0           # Windows環境では0推奨
NUM_EPOCHS = 200          # 長期学習（論文通り）

# 拡張損失の重み
LAMBDA_COLOR = 10.0       # 色恒常性損失
LAMBDA_PERCEPTUAL = 0.1   # 知覚損失
LAMBDA_CONTRAST = 0.05    # コントラスト正則化

# モデルの保存・読み込み設定
LOAD_MODEL = False        # 事前訓練済みモデルの読み込み
SAVE_MODEL = True         # モデルの保存
CHECKPOINT_INTERVAL = 10  # N epoch毎に保存

# チェックポイントファイル名
CHECKPOINT_GEN_A = "enhanced_genh.pth.tar"       # Generator A (B→A変換)
CHECKPOINT_GEN_B = "enhanced_genz.pth.tar"       # Generator B (A→B変換)
CHECKPOINT_CRITIC_A = "enhanced_critich.pth.tar" # Discriminator A
CHECKPOINT_CRITIC_B = "enhanced_criticz.pth.tar" # Discriminator B

# 結果保存設定
SAVE_HISTORY = True       # 学習履歴の保存
SAVE_PROGRESS = True      # 学習進捗の可視化
SAVE_IMAGES_INTERVAL = 200  # N イテレーション毎に生成画像保存

# Two Timescale Update Rule (TTUR)
USE_TTUR = False          # 異なる学習率を使用するか
GEN_LR = 1e-4             # Generator学習率（TTURを使用する場合）
DISC_LR = 4e-4            # Discriminator学習率（TTURを使用する場合）

# 学習安定化オプション
GRADIENT_CLIP = None      # 勾配クリッピング（Noneで無効）
USE_SPECTRAL_NORM = False # SpectralNormの使用有無

# データ前処理・拡張（5.1に従う）
transforms = A.Compose(
    [
        A.Resize(286, 286),                # 大きめにリサイズ
        A.RandomCrop(256, 256),            # ランダムクロップ
        A.HorizontalFlip(p=0.5),           # 水平反転（確率0.5）
        A.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], max_pixel_value=255),
        ToTensorV2(),
    ],
    additional_targets={"image0": "image"}, # 2つの画像に同じ変換を適用
)

# 512x512高解像度用の設定（オプション）
transforms_hr = A.Compose(
    [
        A.Resize(542, 542),                # 大きめにリサイズ
        A.RandomCrop(512, 512),            # ランダムクロップ
        A.HorizontalFlip(p=0.5),           # 水平反転（確率0.5）
        A.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], max_pixel_value=255),
        ToTensorV2(),
    ],
    additional_targets={"image0": "image"}, # 2つの画像に同じ変換を適用
)