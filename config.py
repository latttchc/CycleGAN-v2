import torch
import albumentations as A
from albumentations.pytorch import ToTensorV2

# デバイス設定
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# データセットのパス
TRAIN_DIR = "data/train"
VAL_DIR = "data/val"

# 学習パラメータ
BATCH_SIZE = 1          # バッチサイズ
LEARNING_RATE = 1e-5    # 学習率
LAMBDA_IDENTITY = 0.0   # アイデンティティ損失の重み
LAMBDA_CYCLE = 10       # サイクル一貫性損失の重み
NUM_WORKERS = 4         # データローダーのワーカー数
NUM_EPOCHS = 10         # 学習エポック数

# 提案: 拡張損失の重み
LAMBDA_COLOR = 10.0           # 色恒常性損失（提案値）
LAMBDA_PERCEPTUAL = 0.1       # 知覚損失
LAMBDA_CONTRAST = 0.05        # コントラスト正則化

# モデルの保存・読み込み設定
LOAD_MODEL = False      # 事前訓練済みモデルの読み込み
SAVE_MODEL = True       # モデルの保存

# チェックポイントファイル名
CHECKPOINT_GEN_A = "genh.pth.tar"       # Generator A (B→A変換)
CHECKPOINT_GEN_B = "genz.pth.tar"       # Generator B (A→B変換)
CHECKPOINT_CRITIC_A = "critich.pth.tar" # Discriminator A
CHECKPOINT_CRITIC_B = "criticz.pth.tar" # Discriminator B

# 結果保存設定
SAVE_HISTORY = True     # 学習履歴の保存
SAVE_PROGRESS = True    # 学習進捗の可視化

# データ前処理・拡張
transforms = A.Compose(
    [
        A.Resize(width=256, height=256),    # 画像サイズを256x256に統一
        A.HorizontalFlip(p=0.5),           # 50%の確率で水平反転
        A.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], max_pixel_value=255),  # [-1, 1]に正規化
        ToTensorV2(),                      # PyTorchテンソルに変換
    ],
    additional_targets={"image0": "image"}, # 2つの画像に同じ変換を適用
)