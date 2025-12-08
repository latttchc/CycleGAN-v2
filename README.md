# CycleGAN Implementation

PyTorchを使用したCycleGAN（Cycle-Consistent Adversarial Networks）の実装です。
画像から画像への変換を、ペアになっていない訓練データで学習することができます。

本モデルは、Unpaired Image-to-Image Translation Problem via
Epoch-by-Epoch Comparative using CycleGAN at
Monet’s Cataract Stageでの実験を再現するためのプロジェクトです。
---
### 📚本研究の実績：
- IIAI AAI 2025-Winter 採択
- 情報処理学会 第88回全国大会 発表
---

## プロジェクト構成

```
.
├── config.py           # 設定ファイル（ハイパーパラメータ、パス等）
├── dataset.py          # カスタムデータセットクラス
├── discriminator.py    # Discriminatorモデルの定義
├── generator.py        # Generatorモデルの定義
├── train.py           # 訓練スクリプト
├── utils.py           # ユーティリティ関数（チェックポイント保存/読み込み等）
├── data/              # データセットディレクトリ
│   ├── train/
│   │   ├── trainA/    # ドメインAの訓練画像
│   │   └── trainB/    # ドメインBの訓練画像
│   └── val/
│       ├── testA/     # ドメインAのテスト画像
│       └── testB/     # ドメインBのテスト画像
└── saved_images/      # 生成された画像の保存先
```

## 必要な依存関係

```bash
pip install torch torchvision
pip install albumentations
pip install pillow
pip install numpy
pip install tqdm
```

## 使用方法

### 1. データセットの準備

データを以下のディレクトリ構造で配置してください：

```
data/
├── train/
│   ├── trainA/  # ドメインAの訓練画像
│   └── trainB/  # ドメインBの訓練画像
└── val/
    ├── testA/   # ドメインAのテスト画像
    └── testB/   # ドメインBのテスト画像
```

### 2. 設定の調整

[`config.py`](config.py)でハイパーパラメータを調整できます：

- `BATCH_SIZE`: バッチサイズ（デフォルト: 1）
- `LEARNING_RATE`: 学習率（デフォルト: 1e-5）
- `NUM_EPOCHS`: エポック数（デフォルト: 10）
- `LAMBDA_CYCLE`: サイクル一貫性損失の重み（デフォルト: 10）
- `LAMBDA_IDENTITY`: アイデンティティ損失の重み（デフォルト: 0.0）

### 3. 訓練の実行

```bash
python train.py
```

## モデル構成

### Generator
- [`Generator`](generator.py)クラス: U-Netベースのアーキテクチャ
- エンコーダ-デコーダ構造にResidual Blocksを組み込み
- Instance Normalizationを使用

### Discriminator
- [`Discriminator`](discriminator.py)クラス: PatchGANアーキテクチャ
- 70×70パッチレベルでの判別
- LeakyReLU活性化関数を使用

## 主要な機能

### 損失関数
- **Adversarial Loss**: GANの基本的な敵対的損失
- **Cycle Consistency Loss**: `G(F(x)) ≈ x` および `F(G(y)) ≈ y`
- **Identity Loss**: `G(y) ≈ y` および `F(x) ≈ x`（オプション）

### データ拡張
[`config.py`](config.py)でAlbumentationsを使用した拡張を定義：
- リサイズ（256×256）
- 水平フリップ
- 正規化

### チェックポイント機能
- [`utils.py`](utils.py)でモデルの保存/読み込み機能
- 訓練の中断/再開が可能

## 生成画像の保存

訓練中、200イテレーションごとに生成画像が`saved_images/`ディレクトリに保存されます：
- `fake_trainA_{idx}.png`: A→B変換結果
- `fake_trainB_{idx}.png`: B→A変換結果

## カスタマイズ

### データセットの変更
[`dataset.py`](dataset.py)の[`CustomDataset`](dataset.py)クラスを修正することで、独自のデータセット形式に対応できます。

### モデルアーキテクチャの変更
- [`generator.py`](generator.py): Generatorの構造変更
- [`discriminator.py`](discriminator.py): Discriminatorの構造変更

## 注意事項

- GPU使用時は十分なVRAMが必要です
- データセットのドメインAとBは同じサイズである必要はありません
- 訓練時間は使用するデータセットサイズとエポック数に依存します

## ライセンス
```
@inproceedings{CycleGAN2017,
  title={Unpaired Image-to-Image Translation using Cycle-Consistent Adversarial Networks},
  author={Zhu, Jun-Yan and Park, Taesung and Isola, Phillip and Efros, Alexei A},
  booktitle={Computer Vision (ICCV), 2017 IEEE International Conference on},
  year={2017}
}


@inproceedings{isola2017image,
  title={Image-to-Image Translation with Conditional Adversarial Networks},
  author={Isola, Phillip and Zhu, Jun-Yan and Zhou, Tinghui and Efros, Alexei A},
  booktitle={Computer Vision and Pattern Recognition (CVPR), 2017 IEEE Conference on},
  year={2017}
}
```
