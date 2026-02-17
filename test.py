import torch
import torch.optim as optim
import config
from torch.utils.data import DataLoader
from torchvision.utils import save_image
from generator import Generator
from dataset import CustomDataset
from utils import load_checkpoint
from pathlib import Path
import os


# 設定
OUTPUT_DIR = 'saved_images/test'
MAX_IMAGES = 20

def test():
    """
    学習済みCycleGANでテスト画像を生成するスクリプト
    
    Generator A (B→A変換): CHECKPOINT_GEN_A から読み込み
    Generator B (A→B変換): CHECKPOINT_GEN_B から読み込み
    """
    print("=" * 60)
    print("CycleGAN Test - Image Generation")
    print("=" * 60)
    print(f"Device: {config.DEVICE}")
    print(f"Max Images: {MAX_IMAGES}")
    print("=" * 60)

    # Generator の初期化
    gen_A = Generator(img_channels=3, num_residuals=9).to(config.DEVICE)
    gen_B = Generator(img_channels=3, num_residuals=9).to(config.DEVICE)

    # Generator 用のオプティマイザー（チェックポイント読み込みに必要）
    opt_gen = optim.Adam(
        list(gen_A.parameters()) + list(gen_B.parameters()),
        lr=config.LEARNING_RATE,
        betas=(0.5, 0.999)
    )

    # チェックポイントの読み込み（Generator用のチェックポイントを使用）
    if os.path.exists(config.CHECKPOINT_GEN_A):
        load_checkpoint(config.CHECKPOINT_GEN_A, gen_A, opt_gen, config.LEARNING_RATE)
        print(f"✓ Generator A loaded from {config.CHECKPOINT_GEN_A}")
    else:
        print(f"✗ {config.CHECKPOINT_GEN_A} が見つかりません")
        return

    if os.path.exists(config.CHECKPOINT_GEN_B):
        load_checkpoint(config.CHECKPOINT_GEN_B, gen_B, opt_gen, config.LEARNING_RATE)
        print(f"✓ Generator B loaded from {config.CHECKPOINT_GEN_B}")
    else:
        print(f"✗ {config.CHECKPOINT_GEN_B} が見つかりません")
        return

    gen_A.eval()
    gen_B.eval()

    # テスト用データセットの準備（valディレクトリを使用）
    dataset = CustomDataset(
        root_a=config.VAL_DIR + "/testA",
        root_b=config.VAL_DIR + "/testB",
        transform=config.transforms
    )
    loader = DataLoader(
        dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=False,
        num_workers=config.NUM_WORKERS
    )

    # 出力ディレクトリの作成
    output_dir = Path(OUTPUT_DIR)
    (output_dir / "A_to_B").mkdir(parents=True, exist_ok=True)
    (output_dir / "B_to_A").mkdir(parents=True, exist_ok=True)
    print(f"\n出力先: {output_dir.resolve()}")
    print("-" * 60)

    num_images = min(len(dataset), MAX_IMAGES)

    with torch.no_grad():
        for idx, (real_A, real_B) in enumerate(loader):
            if idx >= MAX_IMAGES:
                break

            real_A = real_A.to(config.DEVICE)
            real_B = real_B.to(config.DEVICE)

            # A→B 変換
            fake_B = gen_B(real_A)
            # B→A 変換
            fake_A = gen_A(real_B)
            # サイクル復元
            cycle_A = gen_A(fake_B)
            cycle_B = gen_B(fake_A)

            # [-1, 1] → [0, 1] に変換して保存
            save_image(real_A * 0.5 + 0.5, f"{output_dir}/A_to_B/real_A_{idx}.png")
            save_image(fake_B * 0.5 + 0.5, f"{output_dir}/A_to_B/fake_B_{idx}.png")
            save_image(cycle_A * 0.5 + 0.5, f"{output_dir}/A_to_B/cycle_A_{idx}.png")

            save_image(real_B * 0.5 + 0.5, f"{output_dir}/B_to_A/real_B_{idx}.png")
            save_image(fake_A * 0.5 + 0.5, f"{output_dir}/B_to_A/fake_A_{idx}.png")
            save_image(cycle_B * 0.5 + 0.5, f"{output_dir}/B_to_A/cycle_B_{idx}.png")

            print(f"  [{idx + 1}/{num_images}] 保存完了")

    print("-" * 60)
    print(f"テスト完了！ {output_dir.resolve()} に画像を保存しました")
    print("  - A_to_B/: ドメインA→B変換結果 (real_A, fake_B, cycle_A)")
    print("  - B_to_A/: ドメインB→A変換結果 (real_B, fake_A, cycle_B)")
    print("=" * 60)


if __name__ == "__main__":
    test()