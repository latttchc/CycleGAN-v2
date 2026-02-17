import torch
import torch.optim as optim
import config
from torch.utils.data import DataLoader
from torchvision.utils import save_image
from generator import Generator
from dataset import CustomDataset
from utils import load_checkpoint
from pathlib import Path
import matplotlib.pyplot as plt


# 設定
OUTPUT_DIR = 'saved_images/test'
MAX_IMAGES = 20

def test():
    gen_A = Generator(img_channels=3, num_residuals=9).to(config.DEVICE)
    gen_B = Generator(img_channels=3, num_residuals=9).to(config.DEVICE)

    opt_gen = optim.Adam(
        list(gen_A.parameters()) + list(gen_B.parameters()),
        lr=config.LEARNING_RATE,
    )

    load_checkpoint(config.CHECKPOINT_CRITIC_A, gen_A, opt_gen, config.LEARNING_RATE)
    load_checkpoint(config.CHECKPOINT_CRITIC_B, gen_B, opt_gen, config.LEARNING_RATE)

    gen_A.eval()
    gen_B.eval()

    dataset = CustomDataset(root_a=config.TRAIN_DIR + "/trainA", root_b=config.TRAIN_DIR + "/trainB", transform=config.transforms)

    loader = DataLoader(dataset, batch_size=config.BATCH_SIZE, shuffle=True, num_workers=config.NUM_WORKERS)

    output_dir = Path(OUTPUT_DIR)
    (output_dir / "A_to_B").mkdir(parents=True, exist_ok=True)
    (output_dir / "B_to_A").mkdir(parents=True, exist_ok=True)
    print(f"出力先: {output_dir.resolve()}")
    print("-" * 60)

    with torch.no_grad():
        for idx, (real_A, real_B) in enumerate(loader):
            if idx >= MAX_IMAGES:
                break
            real_A = real_A.to(config.DEVICE)
            real_B = real_B.to(config.DEVICE)
            save_image(real_A * 0.5 + 0.5, f"{output_dir}/real_A_{idx}.png")
            save_image(real_B * 0.5 + 0.5, f"{output_dir}/real_B_{idx}.png")

            fake_B = gen_B(real_A)
            fake_A = gen_A(real_B)

            save_image(fake_A * 0.5 + 0.5, f"{output_dir}/fake_A_{idx}.png")
            save_image(fake_B * 0.5 + 0.5, f"{output_dir}/fake_B_{idx}.png")

            print(f" [{idx + 1}/{min(len(dataset), MAX_IMAGES)}] 保存完了")