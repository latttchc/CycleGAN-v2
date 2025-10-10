# train.py
import os
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from torch.utils.data import DataLoader
from torchvision.utils import save_image

import config
from discriminator import Discriminator
from generator import Generator
from dataset import CustomDataset
from utils import save_checkpoint, load_checkpoint
from losses import ColorConstancyLoss, PerceptualLoss, ContrastRegularization

# config設定によって条件付きでimport（学習履歴/可視化）
if getattr(config, "SAVE_HISTORY", False) or getattr(config, "SAVE_PROGRESS", False):
    from options.train_options import TrainOptions


# ============================ 学習1エポック ============================

def train_fn(
    disc_A, disc_B, gen_B, gen_A, loader,
    opt_disc, opt_gen, l1, mse,
    color_loss, perceptual_loss, contrast_loss,
    d_scaler, g_scaler,
    grad_clip=None
):
    """
    拡張CycleGANの1エポック分の訓練を実行
    """
    disc_A.train(); disc_B.train(); gen_A.train(); gen_B.train()

    # 指標集計
    A_reals = A_fakes = B_reals = B_fakes = 0.0
    total_D_loss = total_G_loss = 0.0
    total_cycle_loss = total_identity_loss = 0.0
    total_adversarial_loss = 0.0
    total_color_loss = total_perc_loss = total_contr_loss = 0.0

    loop = tqdm(loader, leave=True)
    num_batches = len(loader)

    for idx, (trainB, trainA) in enumerate(loop):
        trainB = trainB.to(config.DEVICE, non_blocking=True)  # domain B (before)
        trainA = trainA.to(config.DEVICE, non_blocking=True)  # domain A (after)

        # -------------------- Discriminator 更新 --------------------
        with torch.cuda.amp.autocast():
            # B->A の偽物
            fake_trainA = gen_A(trainB)
            D_A_real = disc_A(trainA)
            D_A_fake = disc_A(fake_trainA.detach())

            D_A_real_loss = mse(D_A_real, torch.ones_like(D_A_real))
            D_A_fake_loss = mse(D_A_fake, torch.zeros_like(D_A_fake))
            D_A_loss = D_A_real_loss + D_A_fake_loss

            A_reals += D_A_real.mean().item()
            A_fakes += D_A_fake.mean().item()

            # A->B の偽物
            fake_trainB = gen_B(trainA)
            D_B_real = disc_B(trainB)
            D_B_fake = disc_B(fake_trainB.detach())

            D_B_real_loss = mse(D_B_real, torch.ones_like(D_B_real))
            D_B_fake_loss = mse(D_B_fake, torch.zeros_like(D_B_fake))
            D_B_loss = D_B_real_loss + D_B_fake_loss

            B_reals += D_B_real.mean().item()
            B_fakes += D_B_fake.mean().item()

            D_loss = 0.5 * (D_A_loss + D_B_loss)

        opt_disc.zero_grad(set_to_none=True)
        d_scaler.scale(D_loss).backward()

        # 勾配クリップ（任意）
        if grad_clip is not None:
            d_scaler.unscale_(opt_disc)
            nn.utils.clip_grad_norm_(list(disc_A.parameters()) + list(disc_B.parameters()), max_norm=grad_clip)

        d_scaler.step(opt_disc)
        d_scaler.update()

        # -------------------- Generator 更新 --------------------
        with torch.cuda.amp.autocast():
            # 敵対的損失（Generator側）
            D_A_fake = disc_A(fake_trainA)
            D_B_fake = disc_B(fake_trainB)
            loss_G_A = mse(D_A_fake, torch.ones_like(D_A_fake))  # B->A を本物判定させる
            loss_G_B = mse(D_B_fake, torch.ones_like(D_B_fake))  # A->B を本物判定させる
            adversarial_loss = loss_G_A + loss_G_B

            # サイクル一貫性
            cycle_trainB = gen_B(fake_trainA)  # B->A->B
            cycle_trainA = gen_A(fake_trainB)  # A->B->A
            cycle_loss = l1(trainB, cycle_trainB) + l1(trainA, cycle_trainA)

            # アイデンティティ
            identity_trainB = gen_B(trainB)
            identity_trainA = gen_A(trainA)
            identity_loss = l1(trainB, identity_trainB) + l1(trainA, identity_trainA)

            # 追加損失：色恒常性（A<->B 両向き）
            cc_loss_A = color_loss(fake_trainA, trainA)  # B->A vs A参照
            cc_loss_B = color_loss(fake_trainB, trainB)  # A->B vs B参照
            total_cc_loss = cc_loss_A + cc_loss_B

            # 追加損失：知覚（VGG）
            perc_loss_A = perceptual_loss(fake_trainA, trainA)
            perc_loss_B = perceptual_loss(fake_trainB, trainB)
            total_perc_loss = perc_loss_A + perc_loss_B

            # 追加損失：コントラスト正則化（生成側のみ）
            contr_loss_A = contrast_loss(fake_trainA)
            contr_loss_B = contrast_loss(fake_trainB)
            total_contr_loss = contr_loss_A + contr_loss_B

            # 合成（λは一部 config 側／一部は内部で乗算）
            G_loss = (
                adversarial_loss +
                cycle_loss * config.LAMBDA_CYCLE +
                identity_loss * config.LAMBDA_IDENTITY +
                total_cc_loss +        # 内部で lambda_cc
                total_perc_loss +      # 内部で lambda_perceptual
                total_contr_loss       # 内部で lambda_contrast
            )

        opt_gen.zero_grad(set_to_none=True)
        g_scaler.scale(G_loss).backward()

        if grad_clip is not None:
            g_scaler.unscale_(opt_gen)
            nn.utils.clip_grad_norm_(list(gen_A.parameters()) + list(gen_B.parameters()), max_norm=grad_clip)

        g_scaler.step(opt_gen)
        g_scaler.update()

        # -------------------- ログ更新・画像保存 --------------------
        total_D_loss += D_loss.item()
        total_G_loss += G_loss.item()
        total_cycle_loss += cycle_loss.item()
        total_identity_loss += identity_loss.item()
        total_adversarial_loss += adversarial_loss.item()
        total_color_loss += total_cc_loss.item()
        total_perc_loss += total_perc_loss.item()
        total_contr_loss += total_contr_loss.item()

        if idx % 200 == 0:
            os.makedirs("saved_images_enhanced", exist_ok=True)
            # [-1,1] -> [0,1]
            save_image(fake_trainA * 0.5 + 0.5, f"saved_images_enhanced/fake_trainA_{idx}.png")
            save_image(fake_trainB * 0.5 + 0.5, f"saved_images_enhanced/fake_trainB_{idx}.png")

        loop.set_postfix(
            A_real=A_reals / (idx + 1),
            A_fake=A_fakes / (idx + 1),
            B_real=B_reals / (idx + 1),
            B_fake=B_fakes / (idx + 1),
            Color=total_color_loss / (idx + 1),
            Perc=total_perc_loss / (idx + 1),
            Contr=total_contr_loss / (idx + 1),
        )

    # 1エポックの指標
    return {
        "D_loss": total_D_loss / num_batches,
        "G_loss": total_G_loss / num_batches,
        "cycle_loss": total_cycle_loss / num_batches,
        "identity_loss": total_identity_loss / num_batches,
        "adversarial_loss": total_adversarial_loss / num_batches,
        "color_loss": total_color_loss / num_batches,
        "perceptual_loss": total_perc_loss / num_batches,
        "contrast_loss": total_contr_loss / num_batches,
        "A_real_score": A_reals / num_batches,
        "A_fake_score": A_fakes / num_batches,
        "B_real_score": B_reals / num_batches,
        "B_fake_score": B_fakes / num_batches,
    }


# ============================ メインループ ============================

def main():
    print("=" * 60)
    print("Enhanced CycleGAN Training")
    print(" + Color Constancy + Perceptual + Contrast Regularization")
    print("=" * 60)
    print(f"Device           : {config.DEVICE}")
    print(f"Epochs           : {config.NUM_EPOCHS}")
    print(f"Batch Size       : {config.BATCH_SIZE}")
    print(f"Learning Rate    : {config.LEARNING_RATE}")
    print(f"Lambda Cycle     : {config.LAMBDA_CYCLE}")
    print(f"Lambda Identity  : {config.LAMBDA_IDENTITY}")
    print(f"Lambda Color     : {config.LAMBDA_COLOR}")
    print(f"Lambda Perceptual: {config.LAMBDA_PERCEPTUAL}")
    print(f"Lambda Contrast  : {config.LAMBDA_CONTRAST}")
    print(f"TTUR             : {getattr(config, 'TTUR', False)}")
    print(f"Grad Clip        : {getattr(config, 'GRAD_CLIP', None)}")
    print(f"Save History     : {getattr(config, 'SAVE_HISTORY', False)}")
    print(f"Save Progress    : {getattr(config, 'SAVE_PROGRESS', False)}")
    print("=" * 60)

    # -------------------- モデル初期化 --------------------
    disc_A = Discriminator(in_channels=3).to(config.DEVICE)
    disc_B = Discriminator(in_channels=3).to(config.DEVICE)
    gen_B = Generator(img_channels=3, num_residuals=9).to(config.DEVICE)  # A->B
    gen_A = Generator(img_channels=3, num_residuals=9).to(config.DEVICE)  # B->A

    # -------------------- 拡張損失 --------------------
    color_loss = ColorConstancyLoss(lambda_cc=config.LAMBDA_COLOR).to(config.DEVICE)
    perceptual_loss = PerceptualLoss(lambda_perceptual=config.LAMBDA_PERCEPTUAL).to(config.DEVICE)
    contrast_loss = ContrastRegularization(lambda_contrast=config.LAMBDA_CONTRAST).to(config.DEVICE)

    print("✓ Enhanced losses initialized:")
    print(f"  - ColorConstancyLoss (λ={config.LAMBDA_COLOR})")
    print(f"  - PerceptualLoss    (λ={config.LAMBDA_PERCEPTUAL})")
    print(f"  - ContrastRegular.  (λ={config.LAMBDA_CONTRAST})")

    # -------------------- Optimizer（TTUR対応） --------------------
    ttur = getattr(config, "TTUR", False)
    lr_G = config.LEARNING_RATE
    lr_D = config.LEARNING_RATE * (2.0 if ttur else 1.0)

    opt_disc = optim.Adam(
        list(disc_A.parameters()) + list(disc_B.parameters()),
        lr=lr_D, betas=(0.5, 0.999)
    )
    opt_gen = optim.Adam(
        list(gen_B.parameters()) + list(gen_A.parameters()),
        lr=lr_G, betas=(0.5, 0.999)
    )

    # -------------------- 損失関数（基本） --------------------
    l1 = nn.L1Loss()
    mse = nn.MSELoss()

    # -------------------- チェックポイント読み込み --------------------
    if getattr(config, "LOAD_MODEL", False):
        load_checkpoint(config.CHECKPOINT_CRITIC_A, disc_A, opt_disc, lr_D)
        load_checkpoint(config.CHECKPOINT_CRITIC_B, disc_B, opt_disc, lr_D)
        load_checkpoint(config.CHECKPOINT_GEN_A, gen_A, opt_gen, lr_G)
        load_checkpoint(config.CHECKPOINT_GEN_B, gen_B, opt_gen, lr_G)

    # -------------------- データセット --------------------
    dataset = CustomDataset(
        root_a=os.path.join(config.TRAIN_DIR, "trainA"),
        root_b=os.path.join(config.TRAIN_DIR, "trainB"),
        transform=config.transforms,
    )
    loader = DataLoader(
        dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=True,
        num_workers=config.NUM_WORKERS,
        pin_memory=True,
        drop_last=True,
    )

    # -------------------- AMP用スケーラ --------------------
    d_scaler = torch.cuda.amp.GradScaler()
    g_scaler = torch.cuda.amp.GradScaler()

    # -------------------- 履歴/可視化オプション --------------------
    train_options = None
    if getattr(config, "SAVE_HISTORY", False) or getattr(config, "SAVE_PROGRESS", False):
        train_options = TrainOptions("plot_data_enhanced")
        print("✓ TrainOptions initialized for history/progress tracking")

    # -------------------- 学習ループ --------------------
    for epoch in range(config.NUM_EPOCHS):
        # 途中から Identity を弱めるスケジューリング（任意）
        if epoch == (config.NUM_EPOCHS // 2) and getattr(config, "SCHEDULE_IDENTITY", True):
            config.LAMBDA_IDENTITY = max(0.1, config.LAMBDA_IDENTITY * 0.5)
            print(f"⇒ Identity weight scheduled to {config.LAMBDA_IDENTITY}")

        print(f"\nEpoch [{epoch + 1}/{config.NUM_EPOCHS}]")
        metrics = train_fn(
            disc_A, disc_B, gen_B, gen_A, loader,
            opt_disc, opt_gen, l1, mse,
            color_loss, perceptual_loss, contrast_loss,
            d_scaler, g_scaler,
            grad_clip=getattr(config, "GRAD_CLIP", None),
        )

        # 可視化/履歴
        if train_options is not None:
            train_options.update_history(metrics)
            train_options.print_summary(metrics, epoch)
        else:
            print(f"  D_Loss: {metrics['D_loss']:.4f} | G_Loss: {metrics['G_loss']:.4f} | "
                  f"Color: {metrics['color_loss']:.4f} | Perc: {metrics['perceptual_loss']:.4f} | "
                  f"Contr: {metrics['contrast_loss']:.4f}")

        # モデル保存
        if getattr(config, "SAVE_MODEL", False):
            save_checkpoint(gen_A, opt_gen, filename="enhanced_" + config.CHECKPOINT_GEN_A)
            save_checkpoint(gen_B, opt_gen, filename="enhanced_" + config.CHECKPOINT_GEN_B)
            save_checkpoint(disc_A, opt_disc, filename="enhanced_" + config.CHECKPOINT_CRITIC_A)
            save_checkpoint(disc_B, opt_disc, filename="enhanced_" + config.CHECKPOINT_CRITIC_B)

        # 進捗図の保存（5エポックごと）
        if getattr(config, "SAVE_PROGRESS", False) and train_options is not None:
            if (epoch + 1) % 5 == 0:
                train_options.save_progress(epoch + 1)
                print(f"✓ Progress saved at epoch {epoch + 1}")

    # 終了処理
    if getattr(config, "SAVE_HISTORY", False) and train_options is not None:
        train_options.data_saver.save_training_history(train_options.history)
        train_options.data_saver.save_csv_format(train_options.history)
        print("✓ Training history saved")

    if getattr(config, "SAVE_PROGRESS", False) and train_options is not None:
        train_options.save_progress(config.NUM_EPOCHS, final=True)
        print("✓ Final progress plots saved")

    print("\n" + "=" * 60)
    print("Enhanced Training completed successfully!")
    print("Generated files:")
    print("  - saved_images_enhanced/: Generated images during training")
    if getattr(config, "SAVE_MODEL", False):
        print("  - enhanced_*.pth.tar: Model checkpoints")
    if getattr(config, "SAVE_HISTORY", False):
        print("  - plot_data_enhanced/: Training history (txt, csv)")
    if getattr(config, "SAVE_PROGRESS", False):
        print("  - plot_data_enhanced/: Training progress plots")
    print("=" * 60)


if __name__ == "__main__":
    main()
