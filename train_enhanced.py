import torch 
import torch.nn as nn
import torch.optim as optim
import config_enhanced as config
import os
from tqdm import tqdm
from torch.utils.data import DataLoader
from torchvision.utils import save_image
from discriminator import Discriminator
from generator import Generator
from dataset import CustomDataset
from utils import save_checkpoint, load_checkpoint

# 拡張損失関数のインポート
from losses.color_constancy_loss import ColorConstancyLoss
from losses.perceptual_loss import PerceptualLoss
from losses.contrast_regularization import ContrastRegularization

# TrainOptionsのインポート（必要に応じて）
if config.SAVE_HISTORY or config.SAVE_PROGRESS:
    from options.train_options import TrainOptions


def train_fn(disc_A, disc_B, gen_B, gen_A, loader, opt_disc, opt_gen, 
             l1, mse, color_loss, perceptual_loss, contrast_loss,
             d_scaler, g_scaler):
    """
    拡張CycleGANの1エポック分の訓練を実行する関数
    
    Args:
        color_loss: 色恒常性損失
        perceptual_loss: 知覚損失
        contrast_loss: コントラスト正則化
        
    Returns:
        dict: エポックの評価指標
    """
    # Discriminatorの性能指標を追跡
    A_reals = 0
    A_fakes = 0
    B_reals = 0
    B_fakes = 0
    
    # 損失値の累計
    total_D_loss = 0
    total_G_loss = 0
    total_cycle_loss = 0
    total_identity_loss = 0
    total_adversarial_loss = 0
    total_color_loss = 0
    total_perceptual_loss = 0
    total_contrast_loss = 0
    
    # プログレスバー
    loop = tqdm(loader, leave=True)
    num_batches = len(loader)

    # 各バッチに対して訓練を実行
    for idx, (trainB, trainA) in enumerate(loop):
        # データをGPU/CPUに移動
        trainB = trainB.to(config.DEVICE)  # ドメインB（白内障前）
        trainA = trainA.to(config.DEVICE)  # ドメインA（白内障後）

        # ========== Discriminatorの訓練 ==========
        with torch.cuda.amp.autocast():
            # ドメインA用Discriminatorの訓練
            fake_trainA = gen_A(trainB)  # B→A変換（白内障前→後）
            D_A_real = disc_A(trainA)    # 本物のA画像に対する判別
            D_A_fake = disc_A(fake_trainA.detach())  # 偽のA画像に対する判別
            
            # 性能指標を記録
            A_reals += D_A_real.mean().item()
            A_fakes += D_A_fake.mean().item()

            # ドメインA用Discriminatorの損失計算
            D_A_real_loss = mse(D_A_real, torch.ones_like(D_A_real))
            D_A_fake_loss = mse(D_A_fake, torch.zeros_like(D_A_fake))
            D_A_loss = D_A_real_loss + D_A_fake_loss

            # ドメインB用Discriminatorの訓練
            fake_trainB = gen_B(trainA)  # A→B変換（白内障後→前）
            D_B_real = disc_B(trainB)    # 本物のB画像に対する判別
            D_B_fake = disc_B(fake_trainB.detach())  # 偽のB画像に対する判別
            
            # 性能指標を記録
            B_reals += D_B_real.mean().item()
            B_fakes += D_B_fake.mean().item()

            # ドメインB用Discriminatorの損失計算
            D_B_real_loss = mse(D_B_real, torch.ones_like(D_B_real))
            D_B_fake_loss = mse(D_B_fake, torch.zeros_like(D_B_fake))
            D_B_loss = D_B_real_loss + D_B_fake_loss

            # 全Discriminator損失
            D_loss = (D_A_loss + D_B_loss) / 2

        # Discriminatorの重み更新
        opt_disc.zero_grad()
        d_scaler.scale(D_loss).backward()
        
        # 勾配クリッピング（設定されている場合）
        if config.GRADIENT_CLIP is not None:
            d_scaler.unscale_(opt_disc)
            torch.nn.utils.clip_grad_norm_(
                list(disc_A.parameters()) + list(disc_B.parameters()), 
                config.GRADIENT_CLIP
            )
        
        d_scaler.step(opt_disc)
        d_scaler.update()

        # ========== Generatorの訓練 ==========
        with torch.cuda.amp.autocast():
            # 敵対的損失
            D_A_fake = disc_A(fake_trainA)
            D_B_fake = disc_B(fake_trainB)
            loss_G_A = mse(D_A_fake, torch.ones_like(D_A_fake))
            loss_G_B = mse(D_B_fake, torch.ones_like(D_B_fake))
            adversarial_loss = loss_G_A + loss_G_B

            # サイクル一貫性損失
            cycle_trainB = gen_B(fake_trainA)  # B→A→B
            cycle_trainA = gen_A(fake_trainB)  # A→B→A
            cycle_trainA_loss = l1(trainB, cycle_trainB)
            cycle_trainB_loss = l1(trainA, cycle_trainA)
            cycle_loss = cycle_trainA_loss + cycle_trainB_loss

            # アイデンティティ損失
            identity_trainB = gen_B(trainB)  # B→B (同一性)
            identity_trainA = gen_A(trainA)  # A→A (同一性)
            identity_trainB_loss = l1(trainB, identity_trainB)
            identity_trainA_loss = l1(trainA, identity_trainA)
            identity_loss = identity_trainA_loss + identity_trainB_loss

            # ========== 拡張損失 ==========
            # 1. 色恒常性損失
            cc_loss_A = color_loss(fake_trainA, trainA)  # B→A vs A参照
            cc_loss_B = color_loss(fake_trainB, trainB)  # A→B vs B参照
            cc_loss = cc_loss_A + cc_loss_B
            
            # 2. 知覚損失
            perc_loss_A = perceptual_loss(fake_trainA, trainA)  # B→A vs A参照
            perc_loss_B = perceptual_loss(fake_trainB, trainB)  # A→B vs B参照
            perc_loss = perc_loss_A + perc_loss_B
            
            # 3. コントラスト正則化
            contr_loss_A = contrast_loss(fake_trainA, trainA)  # B→A vs A参照
            contr_loss_B = contrast_loss(fake_trainB, trainB)  # A→B vs B参照
            contr_loss = contr_loss_A + contr_loss_B

            # 全Generator損失（拡張版）
            G_loss = (
                adversarial_loss +
                cycle_loss * config.LAMBDA_CYCLE +
                identity_loss * config.LAMBDA_IDENTITY +
                cc_loss +        # 色恒常性損失
                perc_loss +      # 知覚損失
                contr_loss       # コントラスト正則化
            )

        # Generatorの重み更新
        opt_gen.zero_grad()
        g_scaler.scale(G_loss).backward()
        
        # 勾配クリッピング（設定されている場合）
        if config.GRADIENT_CLIP is not None:
            g_scaler.unscale_(opt_gen)
            torch.nn.utils.clip_grad_norm_(
                list(gen_A.parameters()) + list(gen_B.parameters()), 
                config.GRADIENT_CLIP
            )
            
        g_scaler.step(opt_gen)
        g_scaler.update()

        # 損失値の累計
        total_D_loss += D_loss.item()
        total_G_loss += G_loss.item()
        total_cycle_loss += cycle_loss.item()
        total_identity_loss += identity_loss.item()
        total_adversarial_loss += adversarial_loss.item()
        total_color_loss += cc_loss.item()
        total_perceptual_loss += perc_loss.item()
        total_contrast_loss += contr_loss.item()

        # 定期的に生成画像を保存
        if idx % config.SAVE_IMAGES_INTERVAL == 0:
            os.makedirs("saved_images_enhanced", exist_ok=True)
            save_image(fake_trainA * 0.5 + 0.5, f"saved_images_enhanced/fake_trainA_{idx}.png")
            save_image(fake_trainB * 0.5 + 0.5, f"saved_images_enhanced/fake_trainB_{idx}.png")
            save_image(trainA * 0.5 + 0.5, f"saved_images_enhanced/real_trainA_{idx}.png")
            save_image(trainB * 0.5 + 0.5, f"saved_images_enhanced/real_trainB_{idx}.png")

        # プログレスバーに統計情報を表示
        loop.set_postfix(
            A_real=A_reals / (idx + 1),
            A_fake=A_fakes / (idx + 1),
            D_loss=total_D_loss / (idx + 1),
            G_loss=total_G_loss / (idx + 1)
        )

    # エポックの評価指標を返す（拡張版）
    return {
        'D_loss': total_D_loss / num_batches,
        'G_loss': total_G_loss / num_batches,
        'cycle_loss': total_cycle_loss / num_batches,
        'identity_loss': total_identity_loss / num_batches,
        'adversarial_loss': total_adversarial_loss / num_batches,
        'color_loss': total_color_loss / num_batches,
        'perceptual_loss': total_perceptual_loss / num_batches,
        'contrast_loss': total_contrast_loss / num_batches,
        'A_real_score': A_reals / num_batches,
        'A_fake_score': A_fakes / num_batches,
        'B_real_score': B_reals / num_batches,
        'B_fake_score': B_fakes / num_batches,
    }


def main():
    """
    拡張CycleGANのメイン訓練ループ
    """
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
    print(f"TTUR             : {config.USE_TTUR}")
    print(f"Grad Clip        : {config.GRADIENT_CLIP}")
    print(f"Save History     : {config.SAVE_HISTORY}")
    print(f"Save Progress    : {config.SAVE_PROGRESS}")
    print("=" * 60)

    # ========== モデルの初期化 ==========
    disc_A = Discriminator(in_channels=3).to(config.DEVICE)  # ドメインA（白内障後）判別器
    disc_B = Discriminator(in_channels=3).to(config.DEVICE)  # ドメインB（白内障前）判別器
    gen_B = Generator(img_channels=3, num_residuals=9).to(config.DEVICE)  # A→B変換（白内障後→前）
    gen_A = Generator(img_channels=3, num_residuals=9).to(config.DEVICE)  # B→A変換（白内障前→後）

    # ========== オプティマイザーの初期化 ==========
    # Two Timescale Update Ruleを使用するかどうかで学習率を変える
    if config.USE_TTUR:
        disc_lr = config.DISC_LR
        gen_lr = config.GEN_LR
    else:
        disc_lr = config.LEARNING_RATE
        gen_lr = config.LEARNING_RATE
    
    opt_disc = optim.Adam(
        list(disc_A.parameters()) + list(disc_B.parameters()),
        lr=disc_lr,
        betas=(0.5, 0.999)
    )
    
    opt_gen = optim.Adam(
        list(gen_B.parameters()) + list(gen_A.parameters()),
        lr=gen_lr,
        betas=(0.5, 0.999)
    )

    # ========== 損失関数の定義 ==========
    l1 = nn.L1Loss()
    mse = nn.MSELoss()
    
    # ========== 拡張損失関数の初期化 ==========
    color_loss = ColorConstancyLoss(lambda_cc=config.LAMBDA_COLOR).to(config.DEVICE)
    perceptual_loss = PerceptualLoss(lambda_perceptual=config.LAMBDA_PERCEPTUAL).to(config.DEVICE)
    contrast_loss = ContrastRegularization(lambda_contrast=config.LAMBDA_CONTRAST).to(config.DEVICE)
    
    print(f"✓ Enhanced losses initialized:")
    print(f"  - ColorConstancyLoss (λ={config.LAMBDA_COLOR})")
    print(f"  - PerceptualLoss    (λ={config.LAMBDA_PERCEPTUAL})")
    print(f"  - ContrastRegular.  (λ={config.LAMBDA_CONTRAST})")

    # ========== チェックポイントの読み込み ==========
    if config.LOAD_MODEL:
        load_checkpoint(config.CHECKPOINT_CRITIC_A, disc_A, opt_disc, disc_lr)
        load_checkpoint(config.CHECKPOINT_CRITIC_B, disc_B, opt_disc, disc_lr)
        load_checkpoint(config.CHECKPOINT_GEN_A, gen_A, opt_gen, gen_lr)
        load_checkpoint(config.CHECKPOINT_GEN_B, gen_B, opt_gen, gen_lr)
        print("✓ Checkpoints loaded")

    # ========== データセットとDataLoaderの準備 ==========
    dataset = CustomDataset(
        root_a=config.TRAIN_DIR + "/trainA",  # ドメインA（白内障後）
        root_b=config.TRAIN_DIR + "/trainB",  # ドメインB（白内障前）
        transform=config.transforms
    )
    loader = DataLoader(
        dataset, 
        batch_size=config.BATCH_SIZE, 
        shuffle=True, 
        num_workers=config.NUM_WORKERS
    )

    # ========== Mixed Precision Training用のScaler ==========
    d_scaler = torch.cuda.amp.GradScaler()
    g_scaler = torch.cuda.amp.GradScaler()

    # ========== TrainOptions初期化（設定に応じて） ==========
    train_options = None
    if config.SAVE_HISTORY or config.SAVE_PROGRESS:
        train_options = TrainOptions("plot_data_enhanced")
        print("✓ TrainOptions initialized for history/progress tracking")

    # ========== メイン訓練ループ ==========
    for epoch in range(config.NUM_EPOCHS):
        print(f"\nEpoch [{epoch+1}/{config.NUM_EPOCHS}]")
        
        # 1エポック分の訓練を実行
        metrics = train_fn(
            disc_A, disc_B, gen_B, gen_A, loader, 
            opt_disc, opt_gen, l1, mse, 
            color_loss, perceptual_loss, contrast_loss,
            d_scaler, g_scaler
        )
        
        # 学習履歴の更新（設定に応じて）
        if config.SAVE_HISTORY or config.SAVE_PROGRESS:
            train_options.update_history(metrics)
            train_options.print_summary(metrics, epoch)
        else:
            # 設定がOFFの場合は簡易表示
            print(f"  D_Loss: {metrics['D_loss']:.4f}")
            print(f"  G_Loss: {metrics['G_loss']:.4f}")
            print(f"  Color Loss: {metrics['color_loss']:.4f}")
            print(f"  Perceptual Loss: {metrics['perceptual_loss']:.4f}")
            print(f"  Contrast Loss: {metrics['contrast_loss']:.4f}")

        # ========== モデルの保存（エポック間隔で） ==========
        if config.SAVE_MODEL and (epoch + 1) % config.CHECKPOINT_INTERVAL == 0:
            save_checkpoint(gen_A, opt_gen, filename=f"epoch_{epoch+1}_" + config.CHECKPOINT_GEN_A)
            save_checkpoint(gen_B, opt_gen, filename=f"epoch_{epoch+1}_" + config.CHECKPOINT_GEN_B)
            save_checkpoint(disc_A, opt_disc, filename=f"epoch_{epoch+1}_" + config.CHECKPOINT_CRITIC_A)
            save_checkpoint(disc_B, opt_disc, filename=f"epoch_{epoch+1}_" + config.CHECKPOINT_CRITIC_B)
            print(f"✓ Checkpoints saved at epoch {epoch+1}")

        # ========== 学習進捗の可視化（設定に応じて） ==========
        if config.SAVE_PROGRESS and train_options:
            # 5エポックごとに中間結果を保存
            if (epoch + 1) % 5 == 0:
                train_options.save_progress(epoch + 1)
                print(f"✓ Progress plots saved at epoch {epoch+1}")

    # ========== 最終モデルの保存 ==========
    if config.SAVE_MODEL:
        save_checkpoint(gen_A, opt_gen, filename=config.CHECKPOINT_GEN_A)
        save_checkpoint(gen_B, opt_gen, filename=config.CHECKPOINT_GEN_B)
        save_checkpoint(disc_A, opt_disc, filename=config.CHECKPOINT_CRITIC_A)
        save_checkpoint(disc_B, opt_disc, filename=config.CHECKPOINT_CRITIC_B)
        print("✓ Final checkpoints saved")

    # ========== 最終結果の保存（設定に応じて） ==========
    if config.SAVE_HISTORY and train_options:
        train_options.data_saver.save_training_history(train_options.history)
        train_options.data_saver.save_csv_format(train_options.history)
        print("✓ Training history saved")

    if config.SAVE_PROGRESS and train_options:
        train_options.save_progress(config.NUM_EPOCHS, final=True)
        print("✓ Final progress plots saved")

    print("\n" + "=" * 60)
    print("Enhanced CycleGAN training completed!")
    print("Generated files:")
    print("  - saved_images_enhanced/: Generated images during training")
    if config.SAVE_MODEL:
        print("  - enhanced_*.pth.tar: Model checkpoints")
    if config.SAVE_HISTORY:
        print("  - plot_data_enhanced/: Training history (txt, csv)")
    if config.SAVE_PROGRESS:
        print("  - plot_data_enhanced/: Training progress plots")
    print("=" * 60)


# スクリプトが直接実行された場合のみmain関数を呼び出し
if __name__ == "__main__":
    main()