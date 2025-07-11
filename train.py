import torch 
import torch.nn as nn
import torch.optim as optim
import config
import os
from tqdm import tqdm
from torch.utils.data import DataLoader
from torchvision.utils import save_image
from discriminator import Discriminator
from generator import Generator
from dataset import CustomDataset
from utils import save_checkpoint, load_checkpoint

# config設定によって条件付きでimport
if config.SAVE_HISTORY or config.SAVE_PROGRESS:
    from options.train_options import TrainOptions


def train_fn(disc_A, disc_B, gen_B, gen_A, loader, opt_disc, opt_gen, l1, mse, d_scaler, g_scaler):
    """
    CycleGANの1エポック分の訓練を実行する関数
    
    Returns:
        dict: エポックの評価指標
    """
    # Discriminatorの性能指標を追跡するための変数
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
    
    # プログレスバーでバッチ処理の進行状況を表示
    loop = tqdm(loader, leave=True)
    num_batches = len(loader)

    # 各バッチに対して訓練を実行
    for idx, (trainB, trainA) in enumerate(loop):
        # データをGPU/CPUに移動
        trainB = trainB.to(config.DEVICE)
        trainA = trainA.to(config.DEVICE)

        # ========== Discriminatorの訓練 ==========
        with torch.cuda.amp.autocast():
            # ドメインA用Discriminatorの訓練
            fake_trainA = gen_A(trainB)
            D_A_real = disc_A(trainA)
            D_A_fake = disc_A(fake_trainA.detach())

            # 性能指標を記録
            A_reals += D_A_real.mean().item()
            A_fakes += D_A_fake.mean().item()

            # ドメインA用Discriminatorの損失計算
            D_A_real_loss = mse(D_A_real, torch.ones_like(D_A_real))
            D_A_fake_loss = mse(D_A_fake, torch.zeros_like(D_A_fake))
            D_A_loss = D_A_real_loss + D_A_fake_loss

            # ドメインB用Discriminatorの訓練
            fake_trainB = gen_B(trainA)
            D_B_real = disc_B(trainB)
            D_B_fake = disc_B(fake_trainB.detach())

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
            cycle_trainB = gen_B(fake_trainA)
            cycle_trainA = gen_A(fake_trainB)
            cycle_trainA_loss = l1(trainB, cycle_trainB)
            cycle_trainB_loss = l1(trainA, cycle_trainA)
            cycle_loss = cycle_trainA_loss + cycle_trainB_loss

            # アイデンティティ損失
            identity_trainB = gen_B(trainB)
            identity_trainA = gen_A(trainA)
            identity_trainB_loss = l1(trainB, identity_trainB)
            identity_trainA_loss = l1(trainA, identity_trainA)
            identity_loss = identity_trainA_loss + identity_trainB_loss

            # 全Generator損失
            G_loss = (
                adversarial_loss +
                cycle_loss * config.LAMBDA_CYCLE +
                identity_loss * config.LAMBDA_IDENTITY
            )

        # Generatorの重み更新
        opt_gen.zero_grad()
        g_scaler.scale(G_loss).backward()
        g_scaler.step(opt_gen)
        g_scaler.update()

        # 損失値の累計
        total_D_loss += D_loss.item()
        total_G_loss += G_loss.item()
        total_cycle_loss += cycle_loss.item()
        total_identity_loss += identity_loss.item()
        total_adversarial_loss += adversarial_loss.item()

        # 定期的に生成画像を保存
        if idx % 200 == 0:
            # saved_imagesディレクトリを作成
            os.makedirs("saved_images", exist_ok=True)
            save_image(fake_trainA * 0.5 + 0.5, f"saved_images/fake_trainA_{idx}.png")
            save_image(fake_trainB * 0.5 + 0.5, f"saved_images/fake_trainB_{idx}.png")

        # プログレスバーに統計情報を表示
        loop.set_postfix(
            A_real=A_reals / (idx + 1),
            A_fake=A_fakes / (idx + 1)
        )

    # エポックの評価指標を返す
    return {
        'D_loss': total_D_loss / num_batches,
        'G_loss': total_G_loss / num_batches,
        'cycle_loss': total_cycle_loss / num_batches,
        'identity_loss': total_identity_loss / num_batches,
        'adversarial_loss': total_adversarial_loss / num_batches,
        'A_real_score': A_reals / num_batches,
        'A_fake_score': A_fakes / num_batches,
        'B_real_score': B_reals / num_batches,
        'B_fake_score': B_fakes / num_batches,
    }


def main():
    """
    CycleGANのメイン訓練ループ
    """
    print("=" * 50)
    print("CycleGAN Training Started")
    print("=" * 50)
    print(f"Device: {config.DEVICE}")
    print(f"Epochs: {config.NUM_EPOCHS}")
    print(f"Batch Size: {config.BATCH_SIZE}")
    print(f"Learning Rate: {config.LEARNING_RATE}")
    print(f"Save History: {config.SAVE_HISTORY}")
    print(f"Save Progress: {config.SAVE_PROGRESS}")
    print("=" * 50)

    # ========== モデルの初期化 ==========
    disc_A = Discriminator(in_channels=3).to(config.DEVICE)
    disc_B = Discriminator(in_channels=3).to(config.DEVICE)
    gen_B = Generator(img_channels=3, num_residuals=9).to(config.DEVICE)
    gen_A = Generator(img_channels=3, num_residuals=9).to(config.DEVICE)

    # ========== オプティマイザーの初期化 ==========
    opt_disc = optim.Adam(
        list(disc_A.parameters()) + list(disc_B.parameters()),
        lr=config.LEARNING_RATE,
        betas=(0.5, 0.999)
    )
    
    opt_gen = optim.Adam(
        list(gen_B.parameters()) + list(gen_A.parameters()),
        lr=config.LEARNING_RATE,
        betas=(0.5, 0.999)
    )

    # ========== 損失関数の定義 ==========
    l1 = nn.L1Loss()
    mse = nn.MSELoss()

    # ========== チェックポイントの読み込み ==========
    if config.LOAD_MODEL:
        load_checkpoint(config.CHECKPOINT_CRITIC_A, disc_A, opt_disc, config.LEARNING_RATE)
        load_checkpoint(config.CHECKPOINT_CRITIC_B, disc_B, opt_disc, config.LEARNING_RATE)
        load_checkpoint(config.CHECKPOINT_GEN_A, gen_A, opt_gen, config.LEARNING_RATE)
        load_checkpoint(config.CHECKPOINT_GEN_B, gen_B, opt_gen, config.LEARNING_RATE)

    # ========== データセットとDataLoaderの準備 ==========
    dataset = CustomDataset(
        root_a=config.TRAIN_DIR + "/trainA", 
        root_b=config.TRAIN_DIR + "/trainB", 
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
        train_options = TrainOptions("plot_data")
        print("✓ TrainOptions initialized for history/progress tracking")

    # ========== メイン訓練ループ ==========
    for epoch in range(config.NUM_EPOCHS):
        print(f"\nEpoch [{epoch+1}/{config.NUM_EPOCHS}]")
        
        # 1エポック分の訓練を実行
        epoch_metrics = train_fn(
            disc_A, disc_B, gen_B, gen_A, loader, 
            opt_disc, opt_gen, l1, mse, d_scaler, g_scaler
        )
        
        # 学習履歴の更新（設定に応じて）
        if config.SAVE_HISTORY or config.SAVE_PROGRESS:
            train_options.update_history(epoch_metrics)
            train_options.print_summary(epoch_metrics, epoch)
        else:
            # 設定がOFFの場合は簡易表示
            print(f"  D_Loss: {epoch_metrics['D_loss']:.4f}")
            print(f"  G_Loss: {epoch_metrics['G_loss']:.4f}")
            print(f"  A_Real_Score: {epoch_metrics['A_real_score']:.3f}")
            print(f"  A_Fake_Score: {epoch_metrics['A_fake_score']:.3f}")

        # ========== モデルの保存 ==========
        if config.SAVE_MODEL:
            save_checkpoint(gen_A, opt_gen, filename=config.CHECKPOINT_GEN_A)
            save_checkpoint(gen_B, opt_gen, filename=config.CHECKPOINT_GEN_B)
            save_checkpoint(disc_A, opt_disc, filename=config.CHECKPOINT_CRITIC_A)
            save_checkpoint(disc_B, opt_disc, filename=config.CHECKPOINT_CRITIC_B)

        # ========== 学習進捗の可視化（設定に応じて） ==========
        if config.SAVE_PROGRESS and train_options:
            # 5エポックごとに中間結果を保存
            if (epoch + 1) % 5 == 0:
                train_options.save_progress(epoch + 1)
                print(f"✓ Progress saved at epoch {epoch + 1}")

    # ========== 最終結果の保存（設定に応じて） ==========
    if config.SAVE_HISTORY and train_options:
        train_options.data_saver.save_training_history(train_options.history)
        train_options.data_saver.save_csv_format(train_options.history)
        print("✓ Training history saved")

    if config.SAVE_PROGRESS and train_options:
        train_options.save_progress(config.NUM_EPOCHS, final=True)
        print("✓ Final progress plots saved")

    print("\n" + "=" * 50)
    print("Training completed successfully!")
    print("Generated files:")
    print("  - saved_images/: Generated images during training")
    if config.SAVE_MODEL:
        print("  - *.pth.tar: Model checkpoints")
    if config.SAVE_HISTORY:
        print("  - plot_data/: Training history (txt, csv)")
    if config.SAVE_PROGRESS:
        print("  - plot_data/: Training progress plots")
    print("=" * 50)


# スクリプトが直接実行された場合のみmain関数を呼び出し
if __name__ == "__main__":
    main()