import torch 
import torch.nn as nn
import torch.optim as optim
import config
from tqdm import tqdm
from torch.utils.data import DataLoader
from torchvision.utils import save_image
from discriminator import Discriminator
from generator import Generator
from dataset import CustomDataset
from utils import save_checkpoint, load_checkpoint
import matplotlib.pyplot as plt
import os
import numpy as np


def train_fn(disc_A, disc_B, gen_B, gen_A, loader, opt_disc, opt_gen, l1, mse, d_scaler, g_scaler, epoch):
    """
    CycleGANの1エポック分の訓練を実行する関数
    
    Args:
        disc_A: ドメインA用のDiscriminator（本物のAか偽物のAかを判別）
        disc_B: ドメインB用のDiscriminator（本物のBか偽物のBかを判別）
        gen_B: A→B変換用のGenerator（A画像をB画像に変換）
        gen_A: B→A変換用のGenerator（B画像をA画像に変換）
        loader: データローダー（trainA, trainBのペアを提供）
        opt_disc: Discriminator用最適化器
        opt_gen: Generator用最適化器
        l1: L1損失関数（サイクル一貫性・アイデンティティ損失用）
        mse: MSE損失関数（敵対的損失用）
        d_scaler: Discriminator用のGradScaler（mixed precision training）
        g_scaler: Generator用のGradScaler（mixed precision training）
        epoch: 現在のエポック番号
    
    Returns:
        dict: エポックの評価指標を含む辞書
    """
    # 各種損失と評価指標の累積値を記録
    total_D_loss = 0.0          # Discriminator損失の累積
    total_G_loss = 0.0          # Generator損失の累積
    total_cycle_loss = 0.0      # サイクル一貫性損失の累積
    total_identity_loss = 0.0   # アイデンティティ損失の累積
    total_adversarial_loss = 0.0 # 敵対的損失の累積
    
    # Discriminatorの性能指標を追跡するための変数
    A_reals = 0  # ドメインAの本物画像に対するDiscriminatorスコアの累計
    A_fakes = 0  # ドメインAの偽物画像に対するDiscriminatorスコアの累計
    B_reals = 0  # ドメインBの本物画像に対するDiscriminatorスコアの累計
    B_fakes = 0  # ドメインBの偽物画像に対するDiscriminatorスコアの累計
    
    # プログレスバーでバッチ処理の進行状況を表示
    loop = tqdm(loader, leave=True)

    # 各バッチに対して訓練を実行
    for idx, (trainB, trainA) in enumerate(loop):
        # データをGPU/CPUに移動
        trainB = trainB.to(config.DEVICE)  # ドメインBの実画像
        trainA = trainA.to(config.DEVICE)  # ドメインAの実画像

        # ========== Discriminatorの訓練 ==========
        # Mixed Precision Trainingを使用してメモリ効率を向上
        with torch.cuda.amp.autocast():
            # ドメインA用Discriminatorの訓練
            fake_trainA = gen_A(trainB)  # B→A変換で偽のA画像を生成
            D_A_real = disc_A(trainA)    # 本物のA画像に対するDiscriminatorの出力
            D_A_fake = disc_A(fake_trainA.detach())  # 偽のA画像に対するDiscriminatorの出力
                                                     # detach()でGeneratorへの勾配伝播を防ぐ

            # Discriminatorの性能指標を記録
            A_reals += D_A_real.mean().item()  # 本物画像への平均スコア
            A_fakes += D_A_fake.mean().item()  # 偽物画像への平均スコア

            # ドメインA用Discriminatorの損失計算
            D_A_real_loss = mse(D_A_real, torch.ones_like(D_A_real))   # 本物→1に近づける
            D_A_fake_loss = mse(D_A_fake, torch.zeros_like(D_A_fake))  # 偽物→0に近づける
            D_A_loss = D_A_real_loss + D_A_fake_loss

            # ドメインB用Discriminatorの訓練
            fake_trainB = gen_B(trainA)  # A→B変換で偽のB画像を生成
            D_B_real = disc_B(trainB)    # 本物のB画像に対するDiscriminatorの出力
            D_B_fake = disc_B(fake_trainB.detach())  # 偽のB画像に対するDiscriminatorの出力

            # ドメインB用Discriminatorの性能指標を記録
            B_reals += D_B_real.mean().item()  # 本物画像への平均スコア
            B_fakes += D_B_fake.mean().item()  # 偽物画像への平均スコア

            # ドメインB用Discriminatorの損失計算
            D_B_real_loss = mse(D_B_real, torch.ones_like(D_B_real))   # 本物→1に近づける
            D_B_fake_loss = mse(D_B_fake, torch.zeros_like(D_B_fake))  # 偽物→0に近づける
            D_B_loss = D_B_real_loss + D_B_fake_loss

            # 全Discriminator損失（2つのDiscriminatorの平均）
            D_loss = (D_A_loss + D_B_loss) / 2

        # Discriminatorの重み更新
        opt_disc.zero_grad()                    # 勾配をリセット
        d_scaler.scale(D_loss).backward()       # 逆伝播（スケーリング付き）
        d_scaler.step(opt_disc)                 # パラメータ更新
        d_scaler.update()                       # スケーラーの更新

        # ========== Generatorの訓練 ==========
        with torch.cuda.amp.autocast():
            # 敵対的損失：DiscriminatorをだますためのGenerator損失
            D_A_fake = disc_A(fake_trainA)  # 偽のA画像（今度はdetach()なし）
            D_B_fake = disc_B(fake_trainB)  # 偽のB画像（今度はdetach()なし）
            loss_G_A = mse(D_A_fake, torch.ones_like(D_A_fake))  # 偽物→1に近づける（Discriminatorをだます）
            loss_G_B = mse(D_B_fake, torch.ones_like(D_B_fake))  # 偽物→1に近づける（Discriminatorをだます）

            # サイクル一貫性損失：A→B→A、B→A→Bの変換で元画像を復元
            cycle_trainB = gen_B(fake_trainA)  # B→A→Bのサイクル
            cycle_trainA = gen_A(fake_trainB)  # A→B→Aのサイクル
            cycle_trainA_loss = l1(trainB, cycle_trainB)  # 元のBと復元されたBの差
            cycle_trainB_loss = l1(trainA, cycle_trainA)  # 元のAと復元されたAの差

            # アイデンティティ損失：同じドメインの画像は変換しない
            identity_trainB = gen_B(trainB)  # B画像をB→A Generatorに入力
            identity_trainA = gen_A(trainA)  # A画像をA→B Generatorに入力
            identity_trainB_loss = l1(trainB, identity_trainB)  # 変換前後で同じであるべき
            identity_trainA_loss = l1(trainA, identity_trainA)  # 変換前後で同じであるべき

            # 各損失成分の計算
            adversarial_loss = loss_G_A + loss_G_B
            cycle_loss = cycle_trainA_loss + cycle_trainB_loss
            identity_loss = identity_trainA_loss + identity_trainB_loss

            # 全Generator損失の計算
            G_loss = (
                adversarial_loss +                          # 敵対的損失
                cycle_loss * config.LAMBDA_CYCLE +          # サイクル一貫性損失（重み付き）
                identity_loss * config.LAMBDA_IDENTITY      # アイデンティティ損失（重み付き）
            )

        # Generatorの重み更新
        opt_gen.zero_grad()                     # 勾配をリセット
        g_scaler.scale(G_loss).backward()       # 逆伝播（スケーリング付き）
        g_scaler.step(opt_gen)                  # パラメータ更新
        g_scaler.update()                       # スケーラーの更新

        # 損失値の累積（平均計算用）
        total_D_loss += D_loss.item()
        total_G_loss += G_loss.item()
        total_cycle_loss += cycle_loss.item()
        total_identity_loss += identity_loss.item()
        total_adversarial_loss += adversarial_loss.item()

        # 定期的に生成画像を保存（200バッチごと）
        if idx % 200 == 0:
            # [-1, 1]の値域を[0, 1]に変換してから保存
            save_image(fake_trainA * 0.5 + 0.5, f"saved_images/fake_trainA_epoch{epoch}_batch{idx}.png")
            save_image(fake_trainB * 0.5 + 0.5, f"saved_images/fake_trainB_epoch{epoch}_batch{idx}.png")

        # プログレスバーに現在の統計情報を表示
        loop.set_postfix(
            D_loss=f"{D_loss.item():.4f}",
            G_loss=f"{G_loss.item():.4f}",
            A_real=f"{A_reals / (idx + 1):.3f}",
            A_fake=f"{A_fakes / (idx + 1):.3f}"
        )

    # エポック終了時の平均値を計算
    num_batches = len(loader)
    epoch_metrics = {
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
    
    return epoch_metrics


def plot_training_progress(history, save_path="plot_data/training_progress.png"):
    """
    学習進捗をグラフで可視化
    
    Args:
        history (dict): 各エポックの評価指標を含む辞書
        save_path (str): グラフの保存パス
    """
    epochs = range(1, len(history['D_loss']) + 1)
    
    # 2x3のサブプロットを作成
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle('CycleGAN Training Progress', fontsize=16, fontweight='bold')
    
    # 1. Discriminator vs Generator損失
    axes[0, 0].plot(epochs, history['D_loss'], 'b-', label='Discriminator Loss', linewidth=2)
    axes[0, 0].plot(epochs, history['G_loss'], 'r-', label='Generator Loss', linewidth=2)
    axes[0, 0].set_title('Discriminator vs Generator Loss')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # 2. Generator損失の内訳
    axes[0, 1].plot(epochs, history['adversarial_loss'], 'orange', label='Adversarial Loss', linewidth=2)
    axes[0, 1].plot(epochs, history['cycle_loss'], 'green', label='Cycle Loss', linewidth=2)
    axes[0, 1].plot(epochs, history['identity_loss'], 'purple', label='Identity Loss', linewidth=2)
    axes[0, 1].set_title('Generator Loss Components')
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('Loss')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # 3. ドメインAのDiscriminatorスコア
    axes[0, 2].plot(epochs, history['A_real_score'], 'blue', label='Real A Score', linewidth=2)
    axes[0, 2].plot(epochs, history['A_fake_score'], 'red', label='Fake A Score', linewidth=2)
    axes[0, 2].axhline(y=0.5, color='gray', linestyle='--', alpha=0.7, label='Ideal Balance')
    axes[0, 2].set_title('Domain A Discriminator Scores')
    axes[0, 2].set_xlabel('Epoch')
    axes[0, 2].set_ylabel('Score')
    axes[0, 2].legend()
    axes[0, 2].grid(True, alpha=0.3)
    
    # 4. ドメインBのDiscriminatorスコア
    axes[1, 0].plot(epochs, history['B_real_score'], 'blue', label='Real B Score', linewidth=2)
    axes[1, 0].plot(epochs, history['B_fake_score'], 'red', label='Fake B Score', linewidth=2)
    axes[1, 0].axhline(y=0.5, color='gray', linestyle='--', alpha=0.7, label='Ideal Balance')
    axes[1, 0].set_title('Domain B Discriminator Scores')
    axes[1, 0].set_xlabel('Epoch')
    axes[1, 0].set_ylabel('Score')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    # 5. 損失比率（Generator vs Discriminator）
    d_g_ratio = [d/g if g > 0 else 0 for d, g in zip(history['D_loss'], history['G_loss'])]
    axes[1, 1].plot(epochs, d_g_ratio, 'purple', label='D_loss / G_loss', linewidth=2)
    axes[1, 1].axhline(y=1.0, color='gray', linestyle='--', alpha=0.7, label='Balanced Training')
    axes[1, 1].set_title('Loss Ratio (D_loss / G_loss)')
    axes[1, 1].set_xlabel('Epoch')
    axes[1, 1].set_ylabel('Ratio')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    
    # 6. 全体的な学習進捗（移動平均）
    if len(epochs) > 1:
        # 簡単な移動平均を計算
        window_size = max(1, len(epochs) // 5)
        d_smooth = np.convolve(history['D_loss'], np.ones(window_size)/window_size, mode='valid')
        g_smooth = np.convolve(history['G_loss'], np.ones(window_size)/window_size, mode='valid')
        smooth_epochs = range(window_size, len(epochs) + 1)
        
        axes[1, 2].plot(smooth_epochs, d_smooth, 'b-', label='D_loss (smoothed)', linewidth=2)
        axes[1, 2].plot(smooth_epochs, g_smooth, 'r-', label='G_loss (smoothed)', linewidth=2)
    else:
        axes[1, 2].plot(epochs, history['D_loss'], 'b-', label='D_loss', linewidth=2)
        axes[1, 2].plot(epochs, history['G_loss'], 'r-', label='G_loss', linewidth=2)
    
    axes[1, 2].set_title('Smoothed Loss Trends')
    axes[1, 2].set_xlabel('Epoch')
    axes[1, 2].set_ylabel('Loss')
    axes[1, 2].legend()
    axes[1, 2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()
    print(f"Training progress plot saved to {save_path}")


def save_training_history(history, save_path="plot_data/training_history.txt"):
    """
    学習履歴をテキストファイルに保存
    
    Args:
        history (dict): 学習履歴
        save_path (str): 保存パス
    """
    with open(save_path, 'w') as f:
        # ヘッダー
        f.write("Epoch\tD_Loss\tG_Loss\tAdv_Loss\tCycle_Loss\tIdentity_Loss\t")
        f.write("A_Real\tA_Fake\tB_Real\tB_Fake\n")
        
        # データ
        for i in range(len(history['D_loss'])):
            f.write(f"{i+1}\t{history['D_loss'][i]:.6f}\t{history['G_loss'][i]:.6f}\t")
            f.write(f"{history['adversarial_loss'][i]:.6f}\t{history['cycle_loss'][i]:.6f}\t")
            f.write(f"{history['identity_loss'][i]:.6f}\t{history['A_real_score'][i]:.6f}\t")
            f.write(f"{history['A_fake_score'][i]:.6f}\t{history['B_real_score'][i]:.6f}\t")
            f.write(f"{history['B_fake_score'][i]:.6f}\n")
    
    print(f"Training history saved to {save_path}")


def main():
    """
    CycleGANのメイン訓練ループ
    
    1. モデルとオプティマイザーの初期化
    2. データセットとDataLoaderの準備
    3. 各エポックでの訓練実行
    4. モデルのチェックポイント保存
    5. 学習進捗の可視化
    """
    # 保存ディレクトリの作成
    os.makedirs("saved_images", exist_ok=True)
    os.makedirs("plot_data", exist_ok=True)
    
    # ========== モデルの初期化 ==========
    # 2つのDiscriminator：それぞれのドメインの真偽を判別
    disc_A = Discriminator(in_channels=3).to(config.DEVICE)  # ドメインA用
    disc_B = Discriminator(in_channels=3).to(config.DEVICE)  # ドメインB用
    
    # 2つのGenerator：相互変換を行う
    gen_B = Generator(img_channels=3).to(config.DEVICE)  # A→B変換
    gen_A = Generator(img_channels=3).to(config.DEVICE)  # B→A変換

    # ========== オプティマイザーの初期化 ==========
    # Discriminator用オプティマイザー：2つのDiscriminatorを同時に最適化
    opt_disc = optim.Adam(
        list(disc_A.parameters()) + list(disc_B.parameters()),  # 両Discriminatorのパラメータ
        lr=config.LEARNING_RATE,    # 学習率
        betas=(0.5, 0.999)          # GANで一般的なbeta値（momentum係数）
    )
    
    # Generator用オプティマイザー：2つのGeneratorを同時に最適化
    opt_gen = optim.Adam(
        list(gen_B.parameters()) + list(gen_A.parameters()),    # 両Generatorのパラメータ
        lr=config.LEARNING_RATE,    # 学習率
        betas=(0.5, 0.999)          # GANで一般的なbeta値
    )

    # ========== 損失関数の定義 ==========
    l1 = nn.L1Loss()   # L1損失：サイクル一貫性・アイデンティティ損失用
    mse = nn.MSELoss() # MSE損失：敵対的損失用（LSGAN形式）

    # ========== チェックポイントの読み込み ==========
    if config.LOAD_MODEL:
        # 事前訓練されたモデルがある場合は読み込み
        load_checkpoint(config.CHECKPOINT_CRITIC_A, disc_A, opt_disc, config.LEARNING_RATE)
        load_checkpoint(config.CHECKPOINT_CRITIC_B, disc_B, opt_disc, config.LEARNING_RATE)
        load_checkpoint(config.CHECKPOINT_GEN_A, gen_A, opt_gen, config.LEARNING_RATE)
        load_checkpoint(config.CHECKPOINT_GEN_B, gen_B, opt_gen, config.LEARNING_RATE)

    # ========== データセットとDataLoaderの準備 ==========
    # ペアになっていないドメインAとBの画像を読み込み
    dataset = CustomDataset(root_a=config.TRAIN_DIR + "/trainA", root_b=config.TRAIN_DIR + "/trainB", transform=config.transforms)
    loader = DataLoader(dataset, batch_size=config.BATCH_SIZE, shuffle=True, num_workers=config.NUM_WORKERS)

    # ========== Mixed Precision Training用のScaler ==========
    d_scaler = torch.cuda.amp.GradScaler()  # Discriminator用
    g_scaler = torch.cuda.amp.GradScaler()  # Generator用

    # ========== 学習履歴の初期化 ==========
    history = {
        'D_loss': [],
        'G_loss': [],
        'cycle_loss': [],
        'identity_loss': [],
        'adversarial_loss': [],
        'A_real_score': [],
        'A_fake_score': [],
        'B_real_score': [],
        'B_fake_score': []
    }

    # ========== メイン訓練ループ ==========
    for epoch in range(config.NUM_EPOCHS):
        print(f"\nEpoch [{epoch+1}/{config.NUM_EPOCHS}]")
        
        # 1エポック分の訓練を実行
        epoch_metrics = train_fn(disc_A, disc_B, gen_B, gen_A, loader, opt_disc, opt_gen, l1, mse, d_scaler, g_scaler, epoch)
        
        # 履歴に追加
        for key, value in epoch_metrics.items():
            history[key].append(value)
        
        # エポック終了時の統計情報を表示
        print(f"Epoch {epoch+1} Summary:")
        print(f"  D_Loss: {epoch_metrics['D_loss']:.4f}")
        print(f"  G_Loss: {epoch_metrics['G_loss']:.4f}")
        print(f"  Cycle_Loss: {epoch_metrics['cycle_loss']:.4f}")
        print(f"  Identity_Loss: {epoch_metrics['identity_loss']:.4f}")
        print(f"  A_Real_Score: {epoch_metrics['A_real_score']:.3f}")
        print(f"  A_Fake_Score: {epoch_metrics['A_fake_score']:.3f}")

        # ========== モデルの保存 ==========
        if config.SAVE_MODEL:
            # 各モデルのチェックポイントを保存
            save_checkpoint(gen_A, opt_gen, filename=config.CHECKPOINT_GEN_A)
            save_checkpoint(gen_B, opt_gen, filename=config.CHECKPOINT_GEN_B)
            save_checkpoint(disc_A, opt_disc, filename=config.CHECKPOINT_CRITIC_A)
            save_checkpoint(disc_B, opt_disc, filename=config.CHECKPOINT_CRITIC_B)

        # ========== 中間結果の可視化 ==========
        # 5エポックごとに途中経過をプロット
        if (epoch + 1) % 5 == 0 or epoch == config.NUM_EPOCHS - 1:
            plot_training_progress(history, f"training_progress_epoch_{epoch+1}.png")

    # ========== 最終結果の保存 ==========
    # 最終的な学習履歴を保存
    if config.SAVE_HISTORY:
        save_training_history(history, "final_training_history.txt")
    if config.SAVE_PROGRESS:
        plot_training_progress(history, "final_training_progress.png")
    
    print("\n" + "="*50)
    print("Training completed successfully!")
    print("="*50)


# スクリプトが直接実行された場合のみmain関数を呼び出し
if __name__ == "__main__":
    main()