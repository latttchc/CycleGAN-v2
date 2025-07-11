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


def train_fn(disc_A, disc_B, gen_B, gen_A, loader, opt_disc, opt_gen, l1, mse, d_scaler, g_scaler):
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
    """
    # Discriminatorの性能指標を追跡するための変数
    A_reals = 0  # ドメインAの本物画像に対するDiscriminatorスコアの累計
    A_fakes = 0  # ドメインAの偽物画像に対するDiscriminatorスコアの累計
    
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
        d_scaler.update                         # スケーラーの更新（注：()が抜けている）

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

            # 全Generator損失の計算
            G_loss = (
                loss_G_B +                                          # B→A Generatorの敵対的損失
                loss_G_A +                                          # A→B Generatorの敵対的損失
                cycle_trainB_loss * config.LAMBDA_CYCLE +           # サイクル一貫性損失（重み付き）
                cycle_trainA_loss * config.LAMBDA_CYCLE +           # サイクル一貫性損失（重み付き）
                identity_trainA_loss *config.LAMBDA_IDENTITY +      # アイデンティティ損失（重み付き）
                identity_trainB_loss * config.LAMBDA_IDENTITY       # アイデンティティ損失（重み付き）
            )

        # Generatorの重み更新
        opt_gen.zero_grad()                     # 勾配をリセット
        g_scaler.scale(G_loss).backward()       # 逆伝播（スケーリング付き）
        g_scaler.step(opt_gen)                  # パラメータ更新
        g_scaler.update()                       # スケーラーの更新

        # 定期的に生成画像を保存（200バッチごと）
        if idx % 200 == 0:
            # [-1, 1]の値域を[0, 1]に変換してから保存
            save_image(fake_trainA * 0.5 + 0.5, f"saved_images/fake_trainA_{idx}.png")
            save_image(fake_trainB * 0.5 + 0.5, f"saved_images/fake_trainB_{idx}.png")

        # プログレスバーに現在の統計情報を表示
        loop.set_postfix(
            A_real=A_reals / (idx + 1),  # ドメインAの本物画像への平均Discriminatorスコア
            A_fake=A_fakes / (idx + 1)   # ドメインAの偽物画像への平均Discriminatorスコア
        )


def main():
    """
    CycleGANのメイン訓練ループ
    
    1. モデルとオプティマイザーの初期化
    2. データセットとDataLoaderの準備
    3. 各エポックでの訓練実行
    4. モデルのチェックポイント保存
    """
    # ========== モデルの初期化 ==========
    # 2つのDiscriminator：それぞれのドメインの真偽を判別
    disc_A = Discriminator(in_channels=3).to(config.DEVICE)  # ドメインA用
    disc_B = Discriminator(in_channels=3).to(config.DEVICE)  # ドメインB用
    
    # 2つのGenerator：相互変換を行う
    gen_B = Generator(img_channels=3, num_residuals=9).to(config.DEVICE)  # A→B変換
    gen_A = Generator(img_channels=3, num_residuals=9).to(config.DEVICE)  # B→A変換

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

    # ========== メイン訓練ループ ==========
    for epoch in range(config.NUM_EPOCHS):
        print(f"Epoch [{epoch}/{config.NUM_EPOCHS}]")
        
        # 1エポック分の訓練を実行
        train_fn(disc_A, disc_B, gen_B, gen_A, loader, opt_disc, opt_gen, l1, mse, d_scaler, g_scaler)

        # ========== モデルの保存 ==========
        if config.SAVE_MODEL:
            # 各モデルのチェックポイントを保存
            save_checkpoint(gen_A, opt_gen, filename=config.CHECKPOINT_GEN_A)
            save_checkpoint(gen_B, opt_gen, filename=config.CHECKPOINT_GEN_B)
            save_checkpoint(disc_A, opt_disc, filename=config.CHECKPOINT_CRITIC_A)
            save_checkpoint(disc_B, opt_disc, filename=config.CHECKPOINT_CRITIC_B)

# スクリプトが直接実行された場合のみmain関数を呼び出し
if __name__ == "__main__":
    main()