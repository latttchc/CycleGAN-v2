"""
CycleGAN評価スクリプト - KID (Kernel Inception Distance) 計算
"""
import os
import glob
import numpy as np
import torch
from PIL import Image
from tqdm import tqdm
from torch.nn.functional import adaptive_avg_pool2d
from sklearn.metrics.pairwise import polynomial_kernel
from torchvision import models, transforms

import config
from generator import Generator


class KIDEvaluator:
    """
    KID (Kernel Inception Distance) 評価クラス
    
    生成画像と実画像の分布の違いをカーネル法で計算
    """
    
    def __init__(self, device=None):
        """
        初期化
        
        Args:
            device: 計算デバイス（None の場合は config.DEVICE を使用）
        """
        self.device = device or config.DEVICE
        self.model = self._load_inception()
        
        # 画像の前処理（InceptionV3用）
        self.transform = transforms.Compose([
            transforms.Resize((299, 299)),  # InceptionV3の入力サイズ
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                               std=[0.229, 0.224, 0.225])
        ])
    
    def _load_inception(self):
        """InceptionV3モデルの読み込み（特徴量抽出用）"""
        model = models.inception_v3(pretrained=True, transform_input=False)
        # 最後の分類層を除去
        model.fc = torch.nn.Identity()
        model = model.to(self.device)
        model.eval()
        return model
    
    def extract_features(self, image_paths, batch_size=32):
        """
        画像から特徴量を抽出
        
        Args:
            image_paths: 画像ファイルパスのリスト
            batch_size: バッチサイズ
            
        Returns:
            np.ndarray: 特徴量 [N, 2048]
        """
        features = []
        
        for i in tqdm(range(0, len(image_paths), batch_size), desc="特徴量抽出"):
            batch_paths = image_paths[i:i + batch_size]
            batch_images = []
            
            for path in batch_paths:
                img = Image.open(path).convert('RGB')
                img = self.transform(img)
                batch_images.append(img)
            
            batch = torch.stack(batch_images).to(self.device)
            
            with torch.no_grad():
                feat = self.model(batch)
                features.append(feat.cpu().numpy())
        
        return np.concatenate(features, axis=0)
    
    def extract_features_from_generator(self, generator, source_paths, batch_size=32):
        """
        Generatorで変換した画像から特徴量を抽出
        
        Args:
            generator: 学習済みGeneratorモデル
            source_paths: 変換元画像のパスリスト
            batch_size: バッチサイズ
            
        Returns:
            np.ndarray: 特徴量 [N, 2048]
        """
        features = []
        generator.eval()
        
        # CycleGAN用の前処理
        cyclegan_transform = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])
        
        for i in tqdm(range(0, len(source_paths), batch_size), desc="生成画像の特徴量抽出"):
            batch_paths = source_paths[i:i + batch_size]
            batch_images = []
            
            for path in batch_paths:
                img = Image.open(path).convert('RGB')
                img = cyclegan_transform(img)
                batch_images.append(img)
            
            batch = torch.stack(batch_images).to(self.device)
            
            with torch.no_grad():
                # CycleGANで画像変換
                generated = generator(batch)
                
                # [-1, 1] → [0, 1] に変換
                generated = (generated + 1) / 2
                
                # InceptionV3用にリサイズと正規化
                generated = torch.nn.functional.interpolate(
                    generated, size=(299, 299), mode='bilinear', align_corners=False
                )
                generated = transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], 
                    std=[0.229, 0.224, 0.225]
                )(generated)
                
                # 特徴量抽出
                feat = self.model(generated)
                features.append(feat.cpu().numpy())
        
        return np.concatenate(features, axis=0)
    
    def compute_kid(self, features_real, features_fake, 
                    n_subsets=100, subset_size=1000):
        """
        KIDを計算
        
        Args:
            features_real: 実画像の特徴量 [N, D]
            features_fake: 生成画像の特徴量 [M, D]
            n_subsets: サブセット数（統計的安定性のため）
            subset_size: 各サブセットのサイズ
            
        Returns:
            tuple: (KID平均値, KID標準偏差)
        """
        # サブセットサイズの調整
        subset_size = min(subset_size, len(features_real), len(features_fake))
        
        kid_values = []
        
        for _ in tqdm(range(n_subsets), desc="KID計算"):
            # ランダムサンプリング
            idx_real = np.random.choice(len(features_real), subset_size, replace=False)
            idx_fake = np.random.choice(len(features_fake), subset_size, replace=False)
            
            real_subset = features_real[idx_real]
            fake_subset = features_fake[idx_fake]
            
            # 多項式カーネルでMMDを計算
            kid = self._polynomial_mmd(real_subset, fake_subset)
            kid_values.append(kid)
        
        return np.mean(kid_values), np.std(kid_values)
    
    def _polynomial_mmd(self, X, Y, degree=3, gamma=None, coef0=1):
        """
        多項式カーネルを用いたMMD計算
        
        k(x, y) = (γ * <x, y> + coef0)^degree
        """
        K_XX = polynomial_kernel(X, degree=degree, gamma=gamma, coef0=coef0)
        K_YY = polynomial_kernel(Y, degree=degree, gamma=gamma, coef0=coef0)
        K_XY = polynomial_kernel(X, Y, degree=degree, gamma=gamma, coef0=coef0)
        
        m = K_XX.shape[0]
        
        # Unbiased MMD^2 estimator
        np.fill_diagonal(K_XX, 0)
        np.fill_diagonal(K_YY, 0)
        
        mmd2 = (K_XX.sum() + K_YY.sum()) / (m * (m - 1)) - 2 * K_XY.mean()
        
        return mmd2


def evaluate_cyclegan(checkpoint_gen_b=None, checkpoint_gen_a=None):
    """
    CycleGANの評価を実行
    
    Args:
        checkpoint_gen_b: A→B Generator のチェックポイントパス
        checkpoint_gen_a: B→A Generator のチェックポイントパス
    """
    print("=" * 60)
    print("CycleGAN KID Evaluation")
    print("=" * 60)
    
    # デフォルトのチェックポイント
    checkpoint_gen_b = checkpoint_gen_b or config.CHECKPOINT_GEN_B
    checkpoint_gen_a = checkpoint_gen_a or config.CHECKPOINT_GEN_A
    
    # 評価器の初期化
    evaluator = KIDEvaluator()
    
    # 画像パスの取得
    real_a_paths = glob.glob(os.path.join(config.VAL_DIR, "testA", "*.jpg")) + \
                   glob.glob(os.path.join(config.VAL_DIR, "testA", "*.png"))
    real_b_paths = glob.glob(os.path.join(config.VAL_DIR, "testB", "*.jpg")) + \
                   glob.glob(os.path.join(config.VAL_DIR, "testB", "*.png"))
    
    print(f"ドメインA画像数: {len(real_a_paths)}")
    print(f"ドメインB画像数: {len(real_b_paths)}")
    
    if len(real_a_paths) == 0 or len(real_b_paths) == 0:
        print("エラー: テスト画像が見つかりません")
        return
    
    # Generatorの読み込み
    gen_B = Generator(img_channels=3, num_residuals=9).to(config.DEVICE)
    gen_A = Generator(img_channels=3, num_residuals=9).to(config.DEVICE)
    
    if os.path.exists(checkpoint_gen_b):
        checkpoint = torch.load(checkpoint_gen_b, map_location=config.DEVICE)
        gen_B.load_state_dict(checkpoint["state_dict"])
        print(f"✓ Generator B loaded from {checkpoint_gen_b}")
    else:
        print(f"警告: {checkpoint_gen_b} が見つかりません")
        return
    
    if os.path.exists(checkpoint_gen_a):
        checkpoint = torch.load(checkpoint_gen_a, map_location=config.DEVICE)
        gen_A.load_state_dict(checkpoint["state_dict"])
        print(f"✓ Generator A loaded from {checkpoint_gen_a}")
    else:
        print(f"警告: {checkpoint_gen_a} が見つかりません")
        return
    
    print("\n" + "-" * 60)
    print("評価開始...")
    print("-" * 60)
    
    # A→B変換の評価
    print("\n[A→B 変換の評価]")
    features_real_b = evaluator.extract_features(real_b_paths)
    features_fake_b = evaluator.extract_features_from_generator(gen_B, real_a_paths)
    
    kid_ab_mean, kid_ab_std = evaluator.compute_kid(features_real_b, features_fake_b)
    print(f"KID (A→B): {kid_ab_mean:.4f} ± {kid_ab_std:.4f}")
    
    # B→A変換の評価
    print("\n[B→A 変換の評価]")
    features_real_a = evaluator.extract_features(real_a_paths)
    features_fake_a = evaluator.extract_features_from_generator(gen_A, real_b_paths)
    
    kid_ba_mean, kid_ba_std = evaluator.compute_kid(features_real_a, features_fake_a)
    print(f"KID (B→A): {kid_ba_mean:.4f} ± {kid_ba_std:.4f}")
    
    # 結果のサマリー
    print("\n" + "=" * 60)
    print("評価結果サマリー")
    print("=" * 60)
    print(f"KID (A→B): {kid_ab_mean:.4f} ± {kid_ab_std:.4f}")
    print(f"KID (B→A): {kid_ba_mean:.4f} ± {kid_ba_std:.4f}")
    print(f"平均 KID:  {(kid_ab_mean + kid_ba_mean) / 2:.4f}")
    print("=" * 60)
    print("\n※ KIDは低いほど良い（0に近いほど実画像の分布に近い）")
    
    return {
        'kid_ab': (kid_ab_mean, kid_ab_std),
        'kid_ba': (kid_ba_mean, kid_ba_std),
        'kid_avg': (kid_ab_mean + kid_ba_mean) / 2
    }


if __name__ == "__main__":
    evaluate_cyclegan()
