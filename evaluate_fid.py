"""
CycleGAN評価スクリプト - FID (Fréchet Inception Distance) 計算
"""
import os
import glob
import numpy as np
import torch
from PIL import Image
from tqdm import tqdm
from scipy.linalg import sqrtm
from torchvision import models, transforms

import config
from generator import Generator


class FIDEvaluator:
    """
    FID (Fréchet Inception Distance) 評価クラス
    
    生成画像と実画像の特徴量分布の差をフレシェ距離で計算
    """
    
    def __init__(self, device=None):
        """
        初期化
        
        Args:
            device: 計算デバイス
        """
        self.device = device or config.DEVICE
        self.model = self._load_inception()
        
        # InceptionV3用の前処理
        self.transform = transforms.Compose([
            transforms.Resize((299, 299)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                               std=[0.229, 0.224, 0.225])
        ])
    
    def _load_inception(self):
        """InceptionV3モデルの読み込み"""
        model = models.inception_v3(pretrained=True, transform_input=False)
        # 最後の分類層を除去して特徴量のみを出力
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
        
        for i in tqdm(range(0, len(image_paths), batch_size), desc="FID: 特徴量抽出"):
            batch_paths = image_paths[i:i + batch_size]
            batch_images = []
            
            for path in batch_paths:
                try:
                    img = Image.open(path).convert('RGB')
                    img = self.transform(img)
                    batch_images.append(img)
                except Exception as e:
                    print(f"警告: {path} の読み込みに失敗しました: {e}")
                    continue
            
            if len(batch_images) == 0:
                continue
            
            batch = torch.stack(batch_images).to(self.device)
            
            with torch.no_grad():
                feat = self.model(batch)
                features.append(feat.cpu().numpy())
        
        if len(features) == 0:
            raise ValueError("特徴量抽出に失敗しました")
        
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
        
        for i in tqdm(range(0, len(source_paths), batch_size), 
                     desc="FID: 生成画像の特徴量抽出"):
            batch_paths = source_paths[i:i + batch_size]
            batch_images = []
            
            for path in batch_paths:
                try:
                    img = Image.open(path).convert('RGB')
                    img = cyclegan_transform(img)
                    batch_images.append(img)
                except Exception as e:
                    print(f"警告: {path} の読み込みに失敗しました: {e}")
                    continue
            
            if len(batch_images) == 0:
                continue
            
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
        
        if len(features) == 0:
            raise ValueError("生成画像の特徴量抽出に失敗しました")
        
        return np.concatenate(features, axis=0)
    
    def compute_fid(self, features_real, features_fake):
        """
        FIDを計算
        
        Args:
            features_real: 実画像の特徴量 [N, D]
            features_fake: 生成画像の特徴量 [M, D]
            
        Returns:
            float: FID値
        """
        # 平均と共分散を計算
        mu_real = np.mean(features_real, axis=0)
        mu_fake = np.mean(features_fake, axis=0)
        
        sigma_real = np.cov(features_real.T)
        sigma_fake = np.cov(features_fake.T)
        
        # フレシェ距離を計算
        # 1. 平均間の二乗ユークリッド距離
        mean_diff = np.sum((mu_real - mu_fake) ** 2)
        
        # 2. 共分散行列間のフレシェ距離
        # sqrt(Σ_real) @ sqrt(Σ_fake) の計算
        try:
            # 共分散行列の平方根を計算
            sqrt_sigma_real = sqrtm(sigma_real).real
            sqrt_sigma_fake = sqrtm(sigma_fake).real
            
            # 共分散行列間の距離
            cov_diff = sqrtm(
                sqrt_sigma_real @ sigma_fake @ sqrt_sigma_real
            ).real
            
            cov_distance = np.trace(
                sigma_real + sigma_fake - 2 * cov_diff
            )
        except np.linalg.LinAlgError as e:
            print(f"警告: 共分散行列の計算に失敗しました: {e}")
            cov_distance = 0
        
        # FID = 平均差 + 共分散差
        fid = mean_diff + cov_distance
        
        return np.sqrt(max(fid, 0))  # 数値誤差対策


def evaluate_cyclegan_fid(checkpoint_gen_b=None, checkpoint_gen_a=None):
    """
    CycleGANのFID評価を実行
    
    Args:
        checkpoint_gen_b: A→B Generator のチェックポイントパス
        checkpoint_gen_a: B→A Generator のチェックポイントパス
    """
    print("=" * 60)
    print("CycleGAN FID Evaluation")
    print("=" * 60)
    
    # デフォルトのチェックポイント
    checkpoint_gen_b = checkpoint_gen_b or config.CHECKPOINT_GEN_B
    checkpoint_gen_a = checkpoint_gen_a or config.CHECKPOINT_GEN_A
    
    # 評価器の初期化
    evaluator = FIDEvaluator()
    
    # 画像パスの取得
    real_a_paths = (glob.glob(os.path.join(config.VAL_DIR, "testA", "*.jpg")) + 
                   glob.glob(os.path.join(config.VAL_DIR, "testA", "*.png")))
    real_b_paths = (glob.glob(os.path.join(config.VAL_DIR, "testB", "*.jpg")) + 
                   glob.glob(os.path.join(config.VAL_DIR, "testB", "*.png")))
    
    print(f"ドメインA画像数: {len(real_a_paths)}")
    print(f"ドメインB画像数: {len(real_b_paths)}")
    
    if len(real_a_paths) == 0 or len(real_b_paths) == 0:
        print("エラー: テスト画像が見つかりません")
        return None
    
    # Generatorの読み込み
    gen_B = Generator(img_channels=3, num_residuals=9).to(config.DEVICE)
    gen_A = Generator(img_channels=3, num_residuals=9).to(config.DEVICE)
    
    if os.path.exists(checkpoint_gen_b):
        checkpoint = torch.load(checkpoint_gen_b, map_location=config.DEVICE)
        gen_B.load_state_dict(checkpoint["state_dict"])
        print(f"✓ Generator B loaded from {checkpoint_gen_b}")
    else:
        print(f"警告: {checkpoint_gen_b} が見つかりません")
        return None
    
    if os.path.exists(checkpoint_gen_a):
        checkpoint = torch.load(checkpoint_gen_a, map_location=config.DEVICE)
        gen_A.load_state_dict(checkpoint["state_dict"])
        print(f"✓ Generator A loaded from {checkpoint_gen_a}")
    else:
        print(f"警告: {checkpoint_gen_a} が見つかりません")
        return None
    
    print("\n" + "-" * 60)
    print("FID評価開始...")
    print("-" * 60)
    
    # A→B変換の評価
    print("\n[A→B 変換の評価]")
    features_real_b = evaluator.extract_features(real_b_paths, config.BATCH_SIZE)
    features_fake_b = evaluator.extract_features_from_generator(gen_B, real_a_paths, config.BATCH_SIZE)
    
    fid_ab = evaluator.compute_fid(features_real_b, features_fake_b)
    print(f"FID (A→B): {fid_ab:.2f}")
    
    # B→A変換の評価
    print("\n[B→A 変換の評価]")
    features_real_a = evaluator.extract_features(real_a_paths, config.BATCH_SIZE)
    features_fake_a = evaluator.extract_features_from_generator(gen_A, real_b_paths, config.BATCH_SIZE)
    
    fid_ba = evaluator.compute_fid(features_real_a, features_fake_a)
    print(f"FID (B→A): {fid_ba:.2f}")
    
    # 結果のサマリー
    print("\n" + "=" * 60)
    print("FID評価結果サマリー")
    print("=" * 60)
    print(f"FID (A→B): {fid_ab:.2f}")
    print(f"FID (B→A): {fid_ba:.2f}")
    print(f"平均 FID:  {(fid_ab + fid_ba) / 2:.2f}")
    print("=" * 60)
    print("\n※ FIDは低いほど良い")
    print("  - < 10: 優秀")
    print("  - 10-20: 良好")
    print("  - 20-50: 普通")
    print("  - > 50: 改善が必要")
    
    return {
        'fid_ab': fid_ab,
        'fid_ba': fid_ba,
        'fid_avg': (fid_ab + fid_ba) / 2
    }


if __name__ == "__main__":
    evaluate_cyclegan_fid()