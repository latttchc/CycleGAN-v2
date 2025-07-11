import matplotlib.pyplot as plt
import numpy as np
import os
from typing import Dict, List, Optional


class TrainVisualizer:
    """
    CycleGAN学習進捗の可視化クラス
    
    グラフの生成、保存、カスタマイズを担当
    """
    
    def __init__(self, save_dir: str = "plot_data"):
        """
        初期化
        
        Args:
            save_dir (str): 結果保存ディレクトリ
        """
        self.save_dir = save_dir
        os.makedirs(save_dir, exist_ok=True)
    
    def plot_training_progress(self, history: Dict, save_path: Optional[str] = None):
        """
        学習進捗をグラフで可視化
        
        Args:
            history (dict): 各エポックの評価指標を含む辞書
            save_path (str, optional): グラフの保存パス
        """
        if save_path is None:
            save_path = os.path.join(self.save_dir, "training_progress.png")
        
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
    
    def plot_loss_only(self, history: Dict, save_path: Optional[str] = None):
        """
        損失のみを可視化（シンプル版）
        
        Args:
            history (dict): 学習履歴
            save_path (str, optional): 保存パス
        """
        if save_path is None:
            save_path = os.path.join(self.save_dir, "loss_only.png")
        
        epochs = range(1, len(history['D_loss']) + 1)
        
        plt.figure(figsize=(10, 6))
        plt.plot(epochs, history['D_loss'], 'b-', label='Discriminator Loss', linewidth=2)
        plt.plot(epochs, history['G_loss'], 'r-', label='Generator Loss', linewidth=2)
        plt.title('CycleGAN Training Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
        print(f"Loss plot saved to {save_path}")


class TrainDataSaver:
    """
    学習データの保存クラス
    
    数値データの保存、読み込み、フォーマットを担当
    """
    
    def __init__(self, save_dir: str = "plot_data"):
        """
        初期化
        
        Args:
            save_dir (str): 保存ディレクトリ
        """
        self.save_dir = save_dir
        os.makedirs(save_dir, exist_ok=True)
    
    def save_training_history(self, history: Dict, save_path: Optional[str] = None):
        """
        学習履歴をテキストファイルに保存
        
        Args:
            history (dict): 学習履歴
            save_path (str, optional): 保存パス
        """
        if save_path is None:
            save_path = os.path.join(self.save_dir, "training_history.txt")
        
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
    
    def save_csv_format(self, history: Dict, save_path: Optional[str] = None):
        """
        CSV形式で学習履歴を保存
        
        Args:
            history (dict): 学習履歴
            save_path (str, optional): 保存パス
        """
        if save_path is None:
            save_path = os.path.join(self.save_dir, "training_history.csv")
        
        with open(save_path, 'w') as f:
            # ヘッダー
            f.write("Epoch,D_Loss,G_Loss,Adversarial_Loss,Cycle_Loss,Identity_Loss,")
            f.write("A_Real_Score,A_Fake_Score,B_Real_Score,B_Fake_Score\n")
            
            # データ
            for i in range(len(history['D_loss'])):
                f.write(f"{i+1},{history['D_loss'][i]:.6f},{history['G_loss'][i]:.6f},")
                f.write(f"{history['adversarial_loss'][i]:.6f},{history['cycle_loss'][i]:.6f},")
                f.write(f"{history['identity_loss'][i]:.6f},{history['A_real_score'][i]:.6f},")
                f.write(f"{history['A_fake_score'][i]:.6f},{history['B_real_score'][i]:.6f},")
                f.write(f"{history['B_fake_score'][i]:.6f}\n")
        
        print(f"Training history (CSV) saved to {save_path}")
    
    def load_training_history(self, load_path: str) -> Dict:
        """
        学習履歴を読み込み
        
        Args:
            load_path (str): 読み込みパス
            
        Returns:
            dict: 学習履歴
        """
        history = {
            'D_loss': [], 'G_loss': [], 'adversarial_loss': [], 'cycle_loss': [],
            'identity_loss': [], 'A_real_score': [], 'A_fake_score': [],
            'B_real_score': [], 'B_fake_score': []
        }
        
        with open(load_path, 'r') as f:
            lines = f.readlines()
            for line in lines[1:]:  # ヘッダーをスキップ
                values = line.strip().split('\t')
                history['D_loss'].append(float(values[1]))
                history['G_loss'].append(float(values[2]))
                history['adversarial_loss'].append(float(values[3]))
                history['cycle_loss'].append(float(values[4]))
                history['identity_loss'].append(float(values[5]))
                history['A_real_score'].append(float(values[6]))
                history['A_fake_score'].append(float(values[7]))
                history['B_real_score'].append(float(values[8]))
                history['B_fake_score'].append(float(values[9]))
        
        print(f"Training history loaded from {load_path}")
        return history


class TrainOptions:
    """
    CycleGAN学習オプションの統合クラス
    
    可視化とデータ保存を統合管理
    """
    
    def __init__(self, save_dir: str = "plot_data"):
        """
        初期化
        
        Args:
            save_dir (str): 結果保存ディレクトリ
        """
        self.save_dir = save_dir
        self.visualizer = TrainVisualizer(save_dir)
        self.data_saver = TrainDataSaver(save_dir)
        
        # 学習進捗の追跡
        self.history = {
            'D_loss': [], 'G_loss': [], 'cycle_loss': [], 'identity_loss': [],
            'adversarial_loss': [], 'A_real_score': [], 'A_fake_score': [],
            'B_real_score': [], 'B_fake_score': []
        }
    
    def update_history(self, epoch_metrics: Dict):
        """
        学習履歴を更新
        
        Args:
            epoch_metrics (dict): エポックの評価指標
        """
        for key, value in epoch_metrics.items():
            if key in self.history:
                self.history[key].append(value)
    
    def save_progress(self, epoch: int, final: bool = False):
        """
        学習進捗を保存
        
        Args:
            epoch (int): 現在のエポック
            final (bool): 最終保存かどうか
        """
        prefix = "final" if final else f"epoch_{epoch}"
        
        # グラフの保存
        plot_path = os.path.join(self.save_dir, f"{prefix}_training_progress.png")
        self.visualizer.plot_training_progress(self.history, plot_path)
        
        # データの保存
        if final:
            txt_path = os.path.join(self.save_dir, f"{prefix}_training_history.txt")
            csv_path = os.path.join(self.save_dir, f"{prefix}_training_history.csv")
            self.data_saver.save_training_history(self.history, txt_path)
            self.data_saver.save_csv_format(self.history, csv_path)
    
    def print_summary(self, epoch_metrics: Dict, epoch: int):
        """
        エポック終了時の統計情報を表示
        
        Args:
            epoch_metrics (dict): エポックの評価指標
            epoch (int): エポック番号
        """
        print(f"Epoch {epoch+1} Summary:")
        print(f"  D_Loss: {epoch_metrics['D_loss']:.4f}")
        print(f"  G_Loss: {epoch_metrics['G_loss']:.4f}")
        print(f"  Cycle_Loss: {epoch_metrics['cycle_loss']:.4f}")
        print(f"  Identity_Loss: {epoch_metrics['identity_loss']:.4f}")
        print(f"  A_Real_Score: {epoch_metrics['A_real_score']:.3f}")
        print(f"  A_Fake_Score: {epoch_metrics['A_fake_score']:.3f}")
        print(f"  B_Real_Score: {epoch_metrics['B_real_score']:.3f}")
        print(f"  B_Fake_Score: {epoch_metrics['B_fake_score']:.3f}")
