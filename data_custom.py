import os
from PIL import Image

def augment_flip(input_dir, output_dir):
    """
    指定ディレクトリ内の画像を上下・左右・両方反転して保存（データ拡張）
    """
    os.makedirs(output_dir, exist_ok=True)
    for fname in os.listdir(input_dir):
        if not fname.lower().endswith(('.png', '.jpg', '.jpeg')):
            continue
        img_path = os.path.join(input_dir, fname)
        img = Image.open(img_path).convert("RGB")
        base, ext = os.path.splitext(fname)

        # 元画像
        img.save(os.path.join(output_dir, f"{base}_orig{ext}"))

        # 上下反転
        img_v = img.transpose(Image.FLIP_TOP_BOTTOM)
        img_v.save(os.path.join(output_dir, f"{base}_vflip{ext}"))

        # 左右反転
        img_h = img.transpose(Image.FLIP_LEFT_RIGHT)
        img_h.save(os.path.join(output_dir, f"{base}_hflip{ext}"))

        # 上下左右反転
        img_vh = img_v.transpose(Image.FLIP_LEFT_RIGHT)
        img_vh.save(os.path.join(output_dir, f"{base}_vhflip{ext}"))

if __name__ == "__main__":
    # フォルダ指定で実行
    input_dir = input("入力フォルダのパスを入力してください: ")
    output_dir = input("出力フォルダのパスを入力してください: ")
    augment_flip(input_dir, output_dir)
    print("拡張画像の保存が完了しました。")