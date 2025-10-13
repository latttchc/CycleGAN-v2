from PIL import Image
import os
from torch.utils.data import Dataset
import numpy as np

class CustomDataset(Dataset):
    def __init__(self, root_a, root_b, transform=None):
        self.root_a = root_a
        self.root_b = root_b
        self.transform = transform

        self.a_images = os.listdir(root_a)
        self.b_images = os.listdir(root_b)

        self.length_dataset = max(len(self.a_images), len(self.b_images))

        self.a_len = len(self.a_images)
        self.b_len = len(self.b_images)

    def __len__(self):
        return self.length_dataset
    
    def __getitem__(self, index):
        a_index = index % self.a_len
        b_index = index % self.b_len

        a_path = os.path.join(self.root_a, self.a_images[a_index])
        b_path = os.path.join(self.root_b, self.b_images[b_index])

        a_img = Image.open(a_path).convert("RGB")
        b_img = Image.open(b_path).convert("RGB")

        size = (256, 256)
        a_img = a_img.resize(size, Image.BICUBIC)
        b_img = b_img.resize(size, Image.BICUBIC)

        if self.transform:
            augmentations = self.transform(image=np.array(a_img), image0=np.array(b_img))
            a_img = augmentations["image"]
            b_img = augmentations["image0"]

        return b_img, a_img