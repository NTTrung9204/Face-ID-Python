import os
import torch
import torch.nn as nn
from torch.utils.data import Dataset, random_split, DataLoader
from PIL import Image

class RealFakeDataset(Dataset):
    def __init__(self, real_dir, fake_dir, transform=None):
        super(RealFakeDataset, self).__init__()
        self.transform = transform
        
        self.real_images = [
            os.path.join(real_dir, f) for f in os.listdir(real_dir)
            if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))
        ]
        self.fake_images = [
            os.path.join(fake_dir, f) for f in os.listdir(fake_dir)
            if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))
        ]
        
        self.images = self.real_images + self.fake_images
        
        self.labels = [0]*len(self.real_images) + [1]*len(self.fake_images)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_path = self.images[idx]
        label = self.labels[idx]
        
        image = Image.open(img_path).convert('RGB')
        
        if self.transform:
            image = self.transform(image)
        
        return image, label