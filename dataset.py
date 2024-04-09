import torch
import torchvision
from torchvision import transforms
from torch.utils.data import DataLoader,random_split, Dataset
from torch import nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision.transforms as transforms
from torchvision.utils import make_grid
import matplotlib.pyplot as plt
import numpy as np
import torchvision.transforms.functional as TF
from PIL import Image
import os
from torchvision.io import read_image
from torchvision.transforms import Compose, Resize, ToTensor


class BlurSharpPairDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        """
        Args:
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        self.root_dir = root_dir
        self.transform = transform
        self.blur_dir = os.path.join(self.root_dir, 'blur')
        self.sharp_dir = os.path.join(self.root_dir, 'sharp')
        self.image_names = os.listdir(self.blur_dir)  # Assumes each blurred image has a corresponding sharp image

    def __len__(self):
        return len(self.image_names)

    def __getitem__(self, idx):
        blur_image_path = os.path.join(self.blur_dir, self.image_names[idx])
        sharp_image_path = os.path.join(self.sharp_dir, self.image_names[idx])
        
        blur_image = Image.open(blur_image_path).convert("RGB")
        sharp_image = Image.open(sharp_image_path).convert("RGB")
        
        if self.transform:
            blur_image = self.transform(blur_image)
            sharp_image = self.transform(sharp_image)
        
        return blur_image, sharp_image

# Example usage
transform = Compose([
    Resize((256, 448)),
    ToTensor(),
])

test_data = BlurSharpPairDataset(root_dir='data/test', transform=transform)

test_dataloader = DataLoader(test_data, batch_size=1, shuffle=True)