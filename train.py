"""
Generating patches from an image and training a simple AE (autoencoder) using self-supervised learning.
"""

import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from config import SAVE_DIR, AE_EPOCHS, AE_LR, PATCH_SIZE, BATCH_SIZE, DEVICE
from data_utils import load_image

class PatchDataset(Dataset):
    def __init__(self, img, patch_size, num_patches=500):
        self.img = img
        self.patch_size = patch_size
        self.num_patches = num_patches
        self.transform = transforms.ToTensor()

    def __len__(self):
        return self.num_patches

    """
    Returns a random patch of size [patch_size, patch_size, 3] from the image.
    The patch is transformed to a tensor and returned as both input and target.
    """
    def __getitem__(self, idx): # idx is not used, but required by Dataset
        # Randomly select a patch from the image
        # 在input图像中随机选取一个patch，并将其作为input和target准备递交给AE进行自监督训练
        H, W, _ = self.img.shape
        y = np.random.randint(0, H - self.patch_size)
        x = np.random.randint(0, W - self.patch_size)
        # Get the [patch_size, patch_size, 3] patch at (y, x)
        patch = self.img[y:y+self.patch_size, x:x+self.patch_size]
        return self.transform(patch), self.transform(patch) # (input, target)

class SimpleAE(nn.Module):
    def __init__(self, in_ch=3, feat_ch=32): # in_ch: input channels = 3 for RGB images; feat_ch: feature channels, the num of output channel for the first layer of the encoder.
        super().__init__()
        self.encoder = nn.Sequential(
            # kernel_size = 3, stride = 2, padding = 1
            # Select the most commonly used 3x3 convolution kernel.
            # Stride 2 means downsampling to 1/4 size each time.
            # Padding 1 keeps the output size consistent with downsampling.
            # 选取最常用的3x3卷积核，步长2即每次下采样至1/4大小，补偿1保持输出尺寸与下采样一致。

            # Chanel 3 -> 32 -> 64 -> 128
            nn.Conv2d(in_ch, feat_ch, 3, 2, 1),
            nn.ReLU(True),
            nn.Conv2d(feat_ch, feat_ch*2, 3, 2, 1),
            nn.ReLU(True),
            nn.Conv2d(feat_ch*2, feat_ch*4, 3, 2, 1),
            nn.ReLU(True)
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(feat_ch*4, feat_ch*2, 4, 2, 1),
            nn.ReLU(True),
            nn.ConvTranspose2d(feat_ch*2, feat_ch, 4, 2, 1),
            nn.ReLU(True),
            nn.ConvTranspose2d(feat_ch, in_ch, 4, 2, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        z = self.encoder(x) # z: latent representation of the input image
        recon = self.decoder(z) # recon: reconstructed image from the latent representation
        return z, recon


def train_autoencoder():
    img = np.array(load_image())
    dataset = PatchDataset(img, PATCH_SIZE)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)
    device = torch.device(DEVICE)
    ae = SimpleAE().to(device)
    criterion = nn.L1Loss() # This part can be modified to other loss according to the users' need.
    optimizer = optim.Adam(ae.parameters(), lr=AE_LR) # This part can be modified to other optimizers according to the users' need.

    for epoch in range(AE_EPOCHS):
        total = 0.0
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            _, recon = ae(x)
            loss = criterion(recon, y)
            loss.backward()
            optimizer.step()
            total += loss.item()
        if epoch % 10 == 0:
            print(f"AE Epoch {epoch}/{AE_EPOCHS}, Loss={total/len(loader):.4f}")
    os.makedirs(SAVE_DIR, exist_ok=True)
    torch.save(ae.encoder.state_dict(), os.path.join(SAVE_DIR, 'encoder.pth'))
    print("Autoencoder trained and encoder saved.")