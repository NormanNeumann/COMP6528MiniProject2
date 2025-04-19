import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from config import DATA_DIR, SAVE_DIR, AE_EPOCHS, AE_LR, PATCH_SIZE, BATCH_SIZE, DEVICE
from data_utils import load_image

class PatchDataset(Dataset):
    def __init__(self, img, patch_size, num_patches=500):
        self.img = img
        self.patch_size = patch_size
        self.num_patches = num_patches
        self.transform = transforms.ToTensor()

    def __len__(self):
        return self.num_patches

    def __getitem__(self, idx):
        H, W, _ = self.img.shape
        y = np.random.randint(0, H - self.patch_size)
        x = np.random.randint(0, W - self.patch_size)
        patch = self.img[y:y+self.patch_size, x:x+self.patch_size]
        return self.transform(patch), self.transform(patch)

class SimpleAE(nn.Module):
    def __init__(self, in_ch=3, feat_ch=32):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(in_ch, feat_ch, 3, 2, 1), nn.ReLU(True),
            nn.Conv2d(feat_ch, feat_ch*2, 3, 2, 1), nn.ReLU(True),
            nn.Conv2d(feat_ch*2, feat_ch*4, 3, 2, 1), nn.ReLU(True)
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(feat_ch*4, feat_ch*2, 4, 2, 1), nn.ReLU(True),
            nn.ConvTranspose2d(feat_ch*2, feat_ch, 4, 2, 1), nn.ReLU(True),
            nn.ConvTranspose2d(feat_ch, in_ch, 4, 2, 1), nn.Sigmoid()
        )

    def forward(self, x):
        z = self.encoder(x)
        recon = self.decoder(z)
        return z, recon


def train_autoencoder():
    img = np.array(load_image())
    dataset = PatchDataset(img, PATCH_SIZE)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)
    device = torch.device(DEVICE)
    ae = SimpleAE().to(device)
    criterion = nn.L1Loss()
    optimizer = optim.Adam(ae.parameters(), lr=AE_LR)
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