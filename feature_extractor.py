import torch
import numpy as np
from config import SAVE_DIR, DEVICE
from train import SimpleAE

def extract_features(img):
    device = torch.device(DEVICE)
    encoder = SimpleAE().encoder.to(device)
    encoder.load_state_dict(torch.load(f"{SAVE_DIR}/encoder.pth", map_location=device))
    encoder.eval()
    x = torch.from_numpy(img.astype(np.float32)/255.0).permute(2,0,1).unsqueeze(0).to(device)
    with torch.no_grad():
        z = encoder(x)
    return z.squeeze(0).cpu().numpy()