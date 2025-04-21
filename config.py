import os
import torch

# Project directories
BASE_DIR = os.path.dirname(__file__)
DATA_DIR = os.path.join(BASE_DIR, 'data')
INPUT_NAME = 'input.jpg'  # input image filename in data directory
OUTPUT_DIR = os.path.join(BASE_DIR, 'output')
SAVE_DIR = os.path.join(BASE_DIR, 'model')

# Segmentation parameters
K = 4  # number of clusters/classes


DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

# Autoencoder training parameters
AE_EPOCHS = 250
AE_LR = 1e-5
PATCH_SIZE = 64
BATCH_SIZE = 16

# Postprocess parameters
MORPH_KERNEL_SIZE = (7, 7)  # morphological operation kernel size

# Other settings
RANDOM_STATE = 0  # for reproducibility in clustering