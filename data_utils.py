import os
import cv2
from config import DATA_DIR, INPUT_NAME


def load_image():
    # Load RGB image from data directory
    path = os.path.join(DATA_DIR, INPUT_NAME)
    img = cv2.imread(path)
    if img is None:
        raise FileNotFoundError(f"Cannot load image: {path}")
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)