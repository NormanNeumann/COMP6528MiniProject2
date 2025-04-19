import os
import cv2
import numpy as np
from config import OUTPUT_DIR, MORPH_KERNEL_SIZE


def save_segments(img, labels):
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    h, w = labels.shape
    for c in np.unique(labels):
        mask = (labels == c).astype(np.uint8) * 255
        kernel = np.ones(MORPH_KERNEL_SIZE, dtype=np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        out = cv2.bitwise_and(img, img, mask=mask)
        path = os.path.join(OUTPUT_DIR, f"segment_{c}.png")
        cv2.imwrite(path, cv2.cvtColor(out, cv2.COLOR_RGB2BGR))