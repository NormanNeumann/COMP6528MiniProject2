"""
数据读取和预处理模块
定义load_image函数，用于读取data路径下input的图片
并返回RGB格式的np.ndarray

Data loading and preprocessing module
Defamation of the load_image function to read the image from the input path under data
Then return the RGB format np.ndarray
"""

import os
import cv2
from config import DATA_DIR, INPUT_NAME

def load_image():
    # Load image from data directory
    path = os.path.join(DATA_DIR, INPUT_NAME)
    img = cv2.imread(path)
    # Check if the image was valid
    if img is None:
        raise FileNotFoundError(f"Cannot load image: {path}")
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB) # Convert to RGB format