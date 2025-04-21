import cv2
import numpy as np
from sklearn.cluster import KMeans
from config import K, RANDOM_STATE

def cluster_pixels(fmap, output_size):
    C, h, w = fmap.shape # Channel, height, width
    flat = fmap.reshape(C, -1).T
    km = KMeans(n_clusters=K, random_state=RANDOM_STATE).fit(flat)
    labels = km.labels_.reshape(h, w).astype(np.uint8)
    return cv2.resize(labels, output_size, interpolation=cv2.INTER_NEAREST)