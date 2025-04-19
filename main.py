# main.py
import os
import sys
# Ensure current script directory is in sys.path for local imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from data_utils import load_image
from feature_extractor import extract_features
from clustering import cluster_pixels
from postprocess import save_segments
from config import OUTPUT_DIR


def main():
    # Load image
    img = load_image()
    # Extract features
    fmap = extract_features(img)
    # Cluster pixels and upsample to original size
    h, w = img.shape[:2]
    labels = cluster_pixels(fmap, (w, h))
    # Save segmented images
    save_segments(img, labels)
    print(f"Segments saved to {OUTPUT_DIR}")

if __name__ == '__main__':
    main()
