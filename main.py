from train_autoencoder import train_autoencoder
from data_utils import load_image
from feature_extractor import extract_features
from clustering import cluster_pixels
from postprocess import save_segments

def main():
    train_autoencoder()
    img = load_image()
    fmap = extract_features(img)
    h, w = img.shape[:2]
    labels = cluster_pixels(fmap, (w, h))
    save_segments(img, labels)
    print("Segmentation complete. Results saved.")

if __name__ == '__main__':
    main()