import cv2
import numpy as np
from PIL import Image
import imagehash

def compute_perceptual_hash(image_path):
    """Compute perceptual hash (phash) for an image file."""
    try:
        img = Image.open(image_path).convert('RGB')
        phash = imagehash.phash(img)
        return str(phash)
    except Exception as e:
        print(f"Error computing perceptual hash for {image_path}: {e}")
        return None
