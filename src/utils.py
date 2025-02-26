import cv2
import torch
import numpy as np
import random
from src.config import CLASSES

def unsharp_mask(image, alpha=1.5, beta=-0.5):
    """Enhances edges by adding a weighted difference of the blurred image to the original."""
    blurred = cv2.GaussianBlur(image, (3, 3), 0)
    sharp = cv2.addWeighted(image, alpha, blurred, beta, 0)
    return sharp

def enhance_edges(image):
    """Applies Laplacian filter to enhance edges after blurring."""
    laplacian = cv2.Laplacian(image, cv2.CV_64F)
    enhanced = cv2.convertScaleAbs(image - laplacian)  # Subtract edges from the image
    return enhanced

def preprocess_image(image_path, margin=20):
    # Load the image in grayscale
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

    # Find the bounding box of the drawn object
    _, thresh = cv2.threshold(img, 1, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        x, y, w, h = cv2.boundingRect(contours[0])
        # Add margin to the bounding box
        x = max(0, x - margin)
        y = max(0, y - margin)
        w = min(img.shape[1] - x, w + 2 * margin)
        h = min(img.shape[0] - y, h + 2 * margin)
        img = img[y:y+h, x:x+w]

    # Resize the image to 28x28
    img = cv2.resize(img, (28, 28))

    # Apply Gaussian blur to soften sharp edges (adjust kernel size if needed)
    img = cv2.GaussianBlur(img, (3, 3), 0)

    # Apply Unsharp Masking for clarity
    img = unsharp_mask(img, alpha=1.5, beta=-0.5)

    img = enhance_edges(img)

    # Normalize pixel values to be between 0 and 1
    img = img.astype(np.float32)

    cv2.imwrite("drawing-resized.png", img)

    # Add batch and channel dimensions (1, 1, 28, 28) for PyTorch
    img = np.expand_dims(img, axis=(0, 1))

    # Convert to PyTorch tensor
    img_tensor = torch.tensor(img, dtype=torch.float32)

    return img_tensor

def get_random_class():
    return random.choice(CLASSES)