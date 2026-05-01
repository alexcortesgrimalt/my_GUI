import numpy as np
import cv2

def preprocess_sem(sem_img):
    """Denoise + normalize SEM image."""
    img = sem_img.copy()

    # Normalize
    img = (img - np.min(img)) / (np.max(img) - np.min(img))

    # Convert to uint8 for OpenCV
    img = (img * 255).astype(np.uint8)

    # Gaussian blur
    img = cv2.GaussianBlur(img, (5, 5), 0)

    return img


def detect_faraday_region(sem_img):
    """Return binary mask of Faraday cup."""
    img = preprocess_sem(sem_img)

    # Otsu threshold
    _, mask = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Morphological cleanup
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

    return mask.astype(bool)

def refine_with_ebic(mask, ebic_img):
    """Keep only regions with significant EBIC signal."""
    threshold = np.mean(ebic_img)

    refined = mask & (ebic_img > threshold)
    return refined


def circular_core_mask(mask, fill_fraction=0.9):
    """
    Build a circular mask centered at COM of input mask.
    Circle area = fill_fraction * mask area.
    """

    mask = mask.astype(bool)

    coords = np.argwhere(mask)

    if len(coords) == 0:
        return mask

    # --- center of mass ---
    cy, cx = coords.mean(axis=0)

    # --- target radius from area ---
    area_mask = np.sum(mask)
    radius = np.sqrt((fill_fraction * area_mask) / np.pi)

    # --- build circle ---
    h, w = mask.shape
    yy, xx = np.ogrid[:h, :w]

    circle = (xx - cx)**2 + (yy - cy)**2 <= radius**2

    return circle & mask