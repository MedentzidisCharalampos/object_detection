import cv2
import numpy as np

def apply_clahe(img: np.ndarray) -> np.ndarray:
    """
    Apply CLAHE (Contrast Limited Adaptive Histogram Equalization) to a BGR image.

    Args:
        img (np.ndarray): Input BGR image.

    Returns:
        np.ndarray: CLAHE-enhanced image.
    """
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)

    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    cl = clahe.apply(l)

    merged = cv2.merge((cl, a, b))
    final = cv2.cvtColor(merged, cv2.COLOR_LAB2BGR)
    return final

def adjust_gamma(image: np.ndarray, gamma: float = 1.0) -> np.ndarray:
    """
    Apply gamma correction to a BGR image.

    Args:
        image (np.ndarray): Input image.
        gamma (float): Gamma value.

    Returns:
        np.ndarray: Gamma-adjusted image.
    """
    inv_gamma = 1.0 / gamma
    table = np.array([(i / 255.0) ** inv_gamma * 255
                     for i in np.arange(256)]).astype("uint8")
    return cv2.LUT(image, table)

def resize_image(image: np.ndarray, size: tuple = (640, 640)) -> np.ndarray:
    """
    Resize image to specified size.

    Args:
        image (np.ndarray): Input image.
        size (tuple): Target size (width, height).

    Returns:
        np.ndarray: Resized image.
    """
    return cv2.resize(image, size, interpolation=cv2.INTER_AREA)

def normalize_image(image: np.ndarray) -> np.ndarray:
    """
    Normalize image pixel values to [0, 1] range.

    Args:
        image (np.ndarray): Input image.

    Returns:
        np.ndarray: Normalized image.
    """
    return image.astype(np.float32) / 255.0
