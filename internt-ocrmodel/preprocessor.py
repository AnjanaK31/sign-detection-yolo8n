"""
preprocessor.py
---------------
All preprocessing stages for the blueprint symbol recognition pipeline.
"""

import cv2
import numpy as np
from PIL import Image

def to_grayscale(img_pil: Image.Image) -> np.ndarray:
    """Converts a PIL image to a uint8 grayscale numpy array."""
    return cv2.cvtColor(np.array(img_pil.convert("RGB")), cv2.COLOR_RGB2GRAY)

def apply_threshold(gray_np: np.ndarray, block_size: int = 11, c: int = 2) -> np.ndarray:
    """
    Adaptive Gaussian threshold -> pure binary image.
    Outputs: uint8 array, 0 = black (foreground), 255 = white (background).
    """
    mean_val = np.mean(gray_np)
    if mean_val < 127:
        gray_input = cv2.bitwise_not(gray_np)
    else:
        gray_input = gray_np

    # Prevent double thresholding artifacts on already near-binary/synthetic inputs
    binary_fraction = np.sum((gray_input < 40) | (gray_input > 215)) / gray_input.size
    if binary_fraction > 0.95:
        _, thresh = cv2.threshold(gray_input, 127, 255, cv2.THRESH_BINARY)
        return thresh

    return cv2.adaptiveThreshold(
        gray_input, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        block_size, c
    )

def remove_background_lines(thresh_np: np.ndarray, min_h_len: int = 80, min_v_len: int = 80, dilate_lines: int = 2) -> np.ndarray:
    """Removes long horizontal and vertical lines from page-scale binary images."""
    inv = cv2.bitwise_not(thresh_np)

    h_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (min_h_len, 1))
    h_lines  = cv2.morphologyEx(inv, cv2.MORPH_OPEN, h_kernel, iterations=1)

    v_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, min_v_len))
    v_lines  = cv2.morphologyEx(inv, cv2.MORPH_OPEN, v_kernel, iterations=1)

    all_lines = cv2.add(h_lines, v_lines)

    if dilate_lines > 0:
        d_kernel  = cv2.getStructuringElement(cv2.MORPH_RECT, (dilate_lines * 2 + 1, dilate_lines * 2 + 1))
        all_lines = cv2.dilate(all_lines, d_kernel, iterations=1)

    cleaned_inv = cv2.subtract(inv, all_lines)
    return cv2.bitwise_not(cleaned_inv)

def full_preprocess(img_pil: Image.Image, min_h_len: int = 80, min_v_len: int = 80) -> Image.Image:
    """Convenience wrapper: PIL -> cleaned PIL Image."""
    gray    = to_grayscale(img_pil)
    thresh  = apply_threshold(gray)
    cleaned = remove_background_lines(thresh, min_h_len, min_v_len)
    return Image.fromarray(cv2.cvtColor(cleaned, cv2.COLOR_GRAY2RGB))
