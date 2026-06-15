"""
preprocessor.py
---------------
All preprocessing stages for the blueprint symbol recognition pipeline.

Pipeline stages (in order):
  1. to_grayscale()           — RGB PIL → uint8 grayscale numpy array
  2. apply_threshold()        — grayscale → binary (black lines on white bg)
  3. remove_background_lines()— binary → cleaned binary  (line removal)
  4. full_preprocess()        — convenience wrapper: PIL → cleaned PIL Image

The intermediate numpy arrays are intentionally exposed so the visualizer
can show a before/after panel for each stage.
"""

import cv2
import numpy as np
from PIL import Image


# ─────────────────────────────────────────────────────────────────────────────
# Stage 1 — Grayscale
# ─────────────────────────────────────────────────────────────────────────────

def to_grayscale(img_pil: Image.Image) -> np.ndarray:
    """Converts a PIL image to a uint8 grayscale numpy array."""
    return cv2.cvtColor(np.array(img_pil.convert("RGB")), cv2.COLOR_RGB2GRAY)


# ─────────────────────────────────────────────────────────────────────────────
# Stage 2 — Adaptive Threshold
# ─────────────────────────────────────────────────────────────────────────────

def apply_threshold(gray_np: np.ndarray,
                    block_size: int = 11,
                    c: int = 2) -> np.ndarray:
    """
    Adaptive Gaussian threshold → pure binary image.
    Output: uint8 array, 0 = black (foreground), 255 = white (background).
    """
    return cv2.adaptiveThreshold(
        gray_np, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        block_size, c
    )


# ─────────────────────────────────────────────────────────────────────────────
# Stage 3 — Background Line Removal
# ─────────────────────────────────────────────────────────────────────────────

def remove_background_lines(thresh_np: np.ndarray,
                             min_h_len: int = 80,
                             min_v_len: int = 80,
                             dilate_lines: int = 2) -> np.ndarray:
    """
    Removes long horizontal and vertical ruled lines from a binary image.

    Strategy:
      1. Invert image so foreground (lines + symbols) = white (255).
      2. Use a long horizontal structuring element to detect only horizontal
         strokes longer than min_h_len pixels via morphological OPEN.
         (short symbol strokes can't survive a 80px erosion — they vanish.)
      3. Same with a vertical element for vertical lines.
      4. Dilate the detected line masks slightly to catch any broken ends.
      5. Subtract both line masks from the inverted image.
      6. Re-invert → white background, black symbols only.

    Args:
        thresh_np   : uint8 binary image from apply_threshold().
        min_h_len   : minimum horizontal run length to classify as a line.
                      Set to ~2-5% of image width. Default 80 px.
        min_v_len   : minimum vertical run length to classify as a line.
                      Set to ~2-5% of image height. Default 80 px.
        dilate_lines: how many pixels to expand detected line masks before
                      subtraction (fills small gaps in detected lines).

    Returns:
        uint8 binary image with long H/V lines removed.
    """
    # ── Invert: foreground (strokes) = white ─────────────────────────────────
    inv = cv2.bitwise_not(thresh_np)

    # ── Detect horizontal lines ───────────────────────────────────────────────
    h_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (min_h_len, 1))
    h_lines  = cv2.morphologyEx(inv, cv2.MORPH_OPEN, h_kernel, iterations=1)

    # ── Detect vertical lines ─────────────────────────────────────────────────
    v_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, min_v_len))
    v_lines  = cv2.morphologyEx(inv, cv2.MORPH_OPEN, v_kernel, iterations=1)

    # ── Merge line masks ──────────────────────────────────────────────────────
    all_lines = cv2.add(h_lines, v_lines)

    # ── Dilate the line mask slightly to catch broken/partial pixels ──────────
    if dilate_lines > 0:
        d_kernel  = cv2.getStructuringElement(
            cv2.MORPH_RECT, (dilate_lines * 2 + 1, dilate_lines * 2 + 1)
        )
        all_lines = cv2.dilate(all_lines, d_kernel, iterations=1)

    # ── Subtract lines from foreground ────────────────────────────────────────
    cleaned_inv = cv2.subtract(inv, all_lines)

    # ── Re-invert → white bg, black symbols ──────────────────────────────────
    return cv2.bitwise_not(cleaned_inv)


# ─────────────────────────────────────────────────────────────────────────────
# Convenience wrapper — used by pipeline.py
# ─────────────────────────────────────────────────────────────────────────────

def full_preprocess(img_pil: Image.Image,
                    min_h_len: int = 80,
                    min_v_len: int = 80) -> Image.Image:
    """
    Full preprocessing chain: PIL RGB → cleaned binary PIL Image.
    Equivalent to the old preprocess_image() but now includes line removal.

    Returns a 3-channel RGB PIL Image (for compatibility with downstream code).
    """
    gray    = to_grayscale(img_pil)
    thresh  = apply_threshold(gray)
    cleaned = remove_background_lines(thresh, min_h_len, min_v_len)
    return Image.fromarray(cv2.cvtColor(cleaned, cv2.COLOR_GRAY2RGB))


# ─────────────────────────────────────────────────────────────────────────────
# Helper — returns ALL intermediate stages as numpy arrays
# (used by visualize_pipeline.py to build Panel A)
# ─────────────────────────────────────────────────────────────────────────────

def preprocess_stages(img_pil: Image.Image,
                      min_h_len: int = 80,
                      min_v_len: int = 80):
    """
    Returns all four preprocessing stages as uint8 numpy arrays:
        raw_np     — RGB, straight from PIL
        gray_np    — grayscale
        thresh_np  — after adaptive threshold
        cleaned_np — after line removal
    """
    raw_np    = np.array(img_pil.convert("RGB"))
    gray_np   = to_grayscale(img_pil)
    thresh_np = apply_threshold(gray_np)
    cleaned_np = remove_background_lines(thresh_np, min_h_len, min_v_len)
    return raw_np, gray_np, thresh_np, cleaned_np
