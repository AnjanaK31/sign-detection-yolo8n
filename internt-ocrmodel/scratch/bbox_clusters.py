# -*- coding: utf-8 -*-
"""
bbox_clusters.py
-----------------
On the YOLO+island-cleaned image:
  1. Binarise (black pixels = foreground)
  2. Dilate to merge nearby strokes into one blob  → clusters
  3. For each cluster keep only those where
     (black pixels inside axis-aligned bbox) / (bbox area) >= density_thresh (10 %)
  4. Draw strict axis-aligned rectangles on a copy of the cleaned image
  5. Save the annotated output

Runs on both eval_page and fresh_page cleaned images.
"""
import sys, os
if hasattr(sys.stdout, "reconfigure"): sys.stdout.reconfigure(encoding="utf-8")

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

# ── config ─────────────────────────────────────────────────────────────────────
CLEANED_DIR   = "d:/Internship/OCR_PDF/internt-ocrmodel/scratch/eval_island_yolo"
OUT_DIR       = CLEANED_DIR          # same folder, new files
DENSITY_THRESH = 0.10                # 10 % black-pixel density inside bbox
# Dilation kernel – how far apart can pixels be and still be in the same cluster
DILATION_PX   = 8                    # px gap to bridge between characters
MIN_BBOX_AREA = 200                  # ignore tiny noise blobs (px²)
BOX_COLOR     = (0, 100, 220)        # blue boxes
BOX_THICK     = 2

os.makedirs(OUT_DIR, exist_ok=True)

# ── helpers ────────────────────────────────────────────────────────────────────

def label_banner(img_np, text, font_size=24, bg=(15,15,15), fg=(255,255,255)):
    pil  = Image.fromarray(img_np)
    draw = ImageDraw.Draw(pil)
    try:    font = ImageFont.truetype("C:/Windows/Fonts/consola.ttf", font_size)
    except: font = ImageFont.load_default()
    draw.rectangle([0, 0, pil.width, font_size+16], fill=bg)
    draw.text((8, 8), text, fill=fg, font=font)
    return np.array(pil)


def find_density_boxes(cleaned_pil,
                       dilation_px=DILATION_PX,
                       density_thresh=DENSITY_THRESH,
                       min_area=MIN_BBOX_AREA):
    """
    Return list of (x, y, w, h) axis-aligned boxes where density >= threshold.
    """
    gray      = np.array(cleaned_pil.convert("L"))
    # Binary: foreground = dark pixels (< 180)
    _, binary = cv2.threshold(gray, 180, 255, cv2.THRESH_BINARY_INV)

    # Dilate to merge nearby strokes → single blobs per text group
    kern   = cv2.getStructuringElement(cv2.MORPH_RECT,
                                       (dilation_px*2+1, dilation_px*2+1))
    dilated = cv2.dilate(binary, kern)

    num, labels, stats, _ = cv2.connectedComponentsWithStats(dilated)

    boxes = []
    img_h, img_w = binary.shape
    for i in range(1, num):
        x, y, w, h = (stats[i, cv2.CC_STAT_LEFT],
                      stats[i, cv2.CC_STAT_TOP],
                      stats[i, cv2.CC_STAT_WIDTH],
                      stats[i, cv2.CC_STAT_HEIGHT])
        area = w * h
        if area < min_area:
            continue

        # Skip full-image-width boxes (label banner artefact)
        if w >= img_w * 0.95:
            continue

        # Count actual black pixels in this bbox (use un-dilated binary)
        roi       = binary[y:y+h, x:x+w]
        black_cnt = int(np.sum(roi > 0))
        density   = black_cnt / area

        if density >= density_thresh:
            boxes.append((x, y, w, h, density))

    return boxes


def annotate(cleaned_pil, boxes, box_color=BOX_COLOR, thick=BOX_THICK):
    """Draw axis-aligned rectangles on a copy of the cleaned image."""
    out = np.array(cleaned_pil.convert("RGB"))
    for (x, y, w, h, dens) in boxes:
        cv2.rectangle(out, (x, y), (x+w, y+h), box_color, thick)
    return out


def process(cleaned_path, prefix, title):
    if not os.path.exists(cleaned_path):
        print(f"  SKIP – file not found: {cleaned_path}")
        return

    cleaned_pil = Image.open(cleaned_path)
    print(f"\n[{prefix}]  {cleaned_pil.size[0]}×{cleaned_pil.size[1]}  image")

    boxes = find_density_boxes(cleaned_pil)
    print(f"  Found {len(boxes)} boxes (density ≥ {DENSITY_THRESH*100:.0f}%)")
    for b in boxes:
        print(f"    x={b[0]:4d}  y={b[1]:4d}  w={b[2]:4d}  h={b[3]:4d}  "
              f"density={b[4]*100:.1f}%")

    out_np = annotate(cleaned_pil, boxes)
    out_np = label_banner(out_np,
        f"{title}  |  {len(boxes)} axis-aligned boxes (density ≥ {DENSITY_THRESH*100:.0f}%,  "
        f"dilation={DILATION_PX}px)")

    out_path = os.path.join(OUT_DIR, f"{prefix}_boxed.png")
    Image.fromarray(out_np).save(out_path)
    print(f"  Saved → {out_path}")
    return out_path


# ── run ────────────────────────────────────────────────────────────────────────
process(
    cleaned_path = os.path.join(CLEANED_DIR, "evalpage_2_cleaned.png"),
    prefix  = "evalpage",
    title   = "eval_page  YOLO+island cleaned"
)

process(
    cleaned_path = os.path.join(CLEANED_DIR, "freshpage_2_cleaned.png"),
    prefix  = "freshpage",
    title   = "fresh synthetic page  YOLO+island cleaned"
)

print("\nDone.")
