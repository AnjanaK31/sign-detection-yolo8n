"""
Crop + Clean msil.png labels
==============================
For each bounding box in Label.txt (msil.png entry):
  - Extract the oriented crop from the ORIGINAL msil.png (with 30% buffer)
  - Run CC line detection on the crop: elongated CCs -> erase (white)
  - Compact CCs (text) -> keep original color pixels
  - Save cleaned crop PNG to output folder

Output: d:\Internship\OCR_PDF\INTRA_cleaning\msil_cleaned_crops\
"""

import sys, os, json, math
import cv2
import numpy as np
from PIL import Image
Image.MAX_IMAGE_PIXELS = None   # disable decompression bomb check for large CAD images

sys.path.append(r"d:\Internship\OCR_PDF\BoudningBoxCleaning")
from clean_page_expressions import rectify_crop

IMG_PATH   = r"d:\Internship\OCR_PDF\INTRA_cleaning\msil.png"
LABEL_PATH = r"d:\Internship\OCR_PDF\INTRA_cleaning\Label.txt"
OUT_DIR    = r"d:\Internship\OCR_PDF\INTRA_cleaning\msil_cleaned_crops"

# Line detection parameters (at full resolution)
LINE_AR_THRESH  = 4.0    # aspect ratio: elongated = line
LINE_LEN_THRESH = 30     # min long-side px in a CROP to be a line
INTERS_DILATE   = 4      # px around line to protect adjacent text
BUFFER          = 0.30   # crop buffer around bounding box

os.makedirs(OUT_DIR, exist_ok=True)

print("=" * 60)
print("  msil.png Crop Extractor + Line Cleaner")
print("=" * 60)

# ── 1. Load full-res image ────────────────────────────────────────────────────
print("[1/4] Loading msil.png ...")
orig_bgr = cv2.imread(IMG_PATH)     # load with OpenCV (no size limit)
if orig_bgr is None:
    raise FileNotFoundError(IMG_PATH)
orig_np  = cv2.cvtColor(orig_bgr, cv2.COLOR_BGR2RGB)   # H x W x 3, RGB
orig_pil = Image.fromarray(orig_np)                    # for rectify_crop
H, W     = orig_np.shape[:2]
print(f"      {W} x {H} px")

# ── 2. Parse Label.txt for msil.png boxes ─────────────────────────────────────
print("[2/4] Parsing Label.txt for msil.png ...")
msil_boxes = None
with open(LABEL_PATH, "r", encoding="utf-8") as f:
    for line in f:
        parts = line.strip().split("\t")
        if len(parts) >= 2 and "msil.png" in parts[0]:
            msil_boxes = json.loads(parts[1])
            break

if msil_boxes is None:
    print("ERROR: msil.png not found in Label.txt")
    sys.exit(1)

print(f"      Found {len(msil_boxes)} bounding boxes")

# ── 3. Per-crop line detection and cleaning ───────────────────────────────────
print("[3/4] Processing crops ...")
saved = 0

for idx, box in enumerate(msil_boxes):
    transcription = box.get("transcription", f"box_{idx}")
    points = box.get("points", [])
    if len(points) != 4:
        continue

    pts_np = np.array(points, dtype=np.float32)
    rect   = cv2.minAreaRect(pts_np)
    cx, cy = rect[0]
    w_box, h_box = rect[1]
    angle  = rect[2]
    bbox_metrics = {"cx": cx, "cy": cy, "w": w_box, "h": h_box, "angle": angle}

    # Extract color crop using rectify_crop
    crop_color = rectify_crop(orig_pil, bbox_metrics=bbox_metrics, buffer_percent=BUFFER)
    crop_np    = np.array(crop_color)   # H x W x 3, RGB

    if crop_np.size == 0:
        continue

    ch, cw = crop_np.shape[:2]

    # ── Within-crop line detection ────────────────────────────────────────────
    # Convert to gray and threshold
    crop_gray = cv2.cvtColor(crop_np, cv2.COLOR_RGB2GRAY)
    _, thresh  = cv2.threshold(crop_gray, 200, 255, cv2.THRESH_BINARY_INV)

    # CC analysis
    num_cc, cc_labels, stats, _ = cv2.connectedComponentsWithStats(thresh, connectivity=8)

    line_mask = np.zeros((ch, cw), dtype=np.uint8)
    text_mask = np.zeros((ch, cw), dtype=np.uint8)

    for lbl in range(1, num_cc):
        bw      = stats[lbl, cv2.CC_STAT_WIDTH]
        bh      = stats[lbl, cv2.CC_STAT_HEIGHT]
        long_s  = max(bw, bh)
        short_s = min(bw, bh)
        ar      = long_s / short_s if short_s > 0 else 0
        comp    = (cc_labels == lbl).astype(np.uint8) * 255

        area = stats[lbl, cv2.CC_STAT_AREA]

        if ar >= LINE_AR_THRESH and long_s >= LINE_LEN_THRESH:
            line_mask = cv2.bitwise_or(line_mask, comp)
        else:
            text_mask = cv2.bitwise_or(text_mask, comp)

    # Build protection: text + intersection zone
    ik           = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,
                       (INTERS_DILATE * 2 + 1, INTERS_DILATE * 2 + 1))
    line_dilated = cv2.dilate(line_mask, ik)
    inters_mask  = cv2.bitwise_and(line_dilated, text_mask)
    protect_mask = cv2.bitwise_or(text_mask, inters_mask)

    # Final erase: lines minus protected text
    erase_mask = cv2.bitwise_and(line_mask, cv2.bitwise_not(protect_mask))

    # ── Apply: replace erased pixels with white ───────────────────────────────
    cleaned = crop_np.copy()
    cleaned[erase_mask > 0] = [255, 255, 255]

    # ── Save ──────────────────────────────────────────────────────────────────
    clean_name = (transcription
                  .replace("/", "_").replace(":", "_")
                  .replace("°", "deg").replace("×", "x")
                  .replace("±", "+-").replace(" ", "_")
                  .replace("(", "").replace(")", ""))
    fname  = f"crop_{idx:03d}_{clean_name}.png"
    fpath  = os.path.join(OUT_DIR, fname)

    # Save as BGR for OpenCV
    cv2.imwrite(fpath, cv2.cvtColor(cleaned, cv2.COLOR_RGB2BGR))
    erased_px = int(np.sum(erase_mask > 0))
    print(f"  [{idx:3d}] {transcription:<18s}  {cw:4d}x{ch:4d}  erased={erased_px:5d}px  -> {fname}")
    saved += 1

# ── 4. Summary ────────────────────────────────────────────────────────────────
print()
print(f"[4/4] Done. Saved {saved} cleaned crops to:")
print(f"      {OUT_DIR}")
print("=" * 60)
