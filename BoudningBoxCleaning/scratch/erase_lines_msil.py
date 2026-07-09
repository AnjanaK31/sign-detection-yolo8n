"""
Clean msil.png — Erase All Detected Lines  (100% full resolution)
==================================================================
Uses connected-component classification to:
  1. Find all "line" CCs (elongated: AR >= 4, long_side >= 200px)
  2. Build a text-protection mask (compact CCs + intersection zone)
  3. Erase line pixels → white in the output color image
  4. Save msil_cleaned.png

All processing at FULL RESOLUTION (21259 x 9932 px).
"""

import cv2
import numpy as np

IMG_PATH      = r"d:\Internship\OCR_PDF\INTRA_cleaning\msil.png"
OUT_CLEANED   = r"d:\Internship\OCR_PDF\INTRA_cleaning\msil_cleaned.png"
OUT_ERASED_VIS= r"d:\Internship\OCR_PDF\INTRA_cleaning\msil_erase_mask_preview.png"

# Parameters
WORK_SCALE       = 1.0      # 100% full resolution
LINE_AR_THRESH   = 4.0      # aspect ratio to call a CC a "line"
LINE_LEN_THRESH  = 200      # min long-side pixels at full res to be a "line"
INTERS_DILATE    = 10       # pixels around line to protect text at intersections
THRESH_VAL       = 200      # binary threshold value (dark ink = foreground)

print("=" * 60)
print("  msil.png Line Eraser")
print("=" * 60)

# ── 1. Load full-res image ────────────────────────────────────────────────────
print("[1/6] Loading full-res image ...")
orig_full = cv2.imread(IMG_PATH)
if orig_full is None:
    raise FileNotFoundError(IMG_PATH)
H_full, W_full = orig_full.shape[:2]
print(f"      {W_full} x {H_full} px")

# ── 2. Work at 100% — no downsampling ───────────────────────────────────────
print("[2/6] Working at 100% full resolution (no downsampling) ...")
W_w = W_full
H_w = H_full
small = orig_full   # alias — same array
gray  = cv2.cvtColor(orig_full, cv2.COLOR_BGR2GRAY)

# ── 3. Binary threshold ───────────────────────────────────────────────────────
print("[3/6] Binary threshold & CC classification ...")
_, thresh = cv2.threshold(gray, THRESH_VAL, 255, cv2.THRESH_BINARY_INV)

num_cc, cc_labels, stats, _ = cv2.connectedComponentsWithStats(thresh, connectivity=8)
print(f"      Total CCs: {num_cc - 1:,}")

line_mask_w = np.zeros((H_w, W_w), dtype=np.uint8)
text_mask_w = np.zeros((H_w, W_w), dtype=np.uint8)
n_lines = n_text = 0

for lbl in range(1, num_cc):
    bw      = stats[lbl, cv2.CC_STAT_WIDTH]
    bh      = stats[lbl, cv2.CC_STAT_HEIGHT]
    long_s  = max(bw, bh)
    short_s = min(bw, bh)
    ar      = long_s / short_s if short_s > 0 else 0
    comp    = (cc_labels == lbl).astype(np.uint8) * 255

    if ar >= LINE_AR_THRESH and long_s >= LINE_LEN_THRESH:
        line_mask_w = cv2.bitwise_or(line_mask_w, comp)
        n_lines += 1
    else:
        text_mask_w = cv2.bitwise_or(text_mask_w, comp)
        n_text += 1

print(f"      Line CCs: {n_lines:,}  |  Text CCs: {n_text:,}")

# ── 4. Build protection mask (text + intersection zone) ───────────────────────
print("[4/6] Building text-protection mask ...")
ik           = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,
                   (INTERS_DILATE * 2 + 1, INTERS_DILATE * 2 + 1))
line_dilated = cv2.dilate(line_mask_w, ik)
inters_mask  = cv2.bitwise_and(line_dilated, text_mask_w)   # text touched by lines
protect_mask_w = cv2.bitwise_or(text_mask_w, inters_mask)   # protect all text + junctions

# Final erase mask at work scale: line pixels minus protected text
erase_mask_w = cv2.bitwise_and(line_mask_w, cv2.bitwise_not(protect_mask_w))
erased_px_w  = int(np.sum(erase_mask_w > 0))
print(f"      Protected intersection pixels : {int(np.sum(inters_mask > 0)):,}")
print(f"      Pixels to erase (work scale)  : {erased_px_w:,}")

# ── 5. Erase mask is already at full resolution ───────────────────────────────
print("[5/6] Erase mask ready at full resolution ...")
erase_mask_full = erase_mask_w   # already full-res
erased_full = erased_px_w
print(f"      Pixels to erase (full res)    : {erased_full:,}")

# ── 6. Apply erase to original color image ────────────────────────────────────
print("[6/6] Applying erase → white pixels ...")
cleaned = orig_full.copy()
cleaned[erase_mask_full > 0] = [255, 255, 255]   # paint erased pixels white

cv2.imwrite(OUT_CLEANED, cleaned)
print(f"      Saved cleaned image: {OUT_CLEANED}")

# Save a small preview of the erase mask (green=erased, orange=protected)
preview_scale = 0.10
Wp = max(1, int(W_full * preview_scale))
Hp = max(1, int(H_full * preview_scale))
orig_small   = cv2.resize(orig_full,        (Wp, Hp), interpolation=cv2.INTER_AREA)
erase_small  = cv2.resize(erase_mask_full,  (Wp, Hp), interpolation=cv2.INTER_NEAREST)
protect_small= cv2.resize(protect_mask_w,   (Wp, Hp), interpolation=cv2.INTER_NEAREST)
protect_small_full = cv2.resize(protect_small, (Wp, Hp), interpolation=cv2.INTER_NEAREST)

vis = orig_small.copy()
overlay = vis.copy()
overlay[erase_small   > 0] = [0, 200, 0]      # green = erased
overlay[protect_small > 0] = [0, 100, 220]    # blue = protected text
cv2.addWeighted(overlay, 0.5, vis, 0.5, 0, vis)
cv2.imwrite(OUT_ERASED_VIS, vis)
print(f"      Saved erase-mask preview: {OUT_ERASED_VIS}")

total_fg = int(np.sum(cv2.threshold(
    cv2.cvtColor(cv2.resize(orig_full, (W_w, H_w), interpolation=cv2.INTER_AREA),
                 cv2.COLOR_BGR2GRAY), THRESH_VAL, 255, cv2.THRESH_BINARY_INV)[1] > 0))

print()
print("=" * 60)
print("  SUMMARY")
print(f"  Original foreground pixels : {total_fg:,}")
print(f"  Line pixels erased         : {erased_full:,}")
print(f"  Protected (text) pixels    : {int(np.sum(protect_mask_w > 0)):,}")
pct = erased_full / (total_fg * (1/WORK_SCALE)**2) * 100 if total_fg > 0 else 0
print(f"  Erase coverage             : ~{pct:.1f}% of foreground")
print("=" * 60)
print("DONE.")
