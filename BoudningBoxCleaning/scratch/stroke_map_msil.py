"""
Fast Full Stroke Map of msil.png
==================================
Strategy for speed:
  - Downsample to 30% for skeletonization (fast ~2-4s instead of 10+ min)
  - Use cv2.ximgproc.thinning if available, else iterative erosion thinning
  - Upsample skeleton result back to display scale for output

COLOR LEGEND (white background):
  BLACK  = text strokes
  GREEN  = engineering lines (leader / dimension / boundary)  
  BLUE   = bridged gap pixels (closed dashes/dots)
  ORANGE = intersections where lines physically run through text (PRESERVE these)
"""

import cv2
import numpy as np

IMG_PATH   = r"d:\Internship\OCR_PDF\INTRA_cleaning\msil.png"
OUT_STROKE = r"d:\Internship\OCR_PDF\INTRA_cleaning\msil_stroke_map.png"
OUT_SCALE  = 0.15   # final output display scale

# Processing scale for skeletonization (balance quality vs speed)
SKEL_SCALE      = 0.30
CLOSE_KERNEL_SZ = 5     # gap-bridging (at SKEL_SCALE resolution)
STROKE_PX       = 1     # re-render thickness (1px at SKEL_SCALE = ~3px at full res)
LINE_AR_THRESH  = 4.0
LINE_LEN_THRESH = 60    # at 30% scale
INTERS_DILATE   = 3

print("=" * 58)
print("  MSIL Fast Stroke Map")
print("=" * 58)

# ── 1. Load ───────────────────────────────────────────────────────────────────
print("[1/7] Loading image ...")
orig = cv2.imread(IMG_PATH)
if orig is None:
    raise FileNotFoundError(IMG_PATH)
H_full, W_full = orig.shape[:2]
print(f"      Full size: {W_full} x {H_full} px")

# ── 2. Downsample for processing ──────────────────────────────────────────────
print(f"[2/7] Downsampling to {int(SKEL_SCALE*100)}% for fast processing ...")
W_s = int(W_full * SKEL_SCALE)
H_s = int(H_full * SKEL_SCALE)
small = cv2.resize(orig, (W_s, H_s), interpolation=cv2.INTER_AREA)
gray  = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
print(f"      Processing size: {W_s} x {H_s} px")

# ── 3. Binary threshold ───────────────────────────────────────────────────────
print("[3/7] Binary threshold ...")
_, thresh = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY_INV)
print(f"      Foreground pixels: {int(np.sum(thresh>0)):,}")

# ── 4. Morphological closing to bridge gaps ───────────────────────────────────
print(f"[4/7] Closing gaps (kernel={CLOSE_KERNEL_SZ}px) ...")
close_k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (CLOSE_KERNEL_SZ, CLOSE_KERNEL_SZ))
closed  = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, close_k)
print(f"      After closing: {int(np.sum(closed>0)):,} px")

# ── 5. Fast thinning → 1-px skeleton ─────────────────────────────────────────
print("[5/7] Thinning to skeleton ...")

def fast_thin(binary_img):
    """Fast iterative thinning using cv2.ximgproc if available, else morphological."""
    try:
        import cv2.ximgproc as xip
        # ximgproc thinning expects uint8 binary (0/255)
        return xip.thinning(binary_img, thinningType=xip.THINNING_ZHANGSUEN)
    except (ImportError, AttributeError):
        # Fallback: iterative erosion-based thinning (slower but no extra deps)
        img = binary_img.copy()
        prev = np.zeros_like(img)
        kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))
        iteration = 0
        while True:
            eroded  = cv2.erode(img, kernel)
            opened  = cv2.dilate(eroded, kernel)
            temp    = cv2.subtract(img, opened)
            eroded2 = cv2.erode(img, kernel)
            img     = cv2.bitwise_or(eroded2, temp)
            diff    = cv2.absdiff(img, prev)
            prev    = img.copy()
            iteration += 1
            if cv2.countNonZero(diff) == 0 or iteration > 50:
                break
        return img

skel = fast_thin(closed)
print(f"      Skeleton pixels: {int(np.sum(skel>0)):,}")

# ── 6. CC classification ───────────────────────────────────────────────────────
print("[6/7] Classifying connected components ...")
num_cc, cc_labels, stats, _ = cv2.connectedComponentsWithStats(thresh, connectivity=8)

line_mask = np.zeros((H_s, W_s), dtype=np.uint8)
text_mask = np.zeros((H_s, W_s), dtype=np.uint8)
n_lines = n_text = 0

for lbl in range(1, num_cc):
    bw      = stats[lbl, cv2.CC_STAT_WIDTH]
    bh      = stats[lbl, cv2.CC_STAT_HEIGHT]
    long_s  = max(bw, bh)
    short_s = min(bw, bh)
    ar      = long_s / short_s if short_s > 0 else 0
    comp    = (cc_labels == lbl).astype(np.uint8) * 255

    if ar >= LINE_AR_THRESH and long_s >= LINE_LEN_THRESH:
        line_mask = cv2.bitwise_or(line_mask, comp)
        n_lines += 1
    else:
        text_mask = cv2.bitwise_or(text_mask, comp)
        n_text += 1

print(f"      Line CCs: {n_lines:,}  |  Text CCs: {n_text:,}")

# Intersection zone: text pixels that are adjacent to line pixels
ik           = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (INTERS_DILATE*2+1, INTERS_DILATE*2+1))
line_dilated = cv2.dilate(line_mask, ik)
inters_mask  = cv2.bitwise_and(line_dilated, text_mask)
print(f"      Intersection pixels (text touched by lines): {int(np.sum(inters_mask>0)):,}")

# ── 7. Compose stroke-map image ───────────────────────────────────────────────
print("[7/7] Composing stroke map ...")

# Split skeleton by classification
skel_line    = cv2.bitwise_and(skel, line_mask)
skel_text    = cv2.bitwise_and(skel, text_mask)
skel_bridged = cv2.bitwise_and(skel, cv2.bitwise_not(cv2.bitwise_or(line_mask, text_mask)))

# Re-dilate to strict STROKE_PX thickness
sk = cv2.getStructuringElement(cv2.MORPH_RECT, (STROKE_PX + 1, STROKE_PX + 1))
sl = cv2.dilate(skel_line,    sk)
st = cv2.dilate(skel_text,    sk)
sb = cv2.dilate(skel_bridged, sk)
si = cv2.dilate(inters_mask,  cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)))

# White background, paint layers (order matters: intersections on top)
vis = np.ones((H_s, W_s, 3), dtype=np.uint8) * 255
vis[st > 0] = [30,  30,  30]    # text  = near-black
vis[sl > 0] = [0,  180,   0]    # lines = bright green
vis[sb > 0] = [0,  120, 200]    # bridged = blue
vis[si > 0] = [220,  80,   0]   # intersections = orange

# Final scale for output
out_w = max(1, int(W_full * OUT_SCALE))
out_h = max(1, int(H_full * OUT_SCALE))
vis_out = cv2.resize(vis, (out_w, out_h), interpolation=cv2.INTER_NEAREST)
cv2.imwrite(OUT_STROKE, vis_out)

print(f"      Output: {out_w} x {out_h} px")
print(f"      Saved : {OUT_STROKE}")
print()
print("=" * 58)
print("  COLOR LEGEND")
print("  BLACK  = text strokes (compact CCs)")
print("  GREEN  = engineering lines (elongated CCs)")
print("  BLUE   = bridged gap pixels (closed dashes)")
print("  ORANGE = intersections -- text pixels touched by lines")
print("           (must be preserved when cleaning lines!)")
print("=" * 58)
print("DONE.")
