"""
Line Detection on msil.png
===========================
Uses:
  1. Probabilistic Hough Line Transform  (thin/straight dimension lines)
  2. Large Connected-Component filtering  (broad strokes / leader lines)

Outputs:
  d:\Internship\OCR_PDF\INTRA_cleaning\msil_lines_detected.png
  - Green  = Hough detected lines
  - Red    = Large-CC detected line segments (MSER-line blobs > threshold)
  - Background = original color image (downscaled for display)

Also prints statistics: line counts, pixel coverage, etc.
"""

import cv2
import numpy as np
import math
import os

IMG_PATH   = r"d:\Internship\OCR_PDF\INTRA_cleaning\msil.png"
OUT_PATH   = r"d:\Internship\OCR_PDF\INTRA_cleaning\msil_lines_detected.png"
OUT_SCALE  = 0.15   # msil is huge (~8000+ px wide); scale for output image

print("Loading image …")
orig = cv2.imread(IMG_PATH)
if orig is None:
    raise FileNotFoundError(f"Cannot open {IMG_PATH}")

H, W = orig.shape[:2]
print(f"  Image size: {W} x {H} px")

gray = cv2.cvtColor(orig, cv2.COLOR_BGR2GRAY)

# ── 1. Binary threshold (dark lines on white paper) ─────────────────────────
_, thresh = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY_INV)

# ── 2. Probabilistic Hough Transform ─────────────────────────────────────────
# minLineLength: must span at least 120 px to count as a "dimension / leader" line
# maxLineGap:    tolerate small gaps (dotted lines / dash patterns)
print("Running Hough Line Transform …")
edges = cv2.Canny(gray, 50, 150, apertureSize=3)
hough_lines = cv2.HoughLinesP(
    edges,
    rho=1,
    theta=math.pi / 180,
    threshold=80,
    minLineLength=120,
    maxLineGap=20
)
n_hough = len(hough_lines) if hough_lines is not None else 0
print(f"  Hough lines found: {n_hough}")

# ── 3. Large Connected-Component (CC) line detection ─────────────────────────
# CAD lines show up as very elongated connected components in the thresholded image.
# Criteria:  aspect_ratio > 5  AND  long_side > 150 px
print("Running Connected-Component line detection …")
num_labels, cc_labels, stats, _ = cv2.connectedComponentsWithStats(thresh, connectivity=8)

cc_line_segs = []   # list of (x1,y1,x2,y2) centre-point lines per CC bounding box
for lbl in range(1, num_labels):
    x  = stats[lbl, cv2.CC_STAT_LEFT]
    y  = stats[lbl, cv2.CC_STAT_TOP]
    bw = stats[lbl, cv2.CC_STAT_WIDTH]
    bh = stats[lbl, cv2.CC_STAT_HEIGHT]
    long_s  = max(bw, bh)
    short_s = min(bw, bh)
    ar = long_s / short_s if short_s > 0 else 0
    if long_s >= 150 and ar >= 5:
        # represent as a horizontal or vertical centre-line segment
        cx = x + bw // 2
        cy = y + bh // 2
        if bw >= bh:
            cc_line_segs.append((x, cy, x + bw, cy))
        else:
            cc_line_segs.append((cx, y, cx, y + bh))

print(f"  CC line segments found: {len(cc_line_segs)}")

# ── 4. Compose visualisation (at reduced scale) ───────────────────────────────
print("Composing visualisation …")
vis = orig.copy()

# Draw CC lines in RED (thick, semi-transparent)
overlay = vis.copy()
for (x1, y1, x2, y2) in cc_line_segs:
    cv2.line(overlay, (x1, y1), (x2, y2), (0, 0, 220), 6)
cv2.addWeighted(overlay, 0.45, vis, 0.55, 0, vis)

# Draw Hough lines in GREEN (bright, thinner)
if hough_lines is not None:
    for ln in hough_lines:
        x1, y1, x2, y2 = ln[0]
        cv2.line(vis, (x1, y1), (x2, y2), (0, 220, 0), 3)

# Legend
legend_y = 60
for (color, label) in [((0, 220, 0), "Hough Lines"), ((0, 0, 220), "CC Line Blobs")]:
    cv2.rectangle(vis, (30, legend_y - 22), (72, legend_y + 5), color, -1)
    cv2.putText(vis, label, (82, legend_y), cv2.FONT_HERSHEY_SIMPLEX, 2.5, color, 4, cv2.LINE_AA)
    legend_y += 80

# Scale down for output
out_w = int(W * OUT_SCALE)
out_h = int(H * OUT_SCALE)
vis_small = cv2.resize(vis, (out_w, out_h), interpolation=cv2.INTER_AREA)
cv2.imwrite(OUT_PATH, vis_small)
print(f"\n✓ Saved: {OUT_PATH}  ({out_w} x {out_h} px)")

# ── 5. Stats summary ──────────────────────────────────────────────────────────
hough_px = 0
if hough_lines is not None:
    for ln in hough_lines:
        x1, y1, x2, y2 = ln[0]
        hough_px += int(math.hypot(x2-x1, y2-y1))

total_fg = int(np.sum(thresh > 0))
print(f"\n── Summary ─────────────────────────────────────────")
print(f"  Total foreground pixels   : {total_fg:,}")
print(f"  Hough lines detected      : {n_hough}")
print(f"  Hough total pixel length  : {hough_px:,} px")
print(f"  CC line blobs detected    : {len(cc_line_segs)}")
print(f"────────────────────────────────────────────────────")
