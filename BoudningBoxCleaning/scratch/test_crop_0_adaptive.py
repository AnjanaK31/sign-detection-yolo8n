import sys
import os
import cv2
import numpy as np
from PIL import Image

sys.path.append(os.path.abspath(r"d:\Internship\OCR_PDF\BoudningBoxCleaning"))
from clean_page_expressions import clean_crop_lines

img_path = r"d:\Internship\OCR_PDF\sign-detection-yolo8n\Nived-dataset\dataset\dataset\crop_img_backup\intra_crop_0.jpg"
img_bgr = cv2.imread(img_path)
if img_bgr is None:
    print("Failed to load image")
    sys.exit(1)
    
h, w = img_bgr.shape[:2]
img_gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

# Pad to get crop_big
pad_val = 30
img_big_bgr = cv2.copyMakeBorder(img_bgr, pad_val, pad_val, pad_val, pad_val, cv2.BORDER_CONSTANT, value=[255, 255, 255])
crop_pil_std = Image.fromarray(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB))
crop_pil_big = Image.fromarray(cv2.cvtColor(img_big_bgr, cv2.COLOR_BGR2RGB))

# Threshold
_, thresh = cv2.threshold(img_gray, 180, 255, cv2.THRESH_BINARY)

# MSER
mser = cv2.MSER_create(5, 5, 15000)
regions, bboxes = mser.detectRegions(img_gray)
mser_char_mask = np.zeros(img_gray.shape, dtype=np.uint8)
mser_line_mask = np.zeros(img_gray.shape, dtype=np.uint8)

max_mser_char_dim = 0.0
for r in regions:
    rect = cv2.minAreaRect(r)
    box_pts = cv2.boxPoints(rect)
    v1 = box_pts[1] - box_pts[0]
    v2 = box_pts[2] - box_pts[1]
    L1 = np.linalg.norm(v1)
    L2 = np.linalg.norm(v2)
    L_min = min(L1, L2)
    L_max = max(L1, L2)
    aspect_ratio = L_max / L_min if L_min != 0 else 0
    box_pts = np.intp(box_pts)
    
    if 3 <= L1 <= 80 and 3 <= L2 <= 80 and aspect_ratio <= 1.8:
        cv2.drawContours(mser_char_mask, [box_pts], 0, 255, -1)
        max_mser_char_dim = max(max_mser_char_dim, L_max)
    elif L_max >= 5 and aspect_ratio > 1.8:
        cv2.drawContours(mser_line_mask, [box_pts], 0, 255, -1)
        
if max_mser_char_dim == 0.0:
    max_mser_char_dim = 15.0  # smaller default for crops

# Connected components for small/large islands using adaptive thresholds
thresh_inv = cv2.bitwise_not(thresh)
num_labels, labels_im, stats, _ = cv2.connectedComponentsWithStats(thresh_inv, connectivity=8)
small_islands = np.zeros_like(thresh_inv)
large_islands = np.zeros_like(thresh_inv)

# Adaptive small island size for crop-level processing
small_w_thresh = 15
small_h_thresh = 15
small_area_thresh = 100

for label in range(1, num_labels):
    w_c = stats[label, cv2.CC_STAT_WIDTH]
    h_c = stats[label, cv2.CC_STAT_HEIGHT]
    area_c = stats[label, cv2.CC_STAT_AREA]
    comp_mask = (labels_im == label).astype(np.uint8) * 255
    
    if w_c <= small_w_thresh and h_c <= small_h_thresh and area_c <= small_area_thresh:
        small_islands = cv2.bitwise_or(small_islands, comp_mask)
    else:
        large_islands = cv2.bitwise_or(large_islands, comp_mask)
        
crop_pil_mser_char = Image.fromarray(mser_char_mask)
crop_pil_mser_line = Image.fromarray(mser_line_mask)
crop_pil_small_islands = Image.fromarray(small_islands)
crop_pil_large_islands = Image.fromarray(large_islands)

cleaned, erased_pts, review = clean_crop_lines(
    crop_pil_std,
    crop_pil_big,
    crop_pil_mser_char,
    crop_pil_mser_line,
    crop_pil_small_islands,
    crop_pil_large_islands,
    max_mser_char_dim
)

print(f"Adaptive crop-level processing:")
print(f"  Erased pixels: {len(erased_pts)}")

# Print ASCII of cleaned result
cleaned_gray = cv2.cvtColor(np.array(cleaned), cv2.COLOR_RGB2GRAY)
for y in range(h):
    line = ""
    for x in range(w):
        if cleaned_gray[y, x] < 180:
            line += "#"
        else:
            line += "."
    print(line)
