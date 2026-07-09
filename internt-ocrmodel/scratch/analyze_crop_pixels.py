import cv2
import numpy as np

# Load the saved crops
crop_orig = cv2.imread("debug_output/crop_orig.png")
crop_prep = cv2.imread("debug_output/crop_prep.png")
crop_gt_col = cv2.imread("debug_output/crop_gt_col.png")

# Standard gray threshold
gray_orig = cv2.cvtColor(crop_orig, cv2.COLOR_BGR2GRAY)
_, thresh_orig = cv2.threshold(gray_orig, 127, 255, cv2.THRESH_BINARY_INV)

gray_prep = cv2.cvtColor(crop_prep, cv2.COLOR_BGR2GRAY)
_, thresh_prep = cv2.threshold(gray_prep, 127, 255, cv2.THRESH_BINARY_INV)

# Extract gt text (red in BGR is B < 100, G < 100, R > 200)
gt_text = ((crop_gt_col[:, :, 0] < 100) & (crop_gt_col[:, :, 1] < 100) & (crop_gt_col[:, :, 2] > 200)).astype(np.uint8) * 255

print(f"Thresh Orig Text Pixels: {np.sum(thresh_orig > 0)}")
print(f"Thresh Prep Text Pixels: {np.sum(thresh_prep > 0)}")
print(f"GT Text (Red) Pixels: {np.sum(gt_text > 0)}")

# Overlaps
overlap_orig = np.sum((thresh_orig > 0) & (gt_text > 0))
overlap_prep = np.sum((thresh_prep > 0) & (gt_text > 0))

print(f"Overlap Orig with GT: {overlap_orig} (TPR = {overlap_orig / np.sum(gt_text > 0) if np.sum(gt_text > 0) > 0 else 0:.4f})")
print(f"Overlap Prep with GT: {overlap_prep} (TPR = {overlap_prep / np.sum(gt_text > 0) if np.sum(gt_text > 0) > 0 else 0:.4f})")
