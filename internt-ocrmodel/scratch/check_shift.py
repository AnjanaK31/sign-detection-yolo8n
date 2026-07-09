import cv2
import numpy as np

# Load the saved crops
crop_orig = cv2.imread("debug_output/crop_orig.png")
crop_gt_col = cv2.imread("debug_output/crop_gt_col.png")

gray_orig = cv2.cvtColor(crop_orig, cv2.COLOR_BGR2GRAY)
_, thresh_orig = cv2.threshold(gray_orig, 127, 255, cv2.THRESH_BINARY_INV)

gt_text = ((crop_gt_col[:, :, 0] < 100) & (crop_gt_col[:, :, 1] < 100) & (crop_gt_col[:, :, 2] > 200)).astype(np.uint8) * 255

best_overlap = 0
best_shift = (0, 0)
h, w = thresh_orig.shape

for dy in range(-10, 11):
    for dx in range(-10, 11):
        # Shift thresh_orig
        M = np.float32([[1, 0, dx], [0, 1, dy]])
        shifted = cv2.warpAffine(thresh_orig, M, (w, h))
        overlap = np.sum((shifted > 0) & (gt_text > 0))
        if overlap > best_overlap:
            best_overlap = overlap
            best_shift = (dx, dy)

print(f"No shift overlap: {np.sum((thresh_orig > 0) & (gt_text > 0))} (TPR = {np.sum((thresh_orig > 0) & (gt_text > 0)) / np.sum(gt_text > 0):.4f})")
print(f"Best shift: {best_shift} -> Overlap: {best_overlap} (TPR = {best_overlap / np.sum(gt_text > 0):.4f})")
