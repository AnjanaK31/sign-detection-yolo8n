import cv2
import numpy as np

crop_gt_col = cv2.imread("debug_output/crop_gt_col.png")
h, w, _ = crop_gt_col.shape

# Let's count pixels under various thresholds in BGR:
# Red is high R (index 2), low G (index 1) and B (index 0)
for r_thresh in [150, 200, 240]:
    for gb_thresh in [50, 100, 150]:
        red_pixels = (crop_gt_col[:, :, 2] > r_thresh) & (crop_gt_col[:, :, 1] < gb_thresh) & (crop_gt_col[:, :, 0] < gb_thresh)
        print(f"R > {r_thresh}, G & B < {gb_thresh}: {np.sum(red_pixels)} pixels")
