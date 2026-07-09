import sys
import os
import cv2
import numpy as np
import math

sys.path.append(os.path.abspath(r"d:\Internship\OCR_PDF\BoudningBoxCleaning"))
from clean_page_expressions import (
    skeletonize,
    get_skeleton_junctions,
    split_skeleton_into_branches,
    classify_branch_shape
)

img_path = r"d:\Internship\OCR_PDF\sign-detection-yolo8n\Nived-dataset\dataset\dataset\crop_img_backup\intra_crop_0.jpg"
img_bgr = cv2.imread(img_path)
img_gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
_, thresh_std = cv2.threshold(img_gray, 180, 255, cv2.THRESH_BINARY_INV)

dist_to_fg = cv2.distanceTransform(cv2.bitwise_not(thresh_std), cv2.DIST_L2, 5)
skel = skeletonize(thresh_std)
branches, junctions = split_skeleton_into_branches(skel)

print(f"Number of branches detected: {len(branches)}")
for idx, branch_pts in enumerate(branches):
    branch_thickness = 2 * np.mean([dist_to_fg[pt[1], pt[0]] for pt in branch_pts])
    shape_type, score = classify_branch_shape(branch_pts, thresh_px=1.5)
    
    # Bounding box span
    xs = branch_pts[:, 0]
    ys = branch_pts[:, 1]
    w_std, h_std = thresh_std.shape[1], thresh_std.shape[0]
    branch_length = max(np.max(xs) - np.min(xs), np.max(ys) - np.min(ys))
    
    border_margin = 3
    touches_border = np.any((branch_pts[:, 0] < border_margin) | 
                            (branch_pts[:, 0] > w_std - 1 - border_margin) | 
                            (branch_pts[:, 1] < border_margin) | 
                            (branch_pts[:, 1] > h_std - 1 - border_margin))
                            
    is_long_line = (branch_length > 0.5 * max(w_std, h_std)) or (branch_length > 2.0 * 15.0)
    
    print(f"Branch {idx}:")
    print(f"  Length (pixels in skel): {len(branch_pts)}")
    print(f"  Bounding box: x=[{np.min(xs)}, {np.max(xs)}], y=[{np.min(ys)}, {np.max(ys)}]")
    print(f"  Thickness: {branch_thickness:.2f}")
    print(f"  Shape: {shape_type}")
    print(f"  Touches border: {touches_border}")
    print(f"  Is long line: {is_long_line}")
