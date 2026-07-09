import sys
import os
import cv2
import numpy as np
import math
from PIL import Image

sys.path.append(os.path.abspath(r"d:\Internship\OCR_PDF\BoudningBoxCleaning"))
from clean_page_expressions import (
    skeletonize,
    get_skeleton_junctions,
    split_skeleton_into_branches,
    classify_branch_shape,
    evaluate_path_fitness,
    get_straight_line_path,
    get_circle_path,
    find_border_crossing_points
)

img_path = r"d:\Internship\OCR_PDF\sign-detection-yolo8n\Nived-dataset\dataset\dataset\crop_img_backup\intra_crop_0.jpg"
img_bgr = cv2.imread(img_path)
h, w = img_bgr.shape[:2]
img_gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

# Threshold standard crop
_, thresh_std = cv2.threshold(img_gray, 180, 255, cv2.THRESH_BINARY_INV)

# Pad standard crop with 30 pixels to get big crop
pad_val = 30
thresh_big = cv2.copyMakeBorder(thresh_std, pad_val, pad_val, pad_val, pad_val, cv2.BORDER_CONSTANT, value=0)

H_std, W_std = thresh_std.shape
H_big, W_big = thresh_big.shape

# Distance transforms
dist_to_fg = cv2.distanceTransform(cv2.bitwise_not(thresh_std), cv2.DIST_L2, 5)
skel = skeletonize(thresh_std)
junctions = get_skeleton_junctions(skel)

crossing_pts = find_border_crossing_points(thresh_std)
print(f"Crossing points: {crossing_pts}")

# Connected component check on big crop to protect border-touching characters
dilated_big_check = cv2.dilate(thresh_big, cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3)))
num_labels_check, labels_big_check, stats_check, _ = cv2.connectedComponentsWithStats(dilated_big_check, connectivity=8)

dx = pad_val
dy = pad_val

extends_outside = {}
for lbl in range(1, num_labels_check):
    stat = stats_check[lbl]
    left = stat[cv2.CC_STAT_LEFT]
    top = stat[cv2.CC_STAT_TOP]
    width = stat[cv2.CC_STAT_WIDTH]
    height = stat[cv2.CC_STAT_HEIGHT]
    right = left + width
    bottom = top + height
    extends_outside[lbl] = (left < dx or right > dx + W_std or top < dy or bottom > dy + H_std)

# Text Protection: Filter border-touching components
num_std_labels, std_labels_im, std_stats, _ = cv2.connectedComponentsWithStats(thresh_std, connectivity=8)
text_protection_mask = np.zeros_like(thresh_std)
border_touching_mask = np.zeros_like(thresh_std)
margin = 2

for label in range(1, num_std_labels):
    comp_mask = std_labels_im == label
    ys, xs = np.where(comp_mask)
    if len(xs) > 0:
        touches_border = (np.any(xs <= margin) | np.any(xs >= W_std - 1 - margin) |
                          np.any(ys <= margin) | np.any(ys >= H_std - 1 - margin))
        if touches_border:
            border_indices = (xs <= margin) | (xs >= W_std - 1 - margin) | (ys <= margin) | (ys >= H_std - 1 - margin)
            border_xs = xs[border_indices]
            border_ys = ys[border_indices]
            
            is_crossing = False
            for bx, by in zip(border_xs, border_ys):
                big_x = bx + dx
                big_y = by + dy
                if 0 <= big_x < W_big and 0 <= big_y < H_big:
                    lbl_big = labels_big_check[big_y, big_x]
                    if lbl_big > 0 and extends_outside.get(lbl_big, False):
                        is_crossing = True
                        break
            if is_crossing:
                border_touching_mask[comp_mask] = 255
            else:
                text_protection_mask[comp_mask] = 255
        else:
            text_protection_mask[comp_mask] = 255

print(f"text_protection_mask non-zero: {cv2.countNonZero(text_protection_mask)}")

# Trace paths
# Evaluate candidates for each pair of crossing points
dilated_big = cv2.dilate(thresh_big, cv2.getStructuringElement(cv2.MORPH_RECT, (9, 9)))
num_labels, labels_im, stats, _ = cv2.connectedComponentsWithStats(dilated_big, connectivity=8)

crossing_labels = {}
for pt in crossing_pts:
    bx = pt[0] + dx
    by = pt[1] + dy
    if 0 <= bx < W_big and 0 <= by < H_big:
        lbl = labels_im[by, bx]
        if lbl > 0:
            crossing_labels[pt] = lbl

erase_mask_trace = np.zeros_like(thresh_std)
n_pts = len(crossing_pts)
for i in range(n_pts):
    for j in range(i + 1, n_pts):
        p1 = crossing_pts[i]
        p2 = crossing_pts[j]
        
        dist = math.hypot(p2[0] - p1[0], p2[1] - p1[1])
        if dist < 12:
            continue
            
        lbl1 = crossing_labels.get(p1, -1)
        lbl2 = crossing_labels.get(p2, -2)
        
        pts_fg = []
        if lbl1 == lbl2 and lbl1 > 0:
            ys, xs = np.where(labels_im[dy:dy+H_std, dx:dx+W_std] == lbl1)
            matching_mask = thresh_std[ys, xs] > 0
            xs = xs[matching_mask]
            ys = ys[matching_mask]
            if len(xs) > 0:
                pts_fg = np.column_stack((xs, ys))
                
        path_straight = get_straight_line_path(p1, p2)
        cov_s, err_s, pts_s = evaluate_path_fitness(path_straight, dist_to_fg, H_std, W_std)
        
        # Check if straight path fits
        if cov_s >= 0.65 and err_s <= 2.5:
            path_mask = np.zeros_like(thresh_std)
            for (x, y) in pts_s:
                cv2.circle(path_mask, (x, y), 2, 255, -1)
            candidate_erase = cv2.bitwise_and(thresh_std, path_mask)
            erase_mask_trace = cv2.bitwise_or(erase_mask_trace, candidate_erase)

print(f"erase_mask_trace non-zero: {cv2.countNonZero(erase_mask_trace)}")

# Branch classification
branches, _ = split_skeleton_into_branches(skel)
branch_erase_mask = np.zeros_like(thresh_std)
branch_protection_mask = np.zeros_like(thresh_std)
border_margin = 3

for branch_pts in branches:
    branch_thickness = 2 * np.mean([dist_to_fg[pt[1], pt[0]] for pt in branch_pts])
    shape_type, score = classify_branch_shape(branch_pts, thresh_px=1.5)
    touches_border = np.any((branch_pts[:, 0] < border_margin) | 
                            (branch_pts[:, 0] > W_std - 1 - border_margin) | 
                            (branch_pts[:, 1] < border_margin) | 
                            (branch_pts[:, 1] > H_std - 1 - border_margin))
    
    xs = branch_pts[:, 0]
    ys = branch_pts[:, 1]
    branch_length = max(np.max(xs) - np.min(xs), np.max(ys) - np.min(ys))
    is_long_line = (branch_length > 0.5 * max(W_std, H_std)) or (branch_length > 2.0 * 15.0)
    
    is_line = False
    if shape_type in ["straight", "curved"]:
        if touches_border or is_long_line:
            is_line = True
    elif shape_type == "unknown":
        if is_long_line:
            is_line = True
            
    branch_mask = np.zeros_like(thresh_std)
    for pt in branch_pts:
        cv2.circle(branch_mask, (pt[0], pt[1]), 2, 255, -1)
    branch_pixels = cv2.bitwise_and(thresh_std, branch_mask)
    
    if is_line and branch_thickness < 2.6:
        branch_erase_mask = cv2.bitwise_or(branch_erase_mask, branch_pixels)
    else:
        branch_protection_mask = cv2.bitwise_or(branch_protection_mask, branch_pixels)

print(f"branch_erase_mask non-zero: {cv2.countNonZero(branch_erase_mask)}")
print(f"branch_protection_mask non-zero: {cv2.countNonZero(branch_protection_mask)}")
