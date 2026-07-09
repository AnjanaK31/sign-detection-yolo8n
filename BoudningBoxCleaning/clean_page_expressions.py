import os
import sys
import math
import cv2
import numpy as np
import torch
import argparse
from PIL import Image
from ultralytics import YOLO

# Reconfigure stdout/stderr to support printing unicode characters on Windows
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")

# ── Rectification Helpers ───────────────────────────────────────────────────────

def rectify_crop(image, bbox_metrics, buffer_percent=0.20):
    """Extracts and deskews a rotated bounding box from an image.
    Adapted from rectifier.py to be self-contained.
    """
    is_pil = isinstance(image, Image.Image)
    if is_pil:
        img_np = np.array(image.convert("RGB"))
    else:
        img_np = image.copy()
        
    cx = bbox_metrics['cx']
    cy = bbox_metrics['cy']
    w = bbox_metrics['w']
    h = bbox_metrics['h']
    angle = bbox_metrics['angle']
    
    bw = w * (1.0 + buffer_percent)
    bh = h * (1.0 + buffer_percent)
    
    diag = math.sqrt(bw**2 + bh**2)
    crop_size = int(math.ceil(diag)) + 20
    half_size = crop_size // 2
    
    x_min = int(round(cx - half_size))
    y_min = int(round(cy - half_size))
    x_max = x_min + crop_size
    y_max = y_min + crop_size
    
    img_h, img_w = img_np.shape[:2]
    
    pad_left = max(0, -x_min)
    pad_top = max(0, -y_min)
    pad_right = max(0, x_max - img_w)
    pad_bottom = max(0, y_max - img_h)
    
    src_x_min = max(0, x_min)
    src_y_min = max(0, y_min)
    src_x_max = min(img_w, x_max)
    src_y_max = min(img_h, y_max)
    
    sub_img = img_np[src_y_min:src_y_max, src_x_min:src_x_max]
    
    if len(img_np.shape) == 3:
        padded_crop = np.ones((crop_size, crop_size, 3), dtype=np.uint8) * 255
        padded_crop[pad_top:crop_size - pad_bottom, pad_left:crop_size - pad_right] = sub_img
    else:
        padded_crop = np.ones((crop_size, crop_size), dtype=np.uint8) * 255
        padded_crop[pad_top:crop_size - pad_bottom, pad_left:crop_size - pad_right] = sub_img
        
    rot_center = (half_size, half_size)
    M = cv2.getRotationMatrix2D(rot_center, angle, 1.0)
    
    warped = cv2.warpAffine(
        padded_crop, 
        M, 
        (crop_size, crop_size), 
        flags=cv2.INTER_CUBIC, 
        borderMode=cv2.BORDER_CONSTANT, 
        borderValue=(255, 255, 255) if len(img_np.shape) == 2 else (255, 255, 255)
    )
    
    bx_min = int(round(half_size - bw / 2))
    by_min = int(round(half_size - bh / 2))
    bx_max = bx_min + int(round(bw))
    by_max = by_min + int(round(bh))
    
    bx_min = max(0, bx_min)
    by_min = max(0, by_min)
    bx_max = min(crop_size, bx_max)
    by_max = min(crop_size, by_max)
    
    final_crop = warped[by_min:by_max, bx_min:bx_max]
    
    if is_pil:
        return Image.fromarray(final_crop)
    return final_crop


def map_crop_to_page_coordinates(pts_crop, cx, cy, w, h, angle_deg, buffer_percent=0.20):
    """Maps coordinates from a rectified standard crop back to full page coordinates.
    Adapted from line_cleaner.py.
    """
    if len(pts_crop) == 0:
        return np.empty((0, 2))
    bw = w * (1.0 + buffer_percent)
    bh = h * (1.0 + buffer_percent)
    diag = math.sqrt(bw**2 + bh**2)
    crop_size = int(math.ceil(diag)) + 20
    half_size = crop_size // 2
    
    bx_min = int(round(half_size - bw / 2))
    by_min = int(round(half_size - bh / 2))
    
    pts_warped = pts_crop + np.array([bx_min, by_min])
    
    rad = math.radians(-angle_deg)
    cos_a = math.cos(rad)
    sin_a = math.sin(rad)
    
    dx = pts_warped[:, 0] - half_size
    dy = pts_warped[:, 1] - half_size
    
    x_padded = half_size + dx * cos_a - dy * sin_a
    y_padded = half_size + dx * sin_a + dy * cos_a
    
    x_min = int(round(cx - half_size))
    y_min = int(round(cy - half_size))
    
    x_page = x_padded + x_min
    y_page = y_padded + y_min
    
    return np.column_stack((x_page, y_page))


# ── Line Cleaning / Path Fitting Algorithms ──────────────────────────────────────

def skeletonize(img: np.ndarray) -> np.ndarray:
    """Applies Zhang-Suen thinning algorithm to guarantee a thin 1-pixel wide skeleton."""
    binary = (img > 0).astype(np.uint8)
    while True:
        padded = np.pad(binary, 1, mode='constant', constant_values=0)
        P2 = padded[:-2, 1:-1]
        P3 = padded[:-2, 2:]
        P4 = padded[1:-1, 2:]
        P5 = padded[2:, 2:]
        P6 = padded[2:, 1:-1]
        P7 = padded[2:, :-2]
        P8 = padded[1:-1, :-2]
        P9 = padded[:-2, :-2]
        
        B = P2 + P3 + P4 + P5 + P6 + P7 + P8 + P9
        A = ((P2 == 0) & (P3 == 1)).astype(np.uint8) + \
            ((P3 == 0) & (P4 == 1)).astype(np.uint8) + \
            ((P4 == 0) & (P5 == 1)).astype(np.uint8) + \
            ((P5 == 0) & (P6 == 1)).astype(np.uint8) + \
            ((P6 == 0) & (P7 == 1)).astype(np.uint8) + \
            ((P7 == 0) & (P8 == 1)).astype(np.uint8) + \
            ((P8 == 0) & (P9 == 1)).astype(np.uint8) + \
            ((P9 == 0) & (P2 == 1)).astype(np.uint8)
            
        cond1 = (B >= 2) & (B <= 6)
        cond2 = (A == 1)
        cond3 = (P2 * P4 * P6 == 0)
        cond4 = (P4 * P6 * P8 == 0)
        
        to_delete = (binary == 1) & cond1 & cond2 & cond3 & cond4
        if not np.any(to_delete):
            step1_changed = False
        else:
            binary[to_delete] = 0
            step1_changed = True
            
        padded = np.pad(binary, 1, mode='constant', constant_values=0)
        P2 = padded[:-2, 1:-1]
        P3 = padded[:-2, 2:]
        P4 = padded[1:-1, 2:]
        P5 = padded[2:, 2:]
        P6 = padded[2:, 1:-1]
        P7 = padded[2:, :-2]
        P8 = padded[1:-1, :-2]
        P9 = padded[:-2, :-2]
        
        B = P2 + P3 + P4 + P5 + P6 + P7 + P8 + P9
        A = ((P2 == 0) & (P3 == 1)).astype(np.uint8) + \
            ((P3 == 0) & (P4 == 1)).astype(np.uint8) + \
            ((P4 == 0) & (P5 == 1)).astype(np.uint8) + \
            ((P5 == 0) & (P6 == 1)).astype(np.uint8) + \
            ((P6 == 0) & (P7 == 1)).astype(np.uint8) + \
            ((P7 == 0) & (P8 == 1)).astype(np.uint8) + \
            ((P8 == 0) & (P9 == 1)).astype(np.uint8) + \
            ((P9 == 0) & (P2 == 1)).astype(np.uint8)
            
        cond1 = (B >= 2) & (B <= 6)
        cond2 = (A == 1)
        cond3 = (P2 * P4 * P8 == 0)
        cond4 = (P2 * P6 * P8 == 0)
        
        to_delete2 = (binary == 1) & cond1 & cond2 & cond3 & cond4
        if not np.any(to_delete2):
            step2_changed = False
        else:
            binary[to_delete2] = 0
            step2_changed = True
            
        if not step1_changed and not step2_changed:
            break
            
    return (binary * 255).astype(np.uint8)


def get_skeleton_junctions(skel: np.ndarray) -> np.ndarray:
    """Finds junctions in the skeleton where lines meet/intersect."""
    binary = (skel > 0).astype(np.uint8)
    padded = np.pad(binary, 1, mode='constant', constant_values=0)
    
    P2 = padded[:-2, 1:-1]
    P3 = padded[:-2, 2:]
    P4 = padded[1:-1, 2:]
    P5 = padded[2:, 2:]
    P6 = padded[2:, 1:-1]
    P7 = padded[2:, :-2]
    P8 = padded[1:-1, :-2]
    P9 = padded[:-2, :-2]
    
    T = ((P2 == 0) & (P3 == 1)).astype(np.uint8) + \
        ((P3 == 0) & (P4 == 1)).astype(np.uint8) + \
        ((P4 == 0) & (P5 == 1)).astype(np.uint8) + \
        ((P5 == 0) & (P6 == 1)).astype(np.uint8) + \
        ((P6 == 0) & (P7 == 1)).astype(np.uint8) + \
        ((P7 == 0) & (P8 == 1)).astype(np.uint8) + \
        ((P8 == 0) & (P9 == 1)).astype(np.uint8) + \
        ((P9 == 0) & (P2 == 1)).astype(np.uint8)
        
    junctions = (binary == 1) & (T >= 3)
    return (junctions * 255).astype(np.uint8)


def find_border_crossing_points(thresh):
    """Finds centroid coordinates where lines touch/cross the crop boundary."""
    H, W = thresh.shape
    border_pts = []
    
    # Check top and bottom borders (horizontal runs)
    for y in [0, H-1]:
        in_run = False
        run_xs = []
        for x in range(W):
            if thresh[y, x] > 0:
                in_run = True
                run_xs.append(x)
            else:
                if in_run:
                    border_pts.append((int(np.mean(run_xs)), y))
                    in_run = False
                    run_xs = []
        if in_run:
            border_pts.append((int(np.mean(run_xs)), y))
            
    # Check left and right borders (vertical runs)
    for x in [0, W-1]:
        in_run = False
        run_ys = []
        for y in range(H):
            if y == 0 or y == H-1:  # skip corners to avoid duplicate entries
                continue
            if thresh[y, x] > 0:
                in_run = True
                run_ys.append(y)
            else:
                if in_run:
                    border_pts.append((x, int(np.mean(run_ys))))
                    in_run = False
                    run_ys = []
        if in_run:
            border_pts.append((x, int(np.mean(run_ys))))
            
    return border_pts


def fit_circle(pts):
    """Fits a circle to a set of 2D points using linear least-squares."""
    x = pts[:, 0]
    y = pts[:, 1]
    N = len(pts)
    A = np.column_stack((x, y, np.ones(N)))
    B = -x**2 - y**2
    try:
        W, residuals, rank, s = np.linalg.lstsq(A, B, rcond=None)
        a, b, c = W
        xc = -a / 2.0
        yc = -b / 2.0
        r_sq = xc**2 + yc**2 - c
        if r_sq < 0:
            return None, None, None
        R = math.sqrt(r_sq)
        return xc, yc, R
    except:
        return None, None, None


def get_straight_line_path(p1, p2):
    """Generates points along a straight line path connecting two coordinates."""
    x1, y1 = p1
    x2, y2 = p2
    dist = int(math.hypot(x2 - x1, y2 - y1))
    if dist == 0:
        return []
    path_pts = []
    for i in range(dist + 1):
        t = i / dist
        x = int(round(x1 + t * (x2 - x1)))
        y = int(round(y1 + t * (y2 - y1)))
        path_pts.append((x, y))
    return path_pts


def get_quadratic_path(p1, p2, pts_fg):
    """Fits and generates points along a quadratic curve path connecting p1 and p2."""
    x1, y1 = p1
    x2, y2 = p2
    
    xs = pts_fg[:, 0].astype(np.float64)
    ys = pts_fg[:, 1].astype(np.float64)
    
    if np.std(xs) > np.std(ys):
        try:
            poly = np.polyfit(xs, ys, 2)
            num_steps = int(abs(x2 - x1))
            if num_steps == 0:
                return []
            path_pts = []
            for i in range(num_steps + 1):
                t = i / num_steps
                x = x1 + t * (x2 - x1)
                y = np.polyval(poly, x)
                path_pts.append((int(round(x)), int(round(y))))
            return path_pts
        except:
            return []
    else:
        try:
            poly = np.polyfit(ys, xs, 2)
            num_steps = int(abs(y2 - y1))
            if num_steps == 0:
                return []
            path_pts = []
            for i in range(num_steps + 1):
                t = i / num_steps
                y = y1 + t * (y2 - y1)
                x = np.polyval(poly, y)
                path_pts.append((int(round(x)), int(round(y))))
            return path_pts
        except:
            return []


def get_circle_path(p1, p2, pts_fg):
    """Fits and generates points along a circular arc path connecting p1 and p2."""
    x1, y1 = p1
    x2, y2 = p2
    
    xc, yc, R = fit_circle(pts_fg)
    if xc is None or R is None or R > max(abs(x1-x2), abs(y1-y2)) * 10:
        return []
        
    theta1 = math.atan2(y1 - yc, x1 - xc)
    theta2 = math.atan2(y2 - yc, x2 - xc)
    
    diff = theta2 - theta1
    diff = (diff + math.pi) % (2 * math.pi) - math.pi
    
    arc_length = R * abs(diff)
    num_steps = int(max(5, arc_length))
    
    path_pts = []
    for i in range(num_steps + 1):
        t = i / num_steps
        theta = theta1 + t * diff
        x = xc + R * math.cos(theta)
        y = yc + R * math.sin(theta)
        path_pts.append((int(round(x)), int(round(y))))
    return path_pts


def evaluate_path_fitness(path_pts, dist_to_fg, H, W, cover_dist_thresh=2.0):
    """Computes coverage fraction (points near foreground) and average distance for a path."""
    if len(path_pts) == 0:
        return 0.0, float('inf'), []
        
    valid_path_pts = []
    distances = []
    close_count = 0
    
    for (x, y) in path_pts:
        if 0 <= x < W and 0 <= y < H:
            dist = dist_to_fg[y, x]
            distances.append(dist)
            valid_path_pts.append((x, y))
            if dist <= cover_dist_thresh:
                close_count += 1
                
    if len(valid_path_pts) == 0:
        return 0.0, float('inf'), []
        
    coverage = close_count / len(valid_path_pts)
    avg_dist = np.mean(distances)
    
    return coverage, avg_dist, valid_path_pts


def split_skeleton_into_branches(skel: np.ndarray) -> tuple:
    """Splits the skeleton at junction points. Returns (branches, junctions)."""
    junctions = get_skeleton_junctions(skel)
    
    # Dilate junctions to cleanly disconnect intersecting branches
    junc_dilate_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    junctions_dilated = cv2.dilate(junctions, junc_dilate_kernel)
    
    skel_no_junc = cv2.subtract(skel, junctions_dilated)
    
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(skel_no_junc, connectivity=8)
    
    branches = []
    for label in range(1, num_labels):
        ys, xs = np.where(labels == label)
        if len(xs) > 0:
            pts = np.column_stack((xs, ys))
            branches.append(pts)
            
    return branches, junctions


def fit_polynomial(pts: np.ndarray) -> float:
    """Fits quadratic polynomials y = f(x) and x = f(y) and returns the minimum fitting error."""
    x = pts[:, 0].astype(np.float64)
    y = pts[:, 1].astype(np.float64)
    
    try:
        p_y = np.polyfit(x, y, 2)
        pred_y = np.polyval(p_y, x)
        err_y = np.mean(np.abs(y - pred_y))
    except:
        err_y = float('inf')
        
    try:
        p_x = np.polyfit(y, x, 2)
        pred_x = np.polyval(p_x, y)
        err_x = np.mean(np.abs(x - pred_x))
    except:
        err_x = float('inf')
        
    return min(err_y, err_x)


def classify_branch_shape(pts: np.ndarray, thresh_px: float = 1.2) -> tuple:
    """Classifies a branch path as 'straight', 'curved', or 'unknown'."""
    N = len(pts)
    if N < 5:
        return "noise", 0.0
        
    x = pts[:, 0]
    y = pts[:, 1]
    coords = np.column_stack((x, y))
    
    # 1. Straight Line Fit via PCA
    cov = np.cov(coords, rowvar=False)
    if cov.ndim < 2 or np.isnan(cov).any():
        return "noise", 0.0
        
    evals, evecs = np.linalg.eigh(cov)
    min_eval = max(0.0, min(evals))
    sigma_pca = math.sqrt(min_eval)
    
    if sigma_pca < thresh_px:
        score = 1.0 - (sigma_pca / thresh_px)
        return "straight", score
        
    # 2. Circle/Arc Fit
    xc, yc, R = fit_circle(pts)
    if R is not None:
        dists = np.sqrt((x - xc)**2 + (y - yc)**2)
        err_circle = np.mean(np.abs(dists - R))
        if err_circle < thresh_px and R < max(np.max(x) - np.min(x), np.max(y) - np.min(y)) * 5:
            if R <= 22.0:
                return "unknown", 0.0
            score = 1.0 - (err_circle / thresh_px)
            return "curved", score
        
    # 3. Quadratic Polynomial Fit
    err_poly = fit_polynomial(pts)
    if err_poly < thresh_px:
        score = 1.0 - (err_poly / thresh_px)
        return "curved", score
        
    return "unknown", 0.0


def clean_crop_lines(crop_pil_std, crop_pil_big, crop_pil_mser_char=None, crop_pil_mser_line=None, crop_pil_small_islands=None, crop_pil_large_islands=None, max_mser_char_dim=30.0, cover_ratio_thresh=0.65, error_dist_thresh=2.5):
    """Cleans lines (solid, curved, dotted) in standard crop using the big crop for tracing context."""
    img_std = np.array(crop_pil_std.convert("L"))
    img_big = np.array(crop_pil_big.convert("L"))
    
    H_std, W_std = img_std.shape
    H_big, W_big = img_big.shape
    
    # Invert binary threshold (foreground = 255, background = 0)
    _, thresh_std = cv2.threshold(img_std, 180, 255, cv2.THRESH_BINARY_INV)
    _, thresh_big = cv2.threshold(img_big, 180, 255, cv2.THRESH_BINARY_INV)
    
    text_protection_mask = np.zeros_like(thresh_std)
    if crop_pil_mser_char is not None:
        mser_char_np = np.array(crop_pil_mser_char.convert("L"))
        text_protection_mask[mser_char_np > 127] = 255
        
    # Calculate local max MSER character dimension to avoid page-level outlier scaling issues
    max_mser_char_dim_local = 0.0
    if crop_pil_mser_char is not None:
        _, thresh_mser_char = cv2.threshold(np.array(crop_pil_mser_char.convert("L")), 127, 255, cv2.THRESH_BINARY)
        num_labels_mser, labels_mser, stats_mser, _ = cv2.connectedComponentsWithStats(thresh_mser_char, connectivity=8)
        for label in range(1, num_labels_mser):
            w_m = stats_mser[label, cv2.CC_STAT_WIDTH]
            h_m = stats_mser[label, cv2.CC_STAT_HEIGHT]
            max_mser_char_dim_local = max(max_mser_char_dim_local, w_m, h_m)
            
    if max_mser_char_dim_local == 0.0:
        max_mser_char_dim_local = 30.0
        
    # border_touching_mask = np.zeros_like(thresh_std)
    # margin = 2
    # for label in range(1, num_std_labels):
    #     comp_mask = std_labels_im == label
    #     ys, xs = np.where(comp_mask)
    #     if len(xs) > 0:
    #         touches_border = (np.any(xs <= margin) | np.any(xs >= W_std - 1 - margin) |
    #                           np.any(ys <= margin) | np.any(ys >= H_std - 1 - margin))
    #         if touches_border:
    #             border_touching_mask[comp_mask] = 255
    #         else:
    #             text_protection_mask[comp_mask] = 255
                
    # Distance transform to nearest foreground pixel (black in original)
    dist_to_fg = cv2.distanceTransform(cv2.bitwise_not(thresh_std), cv2.DIST_L2, 5)
    
    # Stroke thickness map
    thickness_map = 2 * cv2.distanceTransform(thresh_std, cv2.DIST_L2, 5)
    
    # Skeleton and junctions for junction protection
    skel = skeletonize(thresh_std)
    junctions = get_skeleton_junctions(skel)
    protected_junction_zone = cv2.dilate(junctions, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7)))
    
    # Find boundary crossing points
    crossing_pts = find_border_crossing_points(thresh_std)
    
    # Temporarily dilate the big crop to merge dotted lines/dashes for connected component association
    dilated_big = cv2.dilate(thresh_big, cv2.getStructuringElement(cv2.MORPH_RECT, (9, 9)))
    num_labels, labels_im, stats, _ = cv2.connectedComponentsWithStats(dilated_big, connectivity=8)
    
    # Calculate crop offsets inside the big crop
    dx = (W_big - W_std) // 2
    dy = (H_big - H_std) // 2
    
    # Map crossing points to the big crop space to query connected components
    crossing_labels = {}
    for pt in crossing_pts:
        bx = pt[0] + dx
        by = pt[1] + dy
        if 0 <= bx < W_big and 0 <= by < H_big:
            lbl = labels_im[by, bx]
            if lbl > 0:
                crossing_labels[pt] = lbl
                
    erase_mask = np.zeros_like(thresh_std)
    
    # Evaluate candidates for each pair of crossing points
    n_pts = len(crossing_pts)
    for i in range(n_pts):
        for j in range(i + 1, n_pts):
            p1 = crossing_pts[i]
            p2 = crossing_pts[j]
            
            # Distance filter
            dist = math.hypot(p2[0] - p1[0], p2[1] - p1[1])
            if dist < 12:
                continue
                
            # Grab foreground pixels of the associated component if they belong to the same component in dilated image
            lbl1 = crossing_labels.get(p1, -1)
            lbl2 = crossing_labels.get(p2, -2)
            
            pts_fg = []
            if lbl1 == lbl2 and lbl1 > 0:
                # Find original foreground pixels matching this component label in standard crop
                ys, xs = np.where(labels_im[dy:dy+H_std, dx:dx+W_std] == lbl1)
                # Keep only those that are foreground in the un-dilated standard crop
                matching_mask = thresh_std[ys, xs] > 0
                xs = xs[matching_mask]
                ys = ys[matching_mask]
                if len(xs) > 0:
                    pts_fg = np.column_stack((xs, ys))
            
            # 1. Straight path
            path_straight = get_straight_line_path(p1, p2)
            cov_s, err_s, pts_s = evaluate_path_fitness(path_straight, dist_to_fg, H_std, W_std)
            
            # 2. Circle path
            path_circ = []
            cov_c, err_c, pts_c = 0.0, float('inf'), []
            if len(pts_fg) >= 5:
                path_circ = get_circle_path(p1, p2, pts_fg)
                cov_c, err_c, pts_c = evaluate_path_fitness(path_circ, dist_to_fg, H_std, W_std)
                
            # Choose the best path among the candidates
            best_cov = -1.0
            best_err = float('inf')
            best_path_pts = []
            best_type = ""
            
            for cov, err, path, p_type in [(cov_s, err_s, pts_s, "straight"), 
                                           (cov_c, err_c, pts_c, "circle")]:
                if len(path) == 0:
                    continue
                # Priorities: high coverage, then low error
                if cov > best_cov or (abs(cov - best_cov) < 0.02 and err < best_err):
                    best_cov = cov
                    best_err = err
                    best_path_pts = path
                    best_type = p_type
                    
            # Check thresholds to verify path
            if best_cov >= cover_ratio_thresh and best_err <= error_dist_thresh:
                # Mark pixels near the best verified path for erasure
                path_mask = np.zeros_like(thresh_std)
                for (x, y) in best_path_pts:
                    cv2.circle(path_mask, (x, y), 2, 255, -1)  # dilate the path slightly to cover line thickness
                
                # Highlight and intersect with original foreground
                candidate_erase = cv2.bitwise_and(thresh_std, path_mask)
                
                erase_mask = cv2.bitwise_or(erase_mask, candidate_erase)
                
    # ── Solid & Leader Line Removal (Branch Classification) ──
    branches, _ = split_skeleton_into_branches(skel)
    branch_erase_mask = np.zeros_like(thresh_std)
    branch_protection_mask = np.zeros_like(thresh_std)
    
    border_margin = 3
    for branch_pts in branches:
        # Check thickness of the branch
        branch_thickness = 2 * np.mean([dist_to_fg[pt[1], pt[0]] for pt in branch_pts])
        
        # Build branch mask and get foreground pixels
        branch_mask = np.zeros_like(thresh_std)
        for pt in branch_pts:
            cv2.circle(branch_mask, (pt[0], pt[1]), 2, 255, -1)
        branch_pixels = cv2.bitwise_and(thresh_std, branch_mask)
        
        if branch_thickness >= 2.6:
            branch_protection_mask = cv2.bitwise_or(branch_protection_mask, branch_pixels)
            continue
            
        # Classify the branch shape
        shape_type, score = classify_branch_shape(branch_pts, thresh_px=1.5)
        
        # Check if the branch touches the border of the standard crop (CAD / leader line candidate)
        touches_border = np.any((branch_pts[:, 0] < border_margin) | 
                                (branch_pts[:, 0] > W_std - 1 - border_margin) | 
                                (branch_pts[:, 1] < border_margin) | 
                                (branch_pts[:, 1] > H_std - 1 - border_margin))
                                
        # Bounding box span length of the branch
        xs = branch_pts[:, 0]
        ys = branch_pts[:, 1]
        branch_length = max(np.max(xs) - np.min(xs), np.max(ys) - np.min(ys))
        
        # Long line criteria (even if it doesn't touch the border):
        # 1. > 50% length of the whole image (crop)
        # 2. > 200% of the biggest MSER character box dimension across the page
        is_long_line = (branch_length > 0.5 * max(W_std, H_std)) or (branch_length > 2.0 * max_mser_char_dim)
                                
        is_line = False
        if shape_type in ["straight", "curved"]:
            if touches_border or is_long_line:
                is_line = True
        elif shape_type == "unknown":
            if is_long_line:
                is_line = True
                
        if is_line:
            branch_erase_mask = cv2.bitwise_or(branch_erase_mask, branch_pixels)
        else:
            branch_protection_mask = cv2.bitwise_or(branch_protection_mask, branch_pixels)
            
    # Method A (Tracing): The union of path-tracing candidate pixels (erase_mask) and branch classifier pixels (branch_erase_mask)
    trace_line_mask = cv2.bitwise_or(erase_mask, branch_erase_mask)
    
    # Method B (MSER Lines): The cropped MSER line mask
    # Subtract text_protection pixels: a character stroke cannot also vote as a line
    mser_line_mask = np.zeros_like(thresh_std)
    if crop_pil_mser_line is not None:
        mser_line_np = np.array(crop_pil_mser_line.convert("L"))
        mser_line_mask[mser_line_np > 127] = 255
    mser_line_mask = cv2.bitwise_and(mser_line_mask, thresh_std)
    mser_line_mask = cv2.bitwise_and(mser_line_mask, cv2.bitwise_not(text_protection_mask))
    
    # Method C (Large Islands): The cropped large islands mask from page-level connected components
    # Subtract text_protection pixels: text chars that share a page-level island with a line
    # should not have their pixels vote as line candidates
    large_islands_mask = np.zeros_like(thresh_std)
    if crop_pil_large_islands is not None:
        large_islands_np = np.array(crop_pil_large_islands.convert("L"))
        large_islands_mask[large_islands_np > 127] = 255
    large_islands_mask = cv2.bitwise_and(large_islands_mask, thresh_std)
    large_islands_mask = cv2.bitwise_and(large_islands_mask, cv2.bitwise_not(text_protection_mask))
    
    # Small Islands protection mask from page-level connected components
    small_islands_mask = np.zeros_like(thresh_std)
    if crop_pil_small_islands is not None:
        small_islands_np = np.array(crop_pil_small_islands.convert("L"))
        small_islands_mask[small_islands_np > 127] = 255
    small_islands_mask = cv2.bitwise_and(small_islands_mask, thresh_std)
            
    # Strict No-Erase Zone: Union of MSER character mask, Small Islands, and Branch Protection
    strict_no_erase_mask = cv2.bitwise_or(cv2.bitwise_or(text_protection_mask, small_islands_mask), branch_protection_mask)
    
    # Consensus: Pixel-level consensus of 2-out-of-3 line methods
    consensus_line_mask = ( (trace_line_mask & mser_line_mask) | 
                            (mser_line_mask & large_islands_mask) | 
                            (trace_line_mask & large_islands_mask) )
                            
    # Final Erasure: consensus minus strict protection
    erase_mask = cv2.bitwise_and(consensus_line_mask, cv2.bitwise_not(strict_no_erase_mask))
    
    # ── Connected-Component Safety Sweep ──────────────────────────────────────────
    # For any CC in the binary crop that has erased pixels:
    #   (a) Primary: rescue if it overlaps with MSER char / small islands zone
    #   (b) Rescue if the CC is too small/short to be a significant line (long_side < 15px)
    #   (c) Rescue if the CC doesn't look like a thin spanning line geometrically (AR < 6 OR short_side > 3px)
    num_cc, cc_labels, cc_stats, _ = cv2.connectedComponentsWithStats(thresh_std, connectivity=8)
    rescue_mask = np.zeros_like(thresh_std)
    for lbl in range(1, num_cc):
        comp_mask = (cc_labels == lbl).astype(np.uint8) * 255
        # Skip if nothing erased in this component
        if not np.any(cv2.bitwise_and(comp_mask, erase_mask)):
            continue
        # Primary rescue: overlaps MSER char / small islands zone
        if np.any(cv2.bitwise_and(comp_mask, text_protection_mask)):
            rescue_mask = cv2.bitwise_or(rescue_mask, comp_mask)
            continue
            
        bw_c = cc_stats[lbl, cv2.CC_STAT_WIDTH]
        bh_c = cc_stats[lbl, cv2.CC_STAT_HEIGHT]
        long_s  = max(bw_c, bh_c)
        short_s = min(bw_c, bh_c)
        ar_c = long_s / short_s if short_s > 0 else 0
        
        # Rescue if it's too short to be a line (e.g. maximum dimension is small)
        if long_s < 15:
            rescue_mask = cv2.bitwise_or(rescue_mask, comp_mask)
            continue
            
        # Protect if NOT a very elongated thin stroke
        if ar_c < 6 or short_s > 3:
            rescue_mask = cv2.bitwise_or(rescue_mask, comp_mask)
            
    erase_mask = cv2.bitwise_and(erase_mask, cv2.bitwise_not(rescue_mask))
    # ─────────────────────────────────────────────────────────────────────────────


    
    # Apply erasing
    cleaned_thresh = cv2.subtract(thresh_std, erase_mask)
    cleaned_std = cv2.bitwise_not(cleaned_thresh)
    
    # Store indices of erased pixels (in standard crop coordinate space)
    erased_ys, erased_xs = np.where(erase_mask > 0)
    erased_pts = np.column_stack((erased_xs, erased_ys)) if len(erased_xs) > 0 else np.empty((0, 2), dtype=int)
    
    # Build review image (Red = Text, Blue = Lines)
    H, W = thresh_std.shape
    review_img = np.ones((H, W, 3), dtype=np.uint8) * 255
    review_img[cleaned_thresh > 0] = [220, 30, 30]  # Red (RGB)
    review_img[erase_mask > 0] = [30, 80, 220]     # Blue (RGB)
    
    return Image.fromarray(cleaned_std), erased_pts, review_img


# ── Full Page Pipeline ─────────────────────────────────────────────────────────

def evaluate_cleaning(pred_text_mask: np.ndarray, pred_line_mask: np.ndarray, 
                      gt_text_mask: np.ndarray, gt_line_mask: np.ndarray) -> dict:
    p_txt = pred_text_mask > 0
    p_line = pred_line_mask > 0
    gt_txt = gt_text_mask > 0
    gt_line = gt_line_mask > 0
    
    tot_gt_txt = np.sum(gt_txt)
    tpr = np.sum(p_txt & gt_txt) / tot_gt_txt if tot_gt_txt > 0 else 1.0
    
    tot_gt_line = np.sum(gt_line)
    ldr = np.sum(p_line & gt_line) / tot_gt_line if tot_gt_line > 0 else 1.0
    
    fdr = np.sum(p_line & gt_txt) / tot_gt_txt if tot_gt_txt > 0 else 0.0
    llr = np.sum(p_txt & gt_line) / tot_gt_line if tot_gt_line > 0 else 0.0
    
    f1 = 2 * (tpr * ldr) / (tpr + ldr) if (tpr + ldr) > 0 else 0.0
    
    return {
        "tpr": float(tpr),
        "ldr": float(ldr),
        "fdr": float(fdr),
        "llr": float(llr),
        "f1": float(f1)
    }


def process_and_clean_page(img_path, yolo_model, output_path, conf_thresh=0.25, gt_path=None):
    """Runs YOLO, cleans crossing CAD lines from detected expressions, saves outputs & evaluations."""
    print(f"Loading page image: {img_path}")
    orig_img = Image.open(img_path)
    page_w, page_h = orig_img.size
    
    gt_colored_pil = None
    if gt_path:
        if os.path.exists(gt_path):
            print(f"Loading ground truth colored page: {gt_path}")
            gt_colored_pil = Image.open(gt_path)
        else:
            print(f"WARNING: Ground truth path not found: {gt_path}")
            
    # Preprocess image to grayscale and clean adaptive background lines first
    print("Preprocessing page image...")
    img_gray = cv2.cvtColor(np.array(orig_img.convert("RGB")), cv2.COLOR_RGB2GRAY)
    
    # ── MSER Character & Line Mask Generation for the page ──
    print("Generating MSER character and line masks of the page...")
    mser = cv2.MSER_create(5, 30, 15000)
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
        
        # Characters: aspect ratio <= 1.5, size within [5, 80]
        if 5 <= L1 <= 80 and 5 <= L2 <= 80 and aspect_ratio <= 1.5:
            cv2.drawContours(mser_char_mask, [box_pts], 0, 255, -1)
            max_mser_char_dim = max(max_mser_char_dim, L_max)
            
        # Lines: aspect ratio > 1.5, thickness <= 80, length >= 5
        elif L_min <= 80 and L_max >= 5 and aspect_ratio > 1.5:
            cv2.drawContours(mser_line_mask, [box_pts], 0, 255, -1)
            
    if max_mser_char_dim == 0.0:
        max_mser_char_dim = 30.0
    print(f"Max MSER character dimension on page: {max_mser_char_dim:.1f} pixels")
    
    mser_char_page_pil = Image.fromarray(mser_char_mask)
    mser_line_page_pil = Image.fromarray(mser_line_mask)
    
    # Apply adaptive threshold
    _, thresh_page = cv2.threshold(img_gray, 180, 255, cv2.THRESH_BINARY)
    preprocessed_img = Image.fromarray(cv2.cvtColor(thresh_page, cv2.COLOR_GRAY2RGB))
    
    # ── Run the Small Island Tool on the whole page ──
    print("Running Small Island Tool on the whole page...")
    thresh_page_inv = cv2.bitwise_not(thresh_page)
    num_labels_page, labels_im_page, stats_page, _ = cv2.connectedComponentsWithStats(thresh_page_inv, connectivity=8)
    
    small_islands_page = np.zeros_like(thresh_page_inv)
    large_islands_page = np.zeros_like(thresh_page_inv)
    
    for label in range(1, num_labels_page):
        w_c = stats_page[label, cv2.CC_STAT_WIDTH]
        h_c = stats_page[label, cv2.CC_STAT_HEIGHT]
        area_c = stats_page[label, cv2.CC_STAT_AREA]
        
        comp_mask = (labels_im_page == label).astype(np.uint8) * 255
        
        if w_c <= 50 and h_c <= 50 and area_c <= 350:
            small_islands_page = cv2.bitwise_or(small_islands_page, comp_mask)
        else:
            large_islands_page = cv2.bitwise_or(large_islands_page, comp_mask)
            
    small_islands_page_pil = Image.fromarray(small_islands_page)
    large_islands_page_pil = Image.fromarray(large_islands_page)
    
    # Run YOLOv8-OBB model
    print("Running YOLOv8-OBB inference...")
    img_bgr = cv2.cvtColor(np.array(preprocessed_img), cv2.COLOR_RGB2BGR)
    results = yolo_model(img_bgr, verbose=False, conf=conf_thresh, imgsz=1280)
    result = results[0]
    
    cleaned_page_np = np.array(preprocessed_img)
    
    # Output directories for crop comparisons
    output_dir = os.path.dirname(output_path)
    crops_dir = os.path.join(output_dir, "crops")
    os.makedirs(crops_dir, exist_ok=True)
    
    eval_results = []
    
    if result.obb is not None and len(result.obb) > 0:
        xywhr = result.obb.xywhr.cpu().numpy()  # cx, cy, w, h, r
        print(f"Found {len(xywhr)} bounding boxes of expressions.")
        
        for idx in range(len(xywhr)):
            cx, cy, w, h, r = xywhr[idx]
            angle_deg = math.degrees(r)
            
            # Standard crop (20% buffer)
            crop_std = rectify_crop(
                preprocessed_img,
                bbox_metrics={'cx': cx, 'cy': cy, 'w': w, 'h': h, 'angle': angle_deg},
                buffer_percent=0.20
            )
            
            # Big crop (40% buffer)
            crop_big = rectify_crop(
                preprocessed_img,
                bbox_metrics={'cx': cx, 'cy': cy, 'w': w, 'h': h, 'angle': angle_deg},
                buffer_percent=0.40
            )
            # Crop the MSER character mask crop (20% buffer)
            crop_mser_char = rectify_crop(
                mser_char_page_pil,
                bbox_metrics={'cx': cx, 'cy': cy, 'w': w, 'h': h, 'angle': angle_deg},
                buffer_percent=0.20
            )
            # Crop the MSER line mask crop (20% buffer)
            crop_mser_line = rectify_crop(
                mser_line_page_pil,
                bbox_metrics={'cx': cx, 'cy': cy, 'w': w, 'h': h, 'angle': angle_deg},
                buffer_percent=0.20
            )
            # Crop the page-level small islands mask (20% buffer)
            crop_small_islands = rectify_crop(
                small_islands_page_pil,
                bbox_metrics={'cx': cx, 'cy': cy, 'w': w, 'h': h, 'angle': angle_deg},
                buffer_percent=0.20
            )
            # Crop the page-level large islands mask (20% buffer)
            crop_large_islands = rectify_crop(
                large_islands_page_pil,
                bbox_metrics={'cx': cx, 'cy': cy, 'w': w, 'h': h, 'angle': angle_deg},
                buffer_percent=0.20
            )
            
            # Clean standard crop and extract erased coordinate indices using MSER and Small Island protection
            cleaned_crop, erased_pts_crop, review_img = clean_crop_lines(
                crop_std, crop_big, crop_mser_char, crop_mser_line, crop_small_islands, crop_large_islands, max_mser_char_dim
            )
            
            # Save crops side-by-side: original | MSER char mask | Small Islands mask | review image | cleaned crop
            orig_std_np = np.array(crop_std.convert("RGB"))
            cleaned_np = np.array(cleaned_crop.convert("RGB"))
            
            # Verify and match sizes for horizontal stacking
            H, W = orig_std_np.shape[:2]
            review_resized = cv2.resize(review_img, (W, H))
            
            # Convert binary MSER character mask to RGB and resize
            mser_char_rgb = cv2.cvtColor(np.array(crop_mser_char.convert("L")), cv2.COLOR_GRAY2RGB)
            mser_char_resized = cv2.resize(mser_char_rgb, (W, H))
            
            # Convert binary Small Islands mask to RGB and resize for visualizer
            small_islands_rgb = cv2.cvtColor(np.array(crop_small_islands.convert("L")), cv2.COLOR_GRAY2RGB)
            small_islands_resized = cv2.resize(small_islands_rgb, (W, H))
            
            # Separator lines
            separator = np.ones((H, 4, 3), dtype=np.uint8) * 180
            
            comparison_grid = np.hstack((
                orig_std_np, separator, 
                mser_char_resized, separator, 
                small_islands_resized, separator, 
                review_resized, separator, 
                cleaned_np
            ))
            
            comparison_path = os.path.join(crops_dir, f"crop_{idx}_comparison.png")
            Image.fromarray(comparison_grid).save(comparison_path)
            
            evaluation_metrics = None
            if gt_colored_pil is not None:
                try:
                    # Rectify the colored ground truth crop (matching 20% buffer)
                    crop_pil_gt_col = rectify_crop(
                        gt_colored_pil,
                        bbox_metrics={'cx': cx, 'cy': cy, 'w': w, 'h': h, 'angle': angle_deg},
                        buffer_percent=0.20
                    )
                    
                    # Convert to RGB array
                    gt_rgb = np.array(crop_pil_gt_col.convert("RGB"))
                    
                    # Extract ground truth masks
                    # Pure Red for Text: R > 200, G < 100, B < 100
                    gt_text_mask = ((gt_rgb[:, :, 0] > 200) & (gt_rgb[:, :, 1] < 100) & (gt_rgb[:, :, 2] < 100)).astype(np.uint8) * 255
                    # Pure Blue for Line: R < 100, G < 100, B > 200
                    gt_line_mask = ((gt_rgb[:, :, 0] < 100) & (gt_rgb[:, :, 1] < 100) & (gt_rgb[:, :, 2] > 200)).astype(np.uint8) * 255
                    
                    # Formulate prediction masks
                    cleaned_l = np.array(cleaned_crop.convert("L"))
                    _, pred_text_mask = cv2.threshold(cleaned_l, 127, 255, cv2.THRESH_BINARY_INV)
                    
                    orig_l = np.array(crop_std.convert("L"))
                    _, orig_thresh = cv2.threshold(orig_l, 127, 255, cv2.THRESH_BINARY_INV)
                    
                    # Intersect ground truth masks with orig_thresh to only evaluate on pixels present in the crop
                    gt_text_mask_clean = cv2.bitwise_and(gt_text_mask, orig_thresh)
                    gt_line_mask_clean = cv2.bitwise_and(gt_line_mask, orig_thresh)
                    
                    pred_line_mask = cv2.subtract(orig_thresh, pred_text_mask)
                    
                    # Run evaluate_cleaning
                    evaluation_metrics = evaluate_cleaning(
                        pred_text_mask, pred_line_mask, gt_text_mask_clean, gt_line_mask_clean
                    )
                    eval_results.append(evaluation_metrics)
                except Exception as e:
                    print(f"Warning: Failed to evaluate crop {idx} due to error: {e}")
            
            if len(erased_pts_crop) > 0:
                # Map standard crop erased pixels back to the original page coordinates
                pts_page = map_crop_to_page_coordinates(
                    erased_pts_crop, cx, cy, w, h, angle_deg, buffer_percent=0.20
                )
                
                # Erase these pixels on the full page image (set to white/255)
                for pt in pts_page:
                    px_x = int(round(pt[0]))
                    px_y = int(round(pt[1]))
                    if 0 <= px_x < page_w and 0 <= px_y < page_h:
                        cleaned_page_np[px_y, px_x] = [255, 255, 255]  # Paint white
            
            log_str = f"  Processed box {idx + 1}/{len(xywhr)}: erased {len(erased_pts_crop)} pixels."
            if evaluation_metrics:
                log_str += f" | TPR: {evaluation_metrics['tpr']*100:.1f}% | LDR: {evaluation_metrics['ldr']*100:.1f}%"
            print(log_str)
    else:
        print("No bounding boxes of expressions detected on the page.")
        
    # Save the cleaned page
    cleaned_page_img = Image.fromarray(cleaned_page_np)
    cleaned_page_img.save(output_path)
    print(f"Saved cleaned page image to: {output_path}")
    print(f"Saved crop comparison visualizer grids to: {crops_dir}")
    
    # Calculate page-level averages of evaluation metrics
    if eval_results:
        avg_tpr = np.mean([e["tpr"] for e in eval_results])
        avg_ldr = np.mean([e["ldr"] for e in eval_results])
        avg_f1 = np.mean([e["f1"] for e in eval_results])
        print("\n" + "="*70)
        print("PAGE-LEVEL EVALUATION REPORT")
        print("="*70)
        print(f"{'Crop Index':<12} | {'TPR (Text kept)':<16} | {'LDR (Lines removed)':<18} | {'F1 Score':<10}")
        print("-" * 70)
        for i, r in enumerate(eval_results):
            print(f"{i:<12} | {r['tpr']*100:>14.1f}% | {r['ldr']*100:>16.1f}% | {r['f1']*100:>8.1f}%")
        print("-" * 70)
        print(f"{'AVERAGES':<12} | {avg_tpr*100:>14.1f}% | {avg_ldr*100:>16.1f}% | {avg_f1*100:>8.1f}%")
        print("="*70 + "\n")


def main():
    parser = argparse.ArgumentParser(description="Clean CAD lines from expression bounding boxes on a page.")
    parser.add_argument("--input", default="d:/Internship/OCR_PDF/internt-ocrmodel/eval_output/eval_page.png", help="Path to input page image")
    parser.add_argument("--yolo", default="D:/Internship/OCR_PDF/YOLO_expression_best.pt", help="Path to trained YOLOv8-OBB weights")
    parser.add_argument("--gt", default="d:/Internship/OCR_PDF/internt-ocrmodel/eval_output/eval_page_gt_colored.png", help="Path to ground truth colored page image")
    parser.add_argument("--output", default="d:/Internship/OCR_PDF/BoudningBoxCleaning/cleaned_page.png", help="Path to save cleaned output page image")
    parser.add_argument("--conf", type=float, default=0.25, help="YOLO confidence threshold")
    args = parser.parse_args()
    
    if not os.path.exists(args.input):
        print(f"Error: Input file not found: {args.input}")
        sys.exit(1)
        
    if not os.path.exists(args.yolo):
        print(f"Error: YOLO weights not found: {args.yolo}")
        sys.exit(1)
        
    print(f"Loading YOLOv8 model from: {args.yolo}...")
    yolo_model = YOLO(args.yolo)
    
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    process_and_clean_page(args.input, yolo_model, args.output, args.conf, args.gt)
    print("Done!")


if __name__ == "__main__":
    main()
