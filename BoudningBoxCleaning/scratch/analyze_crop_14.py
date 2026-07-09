import cv2
import math
import numpy as np
import os
from ultralytics import YOLO

# Reconfigure stdout for Windows unicode support
import sys
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

# Copy the core functions from clean_page_expressions.py for local debugging
sys.path.append("d:/Internship/OCR_PDF/BoudningBoxCleaning")
from clean_page_expressions import (
    rectify_crop,
    skeletonize,
    get_skeleton_junctions,
    split_skeleton_into_branches,
    classify_branch_shape,
    evaluate_path_fitness,
    get_straight_line_path,
    get_circle_path,
    find_border_crossing_points
)

def debug_crop_14():
    model_path = "D:/Internship/OCR_PDF/YOLO_expression_best.pt"
    model = YOLO(model_path)
    
    inp = "d:/Internship/OCR_PDF/sign-detection-yolo8n/toTest/2.jpeg"
    img = cv2.imread(inp)
    H, W, C = img.shape
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(img_gray, 180, 255, cv2.THRESH_BINARY)
    img_rgb = cv2.cvtColor(thresh, cv2.COLOR_GRAY2RGB)
    
    results = model(img_rgb, verbose=False, conf=0.25, imgsz=1280)
    result = results[0]
    xywhr = result.obb.xywhr.cpu().numpy()
    
    # We know it is index 14
    idx = 14
    cx, cy, w, h, r = xywhr[idx]
    angle_deg = math.degrees(r)
    print(f"Debugging Box 14: cx={cx:.1f}, cy={cy:.1f}, w={w:.1f}, h={h:.1f}, angle={angle_deg:.1f}")
    
    from PIL import Image
    preprocessed_img = Image.fromarray(img_rgb)
    crop_std = rectify_crop(
        preprocessed_img,
        bbox_metrics={'cx': cx, 'cy': cy, 'w': w, 'h': h, 'angle': angle_deg},
        buffer_percent=0.20
    )
    crop_big = rectify_crop(
        preprocessed_img,
        bbox_metrics={'cx': cx, 'cy': cy, 'w': w, 'h': h, 'angle': angle_deg},
        buffer_percent=0.40
    )
    
    img_std = np.array(crop_std.convert("L"))
    img_big = np.array(crop_big.convert("L"))
    H_std, W_std = img_std.shape
    H_big, W_big = img_big.shape
    
    _, thresh_std = cv2.threshold(img_std, 180, 255, cv2.THRESH_BINARY_INV)
    _, thresh_big = cv2.threshold(img_big, 180, 255, cv2.THRESH_BINARY_INV)
    
    # Calculate crop offsets inside the big crop
    dx = (W_big - W_std) // 2
    dy = (H_big - H_std) // 2

    # Connected component check on big crop to protect border-touching characters
    dilated_big_check = cv2.dilate(thresh_big, cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3)))
    num_labels_check, labels_big_check, stats_check, _ = cv2.connectedComponentsWithStats(dilated_big_check, connectivity=8)
    
    # Identify which components extend outside the standard crop box in the big crop
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

    # ── Text Protection: Filter border-touching vs non-border-touching components ──
    num_std_labels, std_labels_im, std_stats, _ = cv2.connectedComponentsWithStats(thresh_std, connectivity=8)
    text_protection_mask = np.zeros_like(thresh_std)
    border_touching_mask = np.zeros_like(thresh_std)
    margin = 2
    
    print("\n--- Connected Components in thresh_std ---")
    for label in range(1, num_std_labels):
        comp_mask = std_labels_im == label
        ys, xs = np.where(comp_mask)
        if len(xs) > 0:
            touches_border = (np.any(xs <= margin) | np.any(xs >= W_std - 1 - margin) |
                              np.any(ys <= margin) | np.any(ys >= H_std - 1 - margin))
            if touches_border:
                # Check if it actually extends outside standard crop box in big crop
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
                    print(f"Component {label}: size={len(xs)} pixels, box=[x: {np.min(xs)}-{np.max(xs)}, y: {np.min(ys)}-{np.max(ys)}], touches_border=True, EXTENDS_OUTSIDE=True (Not Protected)")
                else:
                    text_protection_mask[comp_mask] = 255
                    print(f"Component {label}: size={len(xs)} pixels, box=[x: {np.min(xs)}-{np.max(xs)}, y: {np.min(ys)}-{np.max(ys)}], touches_border=True, EXTENDS_OUTSIDE=False (PROTECTED)")
            else:
                text_protection_mask[comp_mask] = 255
                print(f"Component {label}: size={len(xs)} pixels, box=[x: {np.min(xs)}-{np.max(xs)}, y: {np.min(ys)}-{np.max(ys)}], touches_border=False (PROTECTED)")
                
    # Distance transform to nearest foreground pixel (black in original)
    dist_to_fg = cv2.distanceTransform(cv2.bitwise_not(thresh_std), cv2.DIST_L2, 5)
    thickness_map = 2 * cv2.distanceTransform(thresh_std, cv2.DIST_L2, 5)
    
    skel = skeletonize(thresh_std)
    junctions = get_skeleton_junctions(skel)
    protected_junction_zone = cv2.dilate(junctions, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7)))
    
    crossing_pts = find_border_crossing_points(thresh_std)
    print(f"Crossing points: {crossing_pts}")
    
    # ── Solid & Leader Line Removal (Branch Classification) ──
    branches, _ = split_skeleton_into_branches(skel)
    branch_erase_mask = np.zeros_like(thresh_std)
    
    border_margin = 3
    print("\n--- Branch Classification ---")
    for b_idx, branch_pts in enumerate(branches):
        branch_thickness = 2 * np.mean([dist_to_fg[pt[1], pt[0]] for pt in branch_pts])
        shape_type, score = classify_branch_shape(branch_pts, thresh_px=1.5)
        touches_border = np.any((branch_pts[:, 0] < border_margin) | 
                                (branch_pts[:, 0] > W_std - 1 - border_margin) | 
                                (branch_pts[:, 1] < border_margin) | 
                                (branch_pts[:, 1] > H_std - 1 - border_margin))
        
        # Print branch info
        # Find if this branch belongs to the number 3
        # In crop 14, where is the number 3? It's on the left side of the crop
        min_x = np.min(branch_pts[:, 0])
        max_x = np.max(branch_pts[:, 0])
        min_y = np.min(branch_pts[:, 1])
        max_y = np.max(branch_pts[:, 1])
        print(f"Branch {b_idx}: length={len(branch_pts)}, shape_type={shape_type}, thickness={branch_thickness:.2f}, touches_border={touches_border}, x_range=[{min_x}, {max_x}], y_range=[{min_y}, {max_y}]")
        
        if branch_thickness >= 2.6:
            continue
        if shape_type not in ["straight", "curved"]:
            continue
            
        if touches_border:
            branch_mask = np.zeros_like(thresh_std)
            for pt in branch_pts:
                cv2.circle(branch_mask, (pt[0], pt[1]), 2, 255, -1)
            branch_erase = cv2.bitwise_and(thresh_std, branch_mask)
            branch_erase_mask = cv2.bitwise_or(branch_erase_mask, branch_erase)
            
    # Save intermediate images
    out_dir = "d:/Internship/OCR_PDF/BoudningBoxCleaning/scratch"
    cv2.imwrite(os.path.join(out_dir, "crop_14_thresh_std.png"), thresh_std)
    cv2.imwrite(os.path.join(out_dir, "crop_14_text_protection.png"), text_protection_mask)
    cv2.imwrite(os.path.join(out_dir, "crop_14_branch_erase_mask.png"), branch_erase_mask)
    print("\nIntermediate debug images saved.")

if __name__ == "__main__":
    debug_crop_14()
