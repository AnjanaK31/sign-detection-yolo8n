import sys
import os
import json
import math
import cv2
import numpy as np
from PIL import Image

sys.path.append(os.path.abspath(r"d:\Internship\OCR_PDF\BoudningBoxCleaning"))
from clean_page_expressions import clean_crop_lines, rectify_crop

def main():
    img_path = r"d:\Internship\OCR_PDF\INTRA_cleaning\intra.png"
    label_path = r"d:\Internship\OCR_PDF\INTRA_cleaning\Label.txt"
    output_dir = r"d:\Internship\OCR_PDF\INTRA_cleaning\multistage_processing"
    
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"Loading page image: {img_path}")
    if not os.path.exists(img_path):
        print(f"ERROR: Image file not found: {img_path}")
        return
        
    orig_img = Image.open(img_path)
    page_w, page_h = orig_img.size
    
    print("Preprocessing image and generating page-level masks...")
    img_gray = cv2.cvtColor(np.array(orig_img.convert("RGB")), cv2.COLOR_RGB2GRAY)
    
    # MSER Character & Line Mask Generation for the page
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
    print(f"  Max MSER character dimension: {max_mser_char_dim:.1f} px")
    
    mser_char_page_pil = Image.fromarray(mser_char_mask)
    mser_line_page_pil = Image.fromarray(mser_line_mask)
    
    # Apply threshold for algorithm input (binary required by clean_crop_lines)
    _, thresh_page = cv2.threshold(img_gray, 180, 255, cv2.THRESH_BINARY)
    preprocessed_img = Image.fromarray(cv2.cvtColor(thresh_page, cv2.COLOR_GRAY2RGB))
    
    # ─── NEW: Keep original color image for display purposes ───────────────────
    orig_color_pil = orig_img.convert("RGB")
    # ──────────────────────────────────────────────────────────────────────────
    
    # Small / Large Island Tool on page level
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
    
    # Read Label.txt coordinates for intra.png
    if not os.path.exists(label_path):
        print(f"ERROR: Label file not found: {label_path}")
        return
        
    intra_boxes = None
    with open(label_path, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split("\t")
            if len(parts) >= 2 and "intra.png" in parts[0]:
                intra_boxes = json.loads(parts[1])
                break
                
    if intra_boxes is None:
        print("ERROR: Did not find entry for intra.png in Label.txt")
        return
        
    print(f"Processing {len(intra_boxes)} bounding boxes...")
    for idx, box in enumerate(intra_boxes):
        transcription = box.get("transcription", "")
        points = box.get("points", [])
        
        if len(points) != 4:
            continue
            
        # Extract center, size, and angle for OBB rectification
        pts_np = np.array(points, dtype=np.float32)
        rect = cv2.minAreaRect(pts_np)
        cx, cy = rect[0]
        w_box, h_box = rect[1]
        angle = rect[2]
        
        bbox_metrics = {'cx': cx, 'cy': cy, 'w': w_box, 'h': h_box, 'angle': angle}
        
        # Standard crop from BINARY image (used by the cleaner algorithm)
        crop_std = rectify_crop(
            preprocessed_img,
            bbox_metrics=bbox_metrics,
            buffer_percent=0.20
        )
        # Big crop from BINARY image (used by the cleaner algorithm)
        crop_big = rectify_crop(
            preprocessed_img,
            bbox_metrics=bbox_metrics,
            buffer_percent=0.40
        )
        # ─── NEW: Standard crop from ORIGINAL COLOR image (for display only) ──
        crop_std_color = rectify_crop(
            orig_color_pil,
            bbox_metrics=bbox_metrics,
            buffer_percent=0.20
        )
        # ────────────────────────────────────────────────────────────────────────
        # MSER char mask crop
        crop_mser_char = rectify_crop(
            mser_char_page_pil,
            bbox_metrics=bbox_metrics,
            buffer_percent=0.20
        )
        # MSER line mask crop
        crop_mser_line = rectify_crop(
            mser_line_page_pil,
            bbox_metrics=bbox_metrics,
            buffer_percent=0.20
        )
        # Small islands mask crop
        crop_small_islands = rectify_crop(
            small_islands_page_pil,
            bbox_metrics=bbox_metrics,
            buffer_percent=0.20
        )
        # Large islands mask crop
        crop_large_islands = rectify_crop(
            large_islands_page_pil,
            bbox_metrics=bbox_metrics,
            buffer_percent=0.20
        )
        
        # Run 3-stage cleaner (uses binary crops internally)
        cleaned_crop, erased_pts_crop, review_img = clean_crop_lines(
            crop_std, crop_big, crop_mser_char, crop_mser_line, crop_small_islands, crop_large_islands, max_mser_char_dim
        )
        
        # ─── Reconstruct cleaned image using ORIGINAL COLOR pixels ─────────────
        # The cleaner returns a binary cleaned image. We want to:
        #   - Show original color pixels where content was KEPT
        #   - Show pure white where lines were ERASED
        #
        # cleaned_crop is a PIL image: white bg, black remaining content (binary)
        # We invert it: black = kept content mask, white = erased/empty
        
        orig_color_np = np.array(crop_std_color.convert("RGB"))     # original RGB crop
        cleaned_binary_np = np.array(cleaned_crop.convert("L"))      # 0=black kept, 255=white bg
        H, W = cleaned_binary_np.shape
        orig_color_np_resized = cv2.resize(orig_color_np, (W, H))
        
        # kept_mask: True where we want to keep original pixels (black pixels in cleaned binary = kept content)
        kept_mask = cleaned_binary_np < 128   # True = this is content (text), False = bg/erased
        
        # Start with white background
        cleaned_color_np = np.ones((H, W, 3), dtype=np.uint8) * 255
        # Paint original color pixels where content is kept
        cleaned_color_np[kept_mask] = orig_color_np_resized[kept_mask]
        # ────────────────────────────────────────────────────────────────────────
        
        # Grid: [Original Color] | [MSER Char Mask] | [Small Islands Mask] | [Review] | [Cleaned Color]
        orig_color_for_grid = np.array(crop_std_color.convert("RGB"))
        orig_color_for_grid = cv2.resize(orig_color_for_grid, (W, H))
        
        review_resized = cv2.resize(review_img, (W, H))
        
        # Convert binary masks to RGB for visualization
        mser_char_rgb = cv2.cvtColor(np.array(crop_mser_char.convert("L")), cv2.COLOR_GRAY2RGB)
        mser_char_resized = cv2.resize(mser_char_rgb, (W, H))
        
        small_islands_rgb = cv2.cvtColor(np.array(crop_small_islands.convert("L")), cv2.COLOR_GRAY2RGB)
        small_islands_resized = cv2.resize(small_islands_rgb, (W, H))
        
        # Add column labels at the top (5-column grid)
        label_h = 20
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.35
        col_labels = ["Original", "MSER Char", "Small Islands", "Review (R=text B=line)", "Cleaned"]
        cols = [orig_color_for_grid, mser_char_resized, small_islands_resized, review_resized, cleaned_color_np]
        
        labeled_cols = []
        for col_img, col_label in zip(cols, col_labels):
            label_bar = np.ones((label_h, W, 3), dtype=np.uint8) * 50
            cv2.putText(label_bar, col_label, (2, label_h - 6), font, font_scale, (255, 255, 255), 1, cv2.LINE_AA)
            labeled_cols.append(np.vstack([label_bar, col_img]))
        
        # Separator height must match labeled columns (H + label_h)
        separator = np.ones((H + label_h, 6, 3), dtype=np.uint8) * 180
        
        comparison_grid = np.hstack((
            labeled_cols[0], separator,
            labeled_cols[1], separator,
            labeled_cols[2], separator,
            labeled_cols[3], separator,
            labeled_cols[4]
        ))
        
        # Save comparison grid image
        clean_name = transcription.replace("/", "_").replace(":", "_").replace("°", "deg")
        grid_fname = f"crop_{idx}_{clean_name}_multistage.png"
        grid_path = os.path.join(output_dir, grid_fname)
        
        # Save image (converting RGB to BGR for OpenCV)
        cv2.imwrite(grid_path, cv2.cvtColor(comparison_grid, cv2.COLOR_RGB2BGR))
        print(f"  Saved multistage view for Box {idx} ({transcription}): {grid_fname} | Erased {len(erased_pts_crop)} px")

if __name__ == "__main__":
    main()
