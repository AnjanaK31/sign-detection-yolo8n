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
    pdf_rendered_path = r"d:\Internship\OCR_PDF\sign-detection-yolo8n\toTest\CLEANINTEST\intra_pdf_rendered.png"
    label_path = r"d:\Internship\OCR_PDF\sign-detection-yolo8n\toTest\CLEANINTEST\Label.txt"
    output_dir = r"d:\Internship\OCR_PDF\sign-detection-yolo8n\toTest\CLEANINTEST\intra_pdf_multistage"
    
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"Loading rendered PDF page image: {pdf_rendered_path}")
    if not os.path.exists(pdf_rendered_path):
        print(f"ERROR: Image file not found: {pdf_rendered_path}")
        return
        
    orig_img = Image.open(pdf_rendered_path)
    page_w, page_h = orig_img.size
    
    # 1. Preprocess image to grayscale and build page-level masks
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
    
    # Apply adaptive threshold (to align with clean_page_expressions pipeline)
    _, thresh_page = cv2.threshold(img_gray, 180, 255, cv2.THRESH_BINARY)
    preprocessed_img = Image.fromarray(cv2.cvtColor(thresh_page, cv2.COLOR_GRAY2RGB))
    
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
        
        # Standard crop (20% buffer)
        crop_std = rectify_crop(
            preprocessed_img,
            bbox_metrics={'cx': cx, 'cy': cy, 'w': w_box, 'h': h_box, 'angle': angle},
            buffer_percent=0.20
        )
        # Big crop (40% buffer)
        crop_big = rectify_crop(
            preprocessed_img,
            bbox_metrics={'cx': cx, 'cy': cy, 'w': w_box, 'h': h_box, 'angle': angle},
            buffer_percent=0.40
        )
        # MSER char mask crop
        crop_mser_char = rectify_crop(
            mser_char_page_pil,
            bbox_metrics={'cx': cx, 'cy': cy, 'w': w_box, 'h': h_box, 'angle': angle},
            buffer_percent=0.20
        )
        # MSER line mask crop
        crop_mser_line = rectify_crop(
            mser_line_page_pil,
            bbox_metrics={'cx': cx, 'cy': cy, 'w': w_box, 'h': h_box, 'angle': angle},
            buffer_percent=0.20
        )
        # Small islands mask crop
        crop_small_islands = rectify_crop(
            small_islands_page_pil,
            bbox_metrics={'cx': cx, 'cy': cy, 'w': w_box, 'h': h_box, 'angle': angle},
            buffer_percent=0.20
        )
        # Large islands mask crop
        crop_large_islands = rectify_crop(
            large_islands_page_pil,
            bbox_metrics={'cx': cx, 'cy': cy, 'w': w_box, 'h': h_box, 'angle': angle},
            buffer_percent=0.20
        )
        
        # Run 3-stage cleaner
        cleaned_crop, erased_pts_crop, review_img = clean_crop_lines(
            crop_std, crop_big, crop_mser_char, crop_mser_line, crop_small_islands, crop_large_islands, max_mser_char_dim
        )
        
        # Create horizontal comparison grid:
        # Original (std) | MSER Character Mask | Small Islands Mask | Review | Cleaned
        orig_std_np = np.array(crop_std.convert("RGB"))
        cleaned_np = np.array(cleaned_crop.convert("RGB"))
        H, W = orig_std_np.shape[:2]
        
        review_resized = cv2.resize(review_img, (W, H))
        
        # Convert binary masks to RGB for visualization
        mser_char_rgb = cv2.cvtColor(np.array(crop_mser_char.convert("L")), cv2.COLOR_GRAY2RGB)
        mser_char_resized = cv2.resize(mser_char_rgb, (W, H))
        
        small_islands_rgb = cv2.cvtColor(np.array(crop_small_islands.convert("L")), cv2.COLOR_GRAY2RGB)
        small_islands_resized = cv2.resize(small_islands_rgb, (W, H))
        
        # Separator line
        separator = np.ones((H, 6, 3), dtype=np.uint8) * 180
        
        comparison_grid = np.hstack((
            orig_std_np, separator,
            mser_char_resized, separator,
            small_islands_resized, separator,
            review_resized, separator,
            cleaned_np
        ))
        
        # Save comparison grid image
        clean_name = transcription.replace("/", "_").replace(":", "_").replace("°", "deg")
        grid_fname = f"crop_{idx}_{clean_name}_multistage.png"
        grid_path = os.path.join(output_dir, grid_fname)
        
        # Save image (converting RGB to BGR)
        cv2.imwrite(grid_path, cv2.cvtColor(comparison_grid, cv2.COLOR_RGB2BGR))
        print(f"  Saved multistage view for Box {idx} ({transcription}): {grid_fname} | Erased {len(erased_pts_crop)} px")

if __name__ == "__main__":
    main()
