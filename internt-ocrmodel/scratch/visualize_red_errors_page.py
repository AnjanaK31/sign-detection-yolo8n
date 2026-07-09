import os
import sys
import cv2
import numpy as np
import torch
from PIL import Image

sys.path.insert(0, "d:/Internship/OCR_PDF/internt-ocrmodel")

from pipeline import load_yolo_model
from classifier import SymbolClassifier
from rectifier import rectify_crop
import line_cleaner

def generate_red_error_visualizations():
    print("Loading page and models...")
    eval_page_path = "eval_output/eval_page.png"
    eval_page_gt_path = "eval_output/eval_page_gt_colored.png"
    
    if not os.path.exists(eval_page_path) or not os.path.exists(eval_page_gt_path):
        print("ERROR: Evaluation page or GT colored page does not exist. Please run run_page_eval.py first.")
        return
        
    page_img = Image.open(eval_page_path)
    page_colored = Image.open(eval_page_gt_path)
    
    W_page, H_page = page_img.size
    
    # Preprocess page to binary for line cleaner
    from preprocessor import full_preprocess
    preprocessed_page = full_preprocess(page_img)
    
    yolo_model = load_yolo_model("../YOLO_expression_best.pt")
    classifier = SymbolClassifier(model_path="../sign-detection-yolo8n/classifier_best.pt", device="cpu")
    
    # Run YOLO OBB
    img_bgr = cv2.cvtColor(np.array(preprocessed_page), cv2.COLOR_RGB2BGR)
    results = yolo_model(img_bgr, verbose=False, conf=0.25, imgsz=1280)
    result = results[0]
    
    if result.obb is None or len(result.obb) == 0:
        print("No bounding boxes detected by YOLO.")
        return
        
    xywhr = result.obb.xywhr.cpu().numpy()
    xyxyxyxy = result.obb.xyxyxyxy.cpu().numpy()
    
    # Calculate full-page connected components and yolo mask to protect symbol-internal lines
    gray_page_lbls = cv2.cvtColor(np.array(preprocessed_page.convert("RGB")), cv2.COLOR_RGB2GRAY)
    _, thresh_page = cv2.threshold(gray_page_lbls, 127, 255, cv2.THRESH_BINARY_INV)
    num_labels, labels_im, stats, _ = cv2.connectedComponentsWithStats(thresh_page)
    
    protected_labels = set()
    yolo_mask = np.zeros(thresh_page.shape, dtype=np.uint8)
    for corners in xyxyxyxy:
        pts = np.array(corners, dtype=np.int32)
        cv2.fillPoly(yolo_mask, [pts], 255)
        
    font_threshold = 35
    for i in range(1, num_labels):
        w_island = stats[i, cv2.CC_STAT_WIDTH]
        h_island = stats[i, cv2.CC_STAT_HEIGHT]
        
        if w_island > font_threshold or h_island > font_threshold:
            island_mask = (labels_im == i)
            total_pixels = np.sum(island_mask)
            inside_pixels = np.sum(yolo_mask[island_mask] > 0)
            
            if total_pixels > 0 and (inside_pixels / total_pixels) > 0.85:
                protected_labels.add(i)
                
    print(f"Page-level protection: {len(protected_labels)} out of {num_labels - 1} islands protected (contain symbol components).")
    
    # 1. Create a canvas with the original page in RGB
    red_errors_page_np = np.array(page_img.convert("RGB"))
    
    # 2. Create a full-page binary mask to accumulate all errors
    full_page_error_mask = np.zeros((H_page, W_page), dtype=np.uint8)
    
    print(f"Processing {len(xywhr)} detected bounding boxes...")
    
    for idx in range(len(xywhr)):
        cx, cy, w, h, r = xywhr[idx]
        angle_deg = np.degrees(r)
        
        # Exact crop parameters
        buffer_percent = 0.20
        bw = w * (1.0 + buffer_percent)
        bh = h * (1.0 + buffer_percent)
        diag = np.sqrt(bw**2 + bh**2)
        crop_size = int(np.ceil(diag)) + 20
        half_size = crop_size // 2
        
        bx_min = int(round(half_size - bw / 2))
        by_min = int(round(half_size - bh / 2))
        bx_max = bx_min + int(round(bw))
        by_max = by_min + int(round(bh))
        
        # Rectify standard crop from preprocessed page (black text, white bg)
        crop_pil_std = rectify_crop(
            preprocessed_page,
            bbox_metrics={'cx': cx, 'cy': cy, 'w': w, 'h': h, 'angle': angle_deg},
            buffer_percent=buffer_percent
        )
        
        # Rectify big crop for line cleaning
        crop_pil_big = rectify_crop(
            preprocessed_page,
            bbox_metrics={'cx': cx, 'cy': cy, 'w': w, 'h': h, 'angle': angle_deg},
            buffer_percent=0.40
        )
        
        # Clean CAD lines from crop
        bbox_metrics = {'cx': cx, 'cy': cy, 'w': w, 'h': h, 'angle': angle_deg}
        cleaned_crop, review_img, _ = line_cleaner.clean_patch_lines(
            crop_pil_std, crop_pil_big,
            bbox_metrics=bbox_metrics,
            labels_im=labels_im,
            protected_labels=protected_labels
        )
        
        # Rectify GT colored crop
        crop_pil_gt_col = rectify_crop(
            page_colored,
            bbox_metrics={'cx': cx, 'cy': cy, 'w': w, 'h': h, 'angle': angle_deg},
            buffer_percent=buffer_percent
        )
        gt_rgb = np.array(crop_pil_gt_col.convert("RGB"))
        
        # Extract GT masks
        gt_text_mask = ((gt_rgb[:, :, 0] > 200) & (gt_rgb[:, :, 1] < 100) & (gt_rgb[:, :, 2] < 100)).astype(np.uint8) * 255
        gt_line_mask = ((gt_rgb[:, :, 0] < 100) & (gt_rgb[:, :, 1] < 100) & (gt_rgb[:, :, 2] > 200)).astype(np.uint8) * 255
        
        # Standard crop thresholded masks
        orig_np = np.array(crop_pil_std.convert("L"))
        _, orig_thresh = cv2.threshold(orig_np, 127, 255, cv2.THRESH_BINARY_INV)
        
        cleaned_np = np.array(cleaned_crop.convert("L"))
        _, pred_text_mask = cv2.threshold(cleaned_np, 127, 255, cv2.THRESH_BINARY_INV)
        
        # Intersect with orig_thresh to only evaluate on pixels present
        gt_text_mask_clean = cv2.bitwise_and(gt_text_mask, orig_thresh)
        gt_line_mask_clean = cv2.bitwise_and(gt_line_mask, orig_thresh)
        
        # Compute error masks
        # Over-deleted text (erased in prediction but present in GT)
        over_deleted = cv2.bitwise_and(gt_text_mask_clean, cv2.bitwise_not(pred_text_mask))
        # Under-deleted lines (kept in prediction but present in GT line)
        under_deleted = cv2.bitwise_and(gt_line_mask_clean, pred_text_mask)
        
        # Combine both errors into a single crop error mask
        crop_error_mask = cv2.bitwise_or(over_deleted, under_deleted)
        
        # Rotate and warp error mask back to page coordinates
        padded_err_mask = np.zeros((crop_size, crop_size), dtype=np.uint8)
        padded_err_mask[by_min:by_max, bx_min:bx_max] = crop_error_mask
        
        padded_mask = np.zeros((crop_size, crop_size), dtype=np.uint8)
        padded_mask[by_min:by_max, bx_min:bx_max] = 255
        
        M_inv = cv2.getRotationMatrix2D((half_size, half_size), -angle_deg, 1.0)
        
        warped_err_mask = cv2.warpAffine(padded_err_mask, M_inv, (crop_size, crop_size), borderValue=0)
        warped_mask = cv2.warpAffine(padded_mask, M_inv, (crop_size, crop_size), borderValue=0)
        
        # Pasting slices
        x_min = int(round(cx - half_size))
        y_min = int(round(cy - half_size))
        
        pad_top = max(0, -y_min)
        pad_left = max(0, -x_min)
        pad_bottom = max(0, (y_min + crop_size) - H_page)
        pad_right = max(0, (x_min + crop_size) - W_page)
        
        w_y1, w_y2 = pad_top, crop_size - pad_bottom
        w_x1, w_x2 = pad_left, crop_size - pad_right
        
        p_y1, p_y2 = max(0, y_min), min(H_page, y_min + crop_size)
        p_x1, p_x2 = max(0, x_min), min(W_page, x_min + crop_size)
        
        crop_mask_bin = warped_mask[w_y1:w_y2, w_x1:w_x2] > 0
        valid_err_mask = (warped_err_mask[w_y1:w_y2, w_x1:w_x2] > 0) & crop_mask_bin
        
        # Accumulate errors onto full page mask
        full_page_error_mask[p_y1:p_y2, p_x1:p_x2][valid_err_mask] = 255

    # 3. Find contours around all error clusters on the full page
    contours, _ = cv2.findContours(full_page_error_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Draw contours in thick RED lines on the original page image
    cv2.drawContours(red_errors_page_np, contours, -1, (255, 0, 0), 2)
    
    # 4. Save the final image
    output_path = "eval_output/eval_page_red_errors_full.png"
    Image.fromarray(red_errors_page_np).save(output_path)
    
    print(f"Saved full page contour error visualization to: {output_path}")
    print("Red error visualization generation complete!")

if __name__ == "__main__":
    generate_red_error_visualizations()
