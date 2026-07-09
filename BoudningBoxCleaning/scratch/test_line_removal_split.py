import cv2
import numpy as np
import os
import glob

images_path = r"d:\Internship\OCR_PDF\sign-detection-yolo8n\toTest\CLEANINTEST\*"
images = glob.glob(images_path)

output_dir = r"d:\Internship\OCR_PDF\TESTIMAGES_SEGMENTED_DEBUG"
os.makedirs(output_dir, exist_ok=True)

def get_foreground_mask(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    corners = [gray[0,0], gray[0,-1], gray[-1,0], gray[-1,-1]]
    avg_corner = np.mean(corners)
    is_white_bg = avg_corner > 127
    
    if is_white_bg:
        _, thresh = cv2.threshold(gray, 240, 255, cv2.THRESH_BINARY_INV)
    else:
        _, thresh = cv2.threshold(gray, 15, 255, cv2.THRESH_BINARY)
    return thresh, is_white_bg

def remove_grid_lines(thresh):
    # Create horizontal and vertical kernels to detect lines
    h, w = thresh.shape
    
    # Horizontal lines
    horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (int(w * 0.1), 1))
    detected_horizontal = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, horizontal_kernel)
    
    # Vertical lines
    vertical_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, int(h * 0.1)))
    detected_vertical = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, vertical_kernel)
    
    # Combined lines mask
    lines_mask = cv2.bitwise_or(detected_horizontal, detected_vertical)
    
    # Subtract lines from threshold
    cleaned_thresh = cv2.subtract(thresh, lines_mask)
    return cleaned_thresh, lines_mask

for img_path in images:
    if not img_path.lower().endswith(('.png', '.jpg', '.jpeg')):
        continue
    print(f"\nProcessing {os.path.basename(img_path)}...")
    img = cv2.imread(img_path)
    if img is None:
        continue
    h, w = img.shape[:2]
    thresh, is_white_bg = get_foreground_mask(img)
    
    cleaned_thresh, lines_mask = remove_grid_lines(thresh)
    
    # Let's count row and col projections on cleaned_thresh
    row_proj = np.sum(cleaned_thresh > 0, axis=1)
    col_proj = np.sum(cleaned_thresh > 0, axis=0)
    
    zero_rows = np.sum(row_proj == 0)
    zero_cols = np.sum(col_proj == 0)
    print(f"  Cleaned - Zero rows: {zero_rows}/{h} ({zero_rows/h*100:.1f}%), Zero cols: {zero_cols}/{w} ({zero_cols/w*100:.1f}%)")
    
    # Now let's try our recursive split algorithm on cleaned_thresh
    def split_bounding_box(x1, y1, x2, y2, mask):
        sub_mask = mask[y1:y2, x1:x2]
        sh, sw = sub_mask.shape
        if sh <= 10 or sw <= 10:
            return []
            
        sub_row_proj = np.sum(sub_mask > 0, axis=1)
        sub_col_proj = np.sum(sub_mask > 0, axis=0)
        
        # Look for a vertical split (along columns) or horizontal split (along rows)
        # We look for a line of pure background.
        # Let's check vertical splits first to split side-by-side drawings
        zero_cols = np.where(sub_col_proj == 0)[0]
        if len(zero_cols) > 0:
            # Group into runs
            runs = []
            start = zero_cols[0]
            for i in range(1, len(zero_cols)):
                if zero_cols[i] != zero_cols[i-1] + 1:
                    runs.append((start, zero_cols[i-1]))
                    start = zero_cols[i]
            runs.append((start, zero_cols[-1]))
            
            # Find runs of width >= 10 pixels
            valid_runs = [r for r in runs if (r[1] - r[0] + 1) >= 10]
            if len(valid_runs) > 0:
                # We will split at the midpoint of each valid run
                splits = [0]
                for r_start, r_end in valid_runs:
                    splits.append((r_start + r_end) // 2)
                splits.append(sw)
                
                # Recursively split each slice
                results = []
                for i in range(len(splits) - 1):
                    s_x1 = splits[i]
                    s_x2 = splits[i+1]
                    if s_x2 - s_x1 > 10:
                        results.extend(split_bounding_box(x1 + s_x1, y1, x1 + s_x2, y2, mask))
                return results
                
        # If no vertical splits, check horizontal splits
        zero_rows = np.where(sub_row_proj == 0)[0]
        if len(zero_rows) > 0:
            runs = []
            start = zero_rows[0]
            for i in range(1, len(zero_rows)):
                if zero_rows[i] != zero_rows[i-1] + 1:
                    runs.append((start, zero_rows[i-1]))
                    start = zero_rows[i]
            runs.append((start, zero_rows[-1]))
            
            valid_runs = [r for r in runs if (r[1] - r[0] + 1) >= 10]
            if len(valid_runs) > 0:
                splits = [0]
                for r_start, r_end in valid_runs:
                    splits.append((r_start + r_end) // 2)
                splits.append(sh)
                
                results = []
                for i in range(len(splits) - 1):
                    s_y1 = splits[i]
                    s_y2 = splits[i+1]
                    if s_y2 - s_y1 > 10:
                        results.extend(split_bounding_box(x1, y1 + s_y1, x2, y1 + s_y2, mask))
                return results
                
        # If no more splits are possible, crop to foreground bounding box
        # only if there are enough foreground pixels
        if np.sum(sub_mask > 0) > 100:
            y_indices, x_indices = np.where(sub_mask > 0)
            min_y, max_y = np.min(y_indices), np.max(y_indices)
            min_x, max_x = np.min(x_indices), np.max(x_indices)
            
            # Add a small margin/padding around the cropped region
            padding = 15
            crop_y1 = max(0, y1 + min_y - padding)
            crop_y2 = min(h, y1 + max_y + padding)
            crop_x1 = max(0, x1 + min_x - padding)
            crop_x2 = min(w, x1 + max_x + padding)
            return [(crop_x1, crop_y1, crop_x2, crop_y2)]
        return []

    # Get segmented bounding boxes
    boxes = split_bounding_box(0, 0, w, h, cleaned_thresh)
    print(f"  Found {len(boxes)} segmented regions.")
    for idx, (bx1, by1, bx2, by2) in enumerate(boxes):
        bw, bh = bx2 - bx1, by2 - by1
        print(f"    Region {idx}: [{bx1}, {by1}, {bx2}, {by2}] (w={bw}, h={bh})")
        # Save a crop of the original image
        crop = img[by1:by2, bx1:bx2]
        crop_name = f"{os.path.splitext(os.path.basename(img_path))[0]}_sub_{idx}.png"
        cv2.imwrite(os.path.join(output_dir, crop_name), crop)
