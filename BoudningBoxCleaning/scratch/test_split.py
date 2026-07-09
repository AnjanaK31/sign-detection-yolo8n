import cv2
import numpy as np
import os
import glob

images_path = r"d:\Internship\OCR_PDF\sign-detection-yolo8n\toTest\CLEANINTEST\*"
images = glob.glob(images_path)

def get_foreground_mask(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    corners = [gray[0,0], gray[0,-1], gray[-1,0], gray[-1,-1]]
    avg_corner = np.mean(corners)
    is_white_bg = avg_corner > 127
    
    if is_white_bg:
        # white background -> foreground is dark pixels
        _, thresh = cv2.threshold(gray, 240, 255, cv2.THRESH_BINARY_INV)
    else:
        # black background -> foreground is bright pixels
        _, thresh = cv2.threshold(gray, 15, 255, cv2.THRESH_BINARY)
    return thresh, is_white_bg

for img_path in images:
    if not img_path.lower().endswith(('.png', '.jpg', '.jpeg')):
        continue
    print(f"\n--- Analyzing {os.path.basename(img_path)} ---")
    img = cv2.imread(img_path)
    if img is None:
        print(f"Failed to load {img_path}")
        continue
    h, w = img.shape[:2]
    
    thresh, is_white_bg = get_foreground_mask(img)
    
    # Let's check projection profiles
    row_proj = np.sum(thresh > 0, axis=1) # count of fg pixels per row
    col_proj = np.sum(thresh > 0, axis=0) # count of fg pixels per col
    
    # Print profile summaries
    zero_rows = np.sum(row_proj == 0)
    zero_cols = np.sum(col_proj == 0)
    print(f"  Dimensions: {w}x{h}")
    print(f"  Zero rows (pure bg): {zero_rows} ({zero_rows/h*100:.1f}%)")
    print(f"  Zero cols (pure bg): {zero_cols} ({zero_cols/w*100:.1f}%)")
    
    # Find contiguous non-zero segments in row_proj (horizontal bands)
    # and within each horizontal band, find contiguous non-zero segments in col_proj
    # or vice versa.
    # Let's write a simple recursive or iterative algorithm that splits the bounding box
    # of the image.
    
    def split_bounding_box(x1, y1, x2, y2, level=0):
        # crop thresh to the bounding box
        sub_mask = thresh[y1:y2, x1:x2]
        sh, sw = sub_mask.shape
        if sh <= 0 or sw <= 0:
            return []
            
        sub_row_proj = np.sum(sub_mask > 0, axis=1)
        sub_col_proj = np.sum(sub_mask > 0, axis=0)
        
        # Check if there is any clean horizontal split (a row with 0 fg pixels)
        # To be robust, let's find the longest run of 0s, or any run of 0s.
        # Let's find runs of 0s.
        zero_row_indices = np.where(sub_row_proj == 0)[0]
        if len(zero_row_indices) > 0:
            # Group into contiguous zero rows
            runs = []
            start = zero_row_indices[0]
            for i in range(1, len(zero_row_indices)):
                if zero_row_indices[i] != zero_row_indices[i-1] + 1:
                    runs.append((start, zero_row_indices[i-1]))
                    start = zero_row_indices[i]
            runs.append((start, zero_row_indices[-1]))
            
            # Find the run that is closest to the middle, or just split by the largest runs.
            # Let's filter runs that have a minimum height, say at least 5 pixels, to avoid splitting on tiny noise.
            valid_runs = [r for r in runs if (r[1] - r[0] + 1) >= 5]
            if len(valid_runs) > 0:
                # Let's split by the largest valid run (or all valid runs)
                # If we split by all valid runs, we get multiple horizontal bands
                bands = []
                last_y = 0
                for r_start, r_end in valid_runs:
                    split_y = (r_start + r_end) // 2
                    if split_y > last_y + 10: # minimum height for a sub-band
                        bands.append((last_y, split_y))
                    last_y = split_y
                if sh - last_y > 10:
                    bands.append((last_y, sh))
                
                if len(bands) > 1:
                    results = []
                    for b_y1, b_y2 in bands:
                        # map back to global coordinates
                        res = split_bounding_box(x1, y1 + b_y1, x2, y1 + b_y2, level + 1)
                        results.extend(res)
                    return results
                    
        # If no horizontal split, check vertical split
        zero_col_indices = np.where(sub_col_proj == 0)[0]
        if len(zero_col_indices) > 0:
            runs = []
            start = zero_col_indices[0]
            for i in range(1, len(zero_col_indices)):
                if zero_col_indices[i] != zero_col_indices[i-1] + 1:
                    runs.append((start, zero_col_indices[i-1]))
                    start = zero_col_indices[i]
            runs.append((start, zero_col_indices[-1]))
            
            valid_runs = [r for r in runs if (r[1] - r[0] + 1) >= 5]
            if len(valid_runs) > 0:
                bands = []
                last_x = 0
                for r_start, r_end in valid_runs:
                    split_x = (r_start + r_end) // 2
                    if split_x > last_x + 10:
                        bands.append((last_x, split_x))
                    last_x = split_x
                if sw - last_x > 10:
                    bands.append((last_x, sw))
                
                if len(bands) > 1:
                    results = []
                    for b_x1, b_x2 in bands:
                        res = split_bounding_box(x1 + b_x1, y1, x1 + b_x2, y2, level + 1)
                        results.extend(res)
                    return results
                    
        # If we cannot split further, return the current bounding box
        # But wait, only if it actually contains foreground pixels!
        if np.sum(sub_mask > 0) > 50: # at least 50 foreground pixels to be a drawing
            # Also shrink the box to the exact bounding box of the foreground pixels inside this region
            y_indices, x_indices = np.where(sub_mask > 0)
            if len(y_indices) > 0:
                min_y, max_y = np.min(y_indices), np.max(y_indices)
                min_x, max_x = np.min(x_indices), np.max(x_indices)
                return [(x1 + min_x, y1 + min_y, x1 + max_x, y1 + max_y)]
        return []

    # Initial split call
    boxes = split_bounding_box(0, 0, w, h)
    print(f"  Split-based segmented regions found: {len(boxes)}")
    for i, box in enumerate(boxes[:10]):
        bx1, by1, bx2, by2 = box
        bw, bh = bx2 - bx1 + 1, by2 - by1 + 1
        print(f"    Region {i}: [{bx1}, {by1}, {bx2}, {by2}] (w={bw}, h={bh})")
    if len(boxes) > 10:
        print(f"    ... and {len(boxes)-10} more regions.")
