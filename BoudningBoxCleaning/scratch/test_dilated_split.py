import cv2
import numpy as np
import os
import glob

images_path = r"d:\Internship\OCR_PDF\sign-detection-yolo8n\toTest\CLEANINTEST\*"
images = glob.glob(images_path)

output_dir = r"d:\Internship\OCR_PDF\TESTIMAGES_SEGMENTED_DEBUG_DILATED"
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

def remove_outer_border(thresh):
    # Find contours and filter out the page-level border frame
    h, w = thresh.shape
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(thresh, connectivity=8)
    cleaned = thresh.copy()
    for label in range(1, num_labels):
        cw = stats[label, cv2.CC_STAT_WIDTH]
        ch = stats[label, cv2.CC_STAT_HEIGHT]
        # Discard components that span almost the entire image width/height (border lines)
        if cw > 0.95 * w or ch > 0.95 * h:
            comp_mask = labels == label
            cleaned[comp_mask] = 0
    return cleaned

for img_path in images:
    if not img_path.lower().endswith(('.png', '.jpg', '.jpeg')):
        continue
    print(f"\nProcessing {os.path.basename(img_path)}...")
    img = cv2.imread(img_path)
    if img is None:
        continue
    h, w = img.shape[:2]
    thresh, is_white_bg = get_foreground_mask(img)
    
    # 1. Remove page-level outer border frame
    cleaned_thresh = remove_outer_border(thresh)
    
    # 2. Dilate with a large kernel to merge sub-drawing contents
    # We want a kernel proportional to the image size (e.g. ~3% of image width/height)
    kernel_size = int(max(w, h) * 0.03)
    if kernel_size % 2 == 0:
        kernel_size += 1
    print(f"  Using dilation kernel size: {kernel_size}x{kernel_size}")
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_size, kernel_size))
    dilated_thresh = cv2.dilate(cleaned_thresh, kernel, iterations=1)
    
    # 3. Projection split on dilated_thresh
    def split_bounding_box(x1, y1, x2, y2, mask):
        sub_mask = mask[y1:y2, x1:x2]
        sh, sw = sub_mask.shape
        if sh <= kernel_size or sw <= kernel_size:
            return []
            
        sub_row_proj = np.sum(sub_mask > 0, axis=1)
        sub_col_proj = np.sum(sub_mask > 0, axis=0)
        
        # Check vertical split first (columns)
        zero_cols = np.where(sub_col_proj == 0)[0]
        if len(zero_cols) > 0:
            runs = []
            start = zero_cols[0]
            for i in range(1, len(zero_cols)):
                if zero_cols[i] != zero_cols[i-1] + 1:
                    runs.append((start, zero_cols[i-1]))
                    start = zero_cols[i]
            runs.append((start, zero_cols[-1]))
            
            # Keep runs wider than half the kernel size to be robust
            valid_runs = [r for r in runs if (r[1] - r[0] + 1) >= (kernel_size // 2)]
            if len(valid_runs) > 0:
                splits = [0]
                for r_start, r_end in valid_runs:
                    splits.append((r_start + r_end) // 2)
                splits.append(sw)
                
                results = []
                for i in range(len(splits) - 1):
                    s_x1 = splits[i]
                    s_x2 = splits[i+1]
                    if s_x2 - s_x1 > kernel_size:
                        results.extend(split_bounding_box(x1 + s_x1, y1, x1 + s_x2, y2, mask))
                return results
                
        # Check horizontal split (rows)
        zero_rows = np.where(sub_row_proj == 0)[0]
        if len(zero_rows) > 0:
            runs = []
            start = zero_rows[0]
            for i in range(1, len(zero_rows)):
                if zero_rows[i] != zero_rows[i-1] + 1:
                    runs.append((start, zero_rows[i-1]))
                    start = zero_rows[i]
            runs.append((start, zero_rows[-1]))
            
            valid_runs = [r for r in runs if (r[1] - r[0] + 1) >= (kernel_size // 2)]
            if len(valid_runs) > 0:
                splits = [0]
                for r_start, r_end in valid_runs:
                    splits.append((r_start + r_end) // 2)
                splits.append(sh)
                
                results = []
                for i in range(len(splits) - 1):
                    s_y1 = splits[i]
                    s_y2 = splits[i+1]
                    if s_y2 - s_y1 > kernel_size:
                        results.extend(split_bounding_box(x1, y1 + s_y1, x2, y1 + s_y2, mask))
                return results
                
        # Bounding box of original foreground pixels in this region
        orig_sub_mask = cleaned_thresh[y1:y2, x1:x2]
        if np.sum(orig_sub_mask > 0) > 200: # ignore minor noise components
            y_indices, x_indices = np.where(orig_sub_mask > 0)
            min_y, max_y = np.min(y_indices), np.max(y_indices)
            min_x, max_x = np.min(x_indices), np.max(x_indices)
            
            padding = 10
            crop_y1 = max(0, y1 + min_y - padding)
            crop_y2 = min(h, y1 + max_y + padding)
            crop_x1 = max(0, x1 + min_x - padding)
            crop_x2 = min(w, x1 + max_x + padding)
            return [(crop_x1, crop_y1, crop_x2, crop_y2)]
        return []

    boxes = split_bounding_box(0, 0, w, h, dilated_thresh)
    print(f"  Found {len(boxes)} segmented regions.")
    for idx, (bx1, by1, bx2, by2) in enumerate(boxes):
        bw, bh = bx2 - bx1, by2 - by1
        print(f"    Region {idx}: [{bx1}, {by1}, {bx2}, {by2}] (w={bw}, h={bh})")
        crop = img[by1:by2, bx1:bx2]
        crop_name = f"{os.path.splitext(os.path.basename(img_path))[0]}_sub_{idx}.png"
        cv2.imwrite(os.path.join(output_dir, crop_name), crop)
