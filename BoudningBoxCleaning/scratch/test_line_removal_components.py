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
        _, thresh = cv2.threshold(gray, 240, 255, cv2.THRESH_BINARY_INV)
    else:
        _, thresh = cv2.threshold(gray, 15, 255, cv2.THRESH_BINARY)
    return thresh, is_white_bg

def remove_grid_lines(thresh):
    h, w = thresh.shape
    # Detect horizontal lines
    horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (int(w * 0.05), 1))
    detected_horizontal = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, horizontal_kernel)
    
    # Detect vertical lines
    vertical_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, int(h * 0.05)))
    detected_vertical = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, vertical_kernel)
    
    # Combined mask
    lines_mask = cv2.bitwise_or(detected_horizontal, detected_vertical)
    cleaned = cv2.subtract(thresh, lines_mask)
    return cleaned

for img_path in images:
    if not img_path.lower().endswith(('.png', '.jpg', '.jpeg')):
        continue
    print(f"\nAnalyzing {os.path.basename(img_path)}...")
    img = cv2.imread(img_path)
    if img is None:
        continue
    h, w = img.shape[:2]
    thresh, is_white_bg = get_foreground_mask(img)
    
    # Remove outer page borders
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(thresh, connectivity=8)
    no_border_thresh = thresh.copy()
    for label in range(1, num_labels):
        cw = stats[label, cv2.CC_STAT_WIDTH]
        ch = stats[label, cv2.CC_STAT_HEIGHT]
        if cw > 0.95 * w or ch > 0.95 * h:
            no_border_thresh[labels == label] = 0
            
    # Remove grid lines
    cleaned_thresh = remove_grid_lines(no_border_thresh)
    
    # Find connected components on cleaned_thresh
    # Let's dilate slightly to keep text and drawings unified (e.g. 10x10 kernel)
    dil_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 15))
    dilated = cv2.dilate(cleaned_thresh, dil_kernel, iterations=1)
    
    num_labels_c, labels_c, stats_c, _ = cv2.connectedComponentsWithStats(dilated, connectivity=8)
    print(f"  Total components after line removal and 15x15 dilation: {num_labels_c}")
    
    valid_boxes = []
    for label in range(1, num_labels_c):
        x = stats_c[label, cv2.CC_STAT_LEFT]
        y = stats_c[label, cv2.CC_STAT_TOP]
        cw = stats_c[label, cv2.CC_STAT_WIDTH]
        ch = stats_c[label, cv2.CC_STAT_HEIGHT]
        area = stats_c[label, cv2.CC_STAT_AREA]
        
        # Discard very small components (noise/isolated characters)
        # and very large page frames
        if cw < 80 or ch < 80 or area < 2000:
            continue
        if cw > 0.9 * w and ch > 0.9 * h:
            continue
            
        valid_boxes.append((x, y, cw, ch, area))
        
    valid_boxes = sorted(valid_boxes, key=lambda x: x[4], reverse=True)
    print(f"  Filtered to {len(valid_boxes)} sub-drawing candidates:")
    for i, (x, y, cw, ch, area) in enumerate(valid_boxes[:15]):
        print(f"    Region {i}: bbox=[x={x}, y={y}, w={cw}, h={ch}], area={area}")
