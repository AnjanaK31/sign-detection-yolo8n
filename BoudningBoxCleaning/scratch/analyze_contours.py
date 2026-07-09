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

for img_path in images:
    if not img_path.lower().endswith(('.png', '.jpg', '.jpeg')):
        continue
    img = cv2.imread(img_path)
    if img is None:
        continue
    h, w = img.shape[:2]
    thresh, is_white_bg = get_foreground_mask(img)
    
    # Let's find connected components
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(thresh)
    
    print(f"\n--- Contours in {os.path.basename(img_path)} ---")
    print(f"Total connected components: {num_labels}")
    
    # Filter out components:
    # 1. Background component (label 0 is background)
    # 2. Components that are very small (e.g. noise, < 20 pixels)
    # 3. Component that represents the frame of the image (if any, e.g. covering > 80% width or height)
    
    valid_components = []
    for label in range(1, num_labels):
        x = stats[label, cv2.CC_STAT_LEFT]
        y = stats[label, cv2.CC_STAT_TOP]
        cw = stats[label, cv2.CC_STAT_WIDTH]
        ch = stats[label, cv2.CC_STAT_HEIGHT]
        area = stats[label, cv2.CC_STAT_AREA]
        
        # Check if it spans almost the entire image (border frame)
        if cw > 0.95 * w or ch > 0.95 * h:
            print(f"  Discarded potential frame: bbox=[{x}, {y}, {cw}, {ch}], area={area}")
            continue
            
        if area < 100: # noise
            continue
            
        valid_components.append((x, y, cw, ch, area))
        
    print(f"Valid components (area >= 100, not frame): {len(valid_components)}")
    # Print the top 10 largest valid components
    valid_components = sorted(valid_components, key=lambda x: x[4], reverse=True)
    for i, (x, y, cw, ch, area) in enumerate(valid_components[:10]):
        print(f"    Comp {i}: bbox=[x={x}, y={y}, w={cw}, h={ch}], area={area}")
