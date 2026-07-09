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

for img_path in images:
    if not img_path.lower().endswith(('.png', '.jpg', '.jpeg')):
        continue
    print(f"Visualizing {os.path.basename(img_path)}...")
    img = cv2.imread(img_path)
    if img is None:
        continue
    h, w = img.shape[:2]
    thresh, is_white_bg = get_foreground_mask(img)
    
    # Draw contours of components with area > 100
    # Let's find contours
    contours, hierarchy = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    vis = img.copy()
    box_count = 0
    for ctr in contours:
        area = cv2.contourArea(ctr)
        if area < 100:
            continue
        x, y, cw, ch = cv2.boundingRect(ctr)
        # Check if it spans the whole image (border frame)
        if cw > 0.95 * w or ch > 0.95 * h:
            cv2.rectangle(vis, (x, y), (x+cw, y+ch), (0, 0, 255), 5) # Red for borders
        else:
            cv2.rectangle(vis, (x, y), (x+cw, y+ch), (0, 255, 0), 3) # Green for components
            box_count += 1
            
    # Resize vis for easy viewing
    target_h = 1000
    target_w = int(w * (target_h / h))
    vis_resized = cv2.resize(vis, (target_w, target_h))
    
    save_path = os.path.join(output_dir, f"vis_{os.path.basename(img_path)}")
    cv2.imwrite(save_path, vis_resized)
    print(f"  Saved visualization with {box_count} components to {save_path}")
