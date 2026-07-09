import cv2
import numpy as np
import os
import glob

# Use Python310 interpreter that has cv2, numpy, PIL
images_path = r"d:\Internship\OCR_PDF\sign-detection-yolo8n\toTest\CLEANINTEST\*"
images = glob.glob(images_path)

print(f"Found {len(images)} images to inspect.")
for img_path in images:
    if not img_path.lower().endswith(('.png', '.jpg', '.jpeg')):
        continue
    img = cv2.imread(img_path)
    if img is None:
        print(f"Failed to load {img_path}")
        continue
    h, w, c = img.shape
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Check background: average corner pixel values or standard min/max
    corners = [gray[0,0], gray[0,-1], gray[-1,0], gray[-1,-1]]
    avg_corner = np.mean(corners)
    
    # Let's count non-white/non-black pixels
    # If background is white (average > 127), foreground is black (values < threshold)
    # If background is black, foreground is white
    is_white_bg = avg_corner > 127
    
    if is_white_bg:
        # Binarize: foreground is 1 (where pixel < 240)
        _, thresh = cv2.threshold(gray, 240, 255, cv2.THRESH_BINARY_INV)
    else:
        # Binarize: foreground is 1 (where pixel > 15)
        _, thresh = cv2.threshold(gray, 15, 255, cv2.THRESH_BINARY)
        
    non_zero = cv2.countNonZero(thresh)
    pct = (non_zero / (h * w)) * 100
    
    # Find connected components or contours to see how they are distributed
    # Let's do a simple dilation to merge nearby pixel regions and find outer boxes
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (20, 20))
    dilated = cv2.dilate(thresh, kernel, iterations=2)
    contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    print(f"Image: {os.path.basename(img_path)}")
    print(f"  Dimensions: {w}x{h}, Channels: {c}")
    print(f"  Estimated Background: {'White' if is_white_bg else 'Black'} (avg corner: {avg_corner:.1f})")
    print(f"  Foreground Pixels: {non_zero} ({pct:.2f}%)")
    print(f"  Number of external contours (with 20x20 dilation, 2 iterations): {len(contours)}")
    
    # Print the areas and bounding boxes of the largest contours
    contours = sorted(contours, key=cv2.contourArea, reverse=True)
    for i, ctr in enumerate(contours[:5]):
        x, y, cw, ch = cv2.boundingRect(ctr)
        area = cv2.contourArea(ctr)
        print(f"    Contour {i}: bbox=[x={x}, y={y}, w={cw}, h={ch}], area={area:.1f}")
