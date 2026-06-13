import os
import cv2
import numpy as np
import math
from PIL import Image, ImageDraw, ImageFont
from pdf2image import convert_from_path

# Paths
WORKSPACE_DIR = r"d:\Internship\OCR_PDF\sign-detection-yolo8n"
ARTIFACTS_DIR = r"C:\Users\lalit\.gemini\antigravity-ide\brain\8315ea56-beda-40a7-8a00-2359e01f15e6"
PDF_PATH = os.path.join(WORKSPACE_DIR, "pdfs", "blueprint_0.pdf")
YOLO_PATH = os.path.join(WORKSPACE_DIR, "runs", "obb", "runs", "obb", "trained_on_1000_pdfs-2", "weights", "best.pt")

# Import workspace modules
import sys
sys.path.append(WORKSPACE_DIR)
from pipeline import load_yolo_model, preprocess_image
from rectifier import rectify_crop

def skeletonize(img):
    size = np.size(img)
    skel = np.zeros(img.shape, np.uint8)
    element = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))
    done = False
    temp_img = img.copy()
    while not done:
        eroded = cv2.erode(temp_img, element)
        temp = cv2.dilate(eroded, element)
        temp = cv2.subtract(temp_img, temp)
        skel = cv2.bitwise_or(skel, temp)
        temp_img = eroded.copy()
        zeros = size - cv2.countNonZero(temp_img)
        if zeros == size:
            done = True
    return skel

def get_junctions(skel):
    kernel = np.array([[1, 1, 1],
                       [1, 0, 1],
                       [1, 1, 1]], dtype=np.uint8)
    binary_skel = (skel > 0).astype(np.uint8)
    neighbor_count = cv2.filter2D(binary_skel, -1, kernel)
    junctions = (binary_skel == 1) & (neighbor_count > 2)
    return junctions.astype(np.uint8) * 255

def clean_image_premium(crop_pil_big, w_std, h_std, crop_label=""):
    img_big = np.array(crop_pil_big.convert("L"))
    H_big, W_big = img_big.shape
    
    # Standard crop dimensions
    W_std, H_std = int(round(w_std)), int(round(h_std))
    
    # Calculate offset of std crop inside big crop
    dx = (W_big - W_std) // 2
    dy = (H_big - H_std) // 2
    
    # 1. Background Clearing & Thresholding on the big crop
    thresh_big = cv2.adaptiveThreshold(
        img_big, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 15, 3
    )
    
    # 2. Speckle / Dot Noise Reduction via Area Filtering
    num_labels_big, labels_big, stats_big, _ = cv2.connectedComponentsWithStats(thresh_big, connectivity=8)
    denoised_big = np.zeros_like(thresh_big)
    for label in range(1, num_labels_big):
        area = stats_big[label, cv2.CC_STAT_AREA]
        if area >= 25:
            denoised_big[labels_big == label] = 255
            
    # 3. Line Detection on the Big Crop (context-aware)
    skel_big = skeletonize(denoised_big)
    
    # Run Hough lines with a moderate length threshold (35px) to capture segments of CAD lines
    lines_big = cv2.HoughLinesP(skel_big, 1, np.pi/180, threshold=15, minLineLength=35, maxLineGap=6)
    
    hough_line_mask_big = np.zeros_like(denoised_big)
    if lines_big is not None:
        for line in lines_big:
            x1, y1, x2, y2 = line[0]
            
            # Boundary check: a CAD grid/dimension line must touch or come close (<= 25px)
            # to the outer boundaries of the big crop.
            dist1 = min(x1, y1, W_big - 1 - x1, H_big - 1 - y1)
            dist2 = min(x2, y2, W_big - 1 - x2, H_big - 1 - y2)
            
            if min(dist1, dist2) <= 25:
                # Draw detected CAD line with thickness 2
                cv2.line(hough_line_mask_big, (x1, y1), (x2, y2), 255, thickness=2)
            
    # Dilate the big mask slightly to make sure we cover the line thickness
    combined_line_mask_big = cv2.dilate(hough_line_mask_big, cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2)))
    
    # Slice standard crop images directly from the big crop
    denoised_std = denoised_big[dy:dy+H_std, dx:dx+W_std]
    line_mask_std = combined_line_mask_big[dy:dy+H_std, dx:dx+W_std]
    
    # 4. Erase the detected lines from the denoised standard image
    cleaned_thresh = cv2.subtract(denoised_std, line_mask_std)
    
    # 5. Repair character stroke gaps using Morphological Closing (3x3 kernel)
    kernel_close = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    cleaned_thresh = cv2.morphologyEx(cleaned_thresh, cv2.MORPH_CLOSE, kernel_close)
    
    # Convert back to white background, black foreground
    cleaned_img = cv2.bitwise_not(cleaned_thresh)
    
    # Generate standard crop original image for visualization
    orig_std = img_big[dy:dy+H_std, dx:dx+W_std]
    
    # Generate visualization masks (showing standard crop with red lines)
    visualization_mask = cv2.cvtColor(orig_std, cv2.COLOR_GRAY2BGR)
    visualization_mask[line_mask_std == 255] = [0, 0, 255]
    
    return cleaned_img, visualization_mask, orig_std

def draw_label(img, text, bg_color=(30, 30, 30), fg_color=(255, 255, 255)):
    # Create canvas for label at the top
    h, w = img.shape[:2]
    canvas = np.ones((h + 40, w, 3), dtype=np.uint8) * 255
    canvas[40:, :] = img if len(img.shape) == 3 else cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    
    # Draw header bar
    cv2.rectangle(canvas, (0, 0), (w, 40), bg_color, -1)
    
    # Put text
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.5
    thickness = 1
    text_size = cv2.getTextSize(text, font, font_scale, thickness)[0]
    text_x = (w - text_size[0]) // 2
    text_y = (40 + text_size[1]) // 2 - 2
    
    cv2.putText(canvas, text, (text_x, text_y), font, font_scale, fg_color, thickness, cv2.LINE_AA)
    return canvas

def main():
    print("Loading models...")
    yolo = load_yolo_model(YOLO_PATH)
    
    print("Converting PDF page 1...")
    pages = convert_from_path(PDF_PATH, dpi=200)
    page_img = pages[0]
    
    print("Preprocessing page...")
    preprocessed_img = preprocess_image(page_img)
    img_np = np.array(preprocessed_img)
    img_bgr = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
    
    print("Running detection...")
    results = yolo(img_bgr, verbose=False, conf=0.25, imgsz=1280)
    result = results[0]
    
    if result.obb is not None and len(result.obb) > 0:
        xywhr = result.obb.xywhr.cpu().numpy()
        xyxyxyxy = result.obb.xyxyxyxy.cpu().numpy()
        
        for crop_idx in [0, 1]:
            if crop_idx >= len(xywhr): continue
            
            cx, cy, w, h, r = xywhr[crop_idx]
            angle_deg = math.degrees(r)
            
            # Extract standard crop dimensions
            w_std = w * 1.10
            h_std = h * 1.10
            
            # Extract big crop (40% buffer)
            crop_pil_big = rectify_crop(
                preprocessed_img,
                bbox_metrics={'cx': cx, 'cy': cy, 'w': w, 'h': h, 'angle': angle_deg},
                buffer_percent=0.40
            )
            
            # Run our premium clean algorithm
            cleaned_img, viz_mask, orig_std = clean_image_premium(crop_pil_big, w_std, h_std, crop_label=f"Crop {crop_idx}")
            
            # Prepare panels with premium labels
            panel_orig = draw_label(orig_std, "Original Blueprint Crop", bg_color=(50, 50, 50))
            panel_viz = draw_label(viz_mask, "Detected Lines (Red) & Junctions (Green)", bg_color=(180, 50, 50))
            panel_cleaned = draw_label(cleaned_img, "Cleaned Character Output", bg_color=(40, 120, 40))
            
            # Combine horizontally
            H, W = panel_orig.shape[:2]
            separator = np.ones((H, 10, 3), dtype=np.uint8) * 200 # Light grey separator
            
            comparison_grid = np.hstack((panel_orig, separator, panel_viz, separator, panel_cleaned))
            
            # Save the final image as artifact
            save_filename = f"crop_{crop_idx}_premium_cleaned.png"
            save_path = os.path.join(ARTIFACTS_DIR, save_filename)
            cv2.imwrite(save_path, comparison_grid)
            print(f"Saved premium cleaned comparison to: {save_path}")
            
    print("Done!")

if __name__ == "__main__":
    main()
