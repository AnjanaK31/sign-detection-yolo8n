import os
import sys
import cv2
import numpy as np
from PIL import Image

sys.path.insert(0, "d:/Internship/OCR_PDF/internt-ocrmodel")

from preprocessor import to_grayscale, apply_threshold
from pipeline import load_yolo_model

def run_yolo_lines_visualization():
    test_dir = "d:/Internship/OCR_PDF/sign-detection-yolo8n/toTest"
    output_dir = os.path.join(test_dir, "predictions")
    os.makedirs(output_dir, exist_ok=True)
    
    # Locate YOLO model
    yolo_path = "../YOLO_expression_best.pt"
    if not os.path.exists(yolo_path):
        yolo_path = "D:/Internship/OCR_PDF/YOLO_expression_best.pt"
        
    if not os.path.exists(yolo_path):
        print(f"ERROR: YOLO model not found at {yolo_path}")
        return
        
    print(f"Loading YOLO model from: {yolo_path}...")
    yolo_model = load_yolo_model(yolo_path)
    
    # Supported image extensions
    valid_exts = (".png", ".jpg", ".jpeg", ".bmp", ".tiff")
    images = [f for f in os.listdir(test_dir) if f.lower().endswith(valid_exts)]
    
    if not images:
        print(f"No images found in {test_dir}")
        return
        
    print(f"Found {len(images)} images to process.")
    font_threshold = 20
    
    for img_name in images:
        img_path = os.path.join(test_dir, img_name)
        print(f"\nProcessing {img_name}...")
        
        try:
            page_img = Image.open(img_path)
            gray = to_grayscale(page_img)
            thresh = apply_threshold(gray)
            binary_foreground = cv2.bitwise_not(thresh)
            
            W_img, H_img = page_img.size
            
            # 1. Run YOLO to get OBB boxes
            img_bgr = cv2.cvtColor(np.array(page_img.convert("RGB")), cv2.COLOR_RGB2BGR)
            results = yolo_model(img_bgr, verbose=False, conf=0.25, imgsz=1280)
            result = results[0]
            
            # Create YOLO mask
            yolo_mask = np.zeros(binary_foreground.shape, dtype=np.uint8)
            if result.obb is not None and len(result.obb) > 0:
                xyxyxyxy = result.obb.xyxyxyxy.cpu().numpy()
                for corners in xyxyxyxy:
                    pts = np.array(corners, dtype=np.int32)
                    cv2.fillPoly(yolo_mask, [pts], 255)
            
            # 2. Run connected components
            num_labels, labels_im, stats, _ = cv2.connectedComponentsWithStats(binary_foreground)
            
            # 3. Classify components into red (CAD lines outside YOLO) and green (YOLO internal text elements)
            # Create mask of all big line components (islands with width or height > 35)
            big_lines_mask = np.zeros(binary_foreground.shape, dtype=np.uint8)
            for i in range(1, num_labels):
                w = stats[i, cv2.CC_STAT_WIDTH]
                h = stats[i, cv2.CC_STAT_HEIGHT]
                if h > font_threshold or w > font_threshold:
                    big_lines_mask[labels_im == i] = 255
            
            # Mask of yolo contents (foreground pixels inside yolo bounding boxes)
            green_mask = cv2.bitwise_and(binary_foreground, yolo_mask)
            
            # Remaining lines mask (big lines outside yolo bounding boxes)
            red_mask = cv2.bitwise_and(big_lines_mask, cv2.bitwise_not(yolo_mask))
            
            # 4. Generate original page canvas in RGB and color pixels directly
            out_canvas = np.array(page_img.convert("RGB"))
            out_canvas[red_mask > 0] = [255, 0, 0]    # Red for lines outside YOLO
            out_canvas[green_mask > 0] = [0, 255, 0]  # Green for content inside YOLO
            
            # 5. Save output
            base_name, _ = os.path.splitext(img_name)
            out_img_name = f"{base_name}_yolo_lines.png"
            out_path = os.path.join(output_dir, out_img_name)
            Image.fromarray(out_canvas).save(out_path)
            print(f"  -> Saved output to: {out_path}")
            
            # Copy to current script folder
            dest = f"d:/Internship/OCR_PDF/internt-ocrmodel/scratch/{out_img_name}"
            Image.fromarray(out_canvas).save(dest)
            print(f"  -> Copied to: {dest}")
            
        except Exception as e:
            print(f"Error processing {img_name}: {e}")
            
    print("\nVisualization complete!")

if __name__ == "__main__":
    run_yolo_lines_visualization()
