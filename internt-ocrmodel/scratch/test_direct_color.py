import os
import sys
import cv2
import numpy as np
from PIL import Image

sys.path.insert(0, "d:/Internship/OCR_PDF/internt-ocrmodel")

from preprocessor import to_grayscale, apply_threshold
from pipeline import load_yolo_model

def test_direct_coloring():
    test_dir = "d:/Internship/OCR_PDF/sign-detection-yolo8n/toTest"
    img_name = "2.jpeg"
    img_path = os.path.join(test_dir, img_name)
    
    yolo_path = "D:/Internship/OCR_PDF/YOLO_expression_best.pt"
    yolo_model = load_yolo_model(yolo_path)
    
    page_img = Image.open(img_path)
    gray = to_grayscale(page_img)
    thresh = apply_threshold(gray)
    binary_foreground = cv2.bitwise_not(thresh)
    
    img_bgr = cv2.cvtColor(np.array(page_img.convert("RGB")), cv2.COLOR_RGB2BGR)
    results = yolo_model(img_bgr, verbose=False, conf=0.25, imgsz=1280)
    result = results[0]
    
    yolo_mask = np.zeros(binary_foreground.shape, dtype=np.uint8)
    if result.obb is not None and len(result.obb) > 0:
        xyxyxyxy = result.obb.xyxyxyxy.cpu().numpy()
        for corners in xyxyxyxy:
            pts = np.array(corners, dtype=np.int32)
            cv2.fillPoly(yolo_mask, [pts], 255)
            
    num_labels, labels_im, stats, _ = cv2.connectedComponentsWithStats(binary_foreground)
    
    font_threshold = 20
    big_lines_mask = np.zeros(binary_foreground.shape, dtype=np.uint8)
    for i in range(1, num_labels):
        w = stats[i, cv2.CC_STAT_WIDTH]
        h = stats[i, cv2.CC_STAT_HEIGHT]
        if h > font_threshold or w > font_threshold:
            big_lines_mask[labels_im == i] = 255
            
    green_mask = cv2.bitwise_and(binary_foreground, yolo_mask)
    red_mask = cv2.bitwise_and(big_lines_mask, cv2.bitwise_not(yolo_mask))
    
    out_canvas = np.array(page_img.convert("RGB"))
    
    # Direct pixel coloring
    out_canvas[red_mask > 0] = [255, 0, 0]
    out_canvas[green_mask > 0] = [0, 255, 0]
    
    Image.fromarray(out_canvas).save("d:/Internship/OCR_PDF/internt-ocrmodel/scratch/test_direct_color.png")
    print("Saved test_direct_color.png")

if __name__ == "__main__":
    test_direct_coloring()
