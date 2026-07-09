import cv2
import math
import numpy as np
import os
from ultralytics import YOLO
import easyocr

# Reconfigure stdout for Windows unicode support
import sys
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

def rectify_crop(image, bbox_metrics, buffer_percent=0.20):
    img_np = np.array(image)
    cx = bbox_metrics['cx']
    cy = bbox_metrics['cy']
    w = bbox_metrics['w']
    h = bbox_metrics['h']
    angle = bbox_metrics['angle']
    
    bw = w * (1.0 + buffer_percent)
    bh = h * (1.0 + buffer_percent)
    
    diag = math.sqrt(bw**2 + bh**2)
    crop_size = int(math.ceil(diag)) + 20
    half_size = crop_size // 2
    
    x_min = int(round(cx - half_size))
    y_min = int(round(cy - half_size))
    x_max = x_min + crop_size
    y_max = y_min + crop_size
    
    img_h, img_w = img_np.shape[:2]
    
    pad_left = max(0, -x_min)
    pad_top = max(0, -y_min)
    pad_right = max(0, x_max - img_w)
    pad_bottom = max(0, y_max - img_h)
    
    src_x_min = max(0, x_min)
    src_y_min = max(0, y_min)
    src_x_max = min(img_w, x_max)
    src_y_max = min(img_h, y_max)
    
    sub_img = img_np[src_y_min:src_y_max, src_x_min:src_x_max]
    
    if len(img_np.shape) == 3:
        padded_crop = np.ones((crop_size, crop_size, 3), dtype=np.uint8) * 255
        padded_crop[pad_top:crop_size - pad_bottom, pad_left:crop_size - pad_right] = sub_img
    else:
        padded_crop = np.ones((crop_size, crop_size), dtype=np.uint8) * 255
        padded_crop[pad_top:crop_size - pad_bottom, pad_left:crop_size - pad_right] = sub_img
        
    rot_center = (half_size, half_size)
    M = cv2.getRotationMatrix2D(rot_center, angle, 1.0)
    
    warped = cv2.warpAffine(
        padded_crop, 
        M, 
        (crop_size, crop_size), 
        flags=cv2.INTER_CUBIC, 
        borderMode=cv2.BORDER_CONSTANT, 
        borderValue=(255, 255, 255) if len(img_np.shape) == 2 else (255, 255, 255)
    )
    
    bx_min = int(round(half_size - bw / 2))
    by_min = int(round(half_size - bh / 2))
    bx_max = bx_min + int(round(bw))
    by_max = by_min + int(round(bh))
    
    bx_min = max(0, bx_min)
    by_min = max(0, by_min)
    bx_max = min(crop_size, bx_max)
    by_max = min(crop_size, by_max)
    
    return warped[by_min:by_max, bx_min:bx_max]

def search():
    model_path = "D:/Internship/OCR_PDF/YOLO_expression_best.pt"
    model = YOLO(model_path)
    
    inputs = [
        "d:/Internship/OCR_PDF/internt-ocrmodel/eval_output/eval_page.png",
        "d:/Internship/OCR_PDF/sign-detection-yolo8n/toTest/1.jpeg",
        "d:/Internship/OCR_PDF/sign-detection-yolo8n/toTest/2.jpeg"
    ]
    
    reader = easyocr.Reader(['en'])
    
    for inp in inputs:
        if not os.path.exists(inp):
            print(f"Skipping {inp} (not found)")
            continue
        print(f"Processing {inp}...")
        img = cv2.imread(inp)
        H, W, C = img.shape
        img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        _, thresh = cv2.threshold(img_gray, 180, 255, cv2.THRESH_BINARY)
        img_rgb = cv2.cvtColor(thresh, cv2.COLOR_GRAY2RGB)
        
        results = model(img_rgb, verbose=False, conf=0.25, imgsz=1280)
        result = results[0]
        if result.obb is None or len(result.obb) == 0:
            print("No boxes found")
            continue
            
        xywhr = result.obb.xywhr.cpu().numpy()
        for idx, (cx, cy, w, h, r) in enumerate(xywhr):
            angle_deg = math.degrees(r)
            crop = rectify_crop(
                img_rgb,
                bbox_metrics={'cx': cx, 'cy': cy, 'w': w, 'h': h, 'angle': angle_deg},
                buffer_percent=0.20
            )
            ocr_res = reader.readtext(crop)
            texts = [x[1] for x in ocr_res]
            text_str = " ".join(texts)
            if any(k in text_str for k in ["3.44", "83", "1.25", "344", "44"]):
                print(f"Found match in {inp} at index {idx}: cx={cx:.1f}, cy={cy:.1f}, w={w:.1f}, h={h:.1f}, angle={angle_deg:.1f}")
                print(f"Text: {texts}")
                # Save the cropped region for visual debug
                out_path = f"d:/Internship/OCR_PDF/BoudningBoxCleaning/scratch/found_{os.path.basename(inp)}_{idx}.png"
                cv2.imwrite(out_path, cv2.cvtColor(crop, cv2.COLOR_RGB2BGR))
                print(f"Saved to {out_path}")

if __name__ == "__main__":
    search()
