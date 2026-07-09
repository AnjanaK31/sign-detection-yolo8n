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

def inspect_2():
    model_path = "D:/Internship/OCR_PDF/YOLO_expression_best.pt"
    model = YOLO(model_path)
    
    inp = "d:/Internship/OCR_PDF/sign-detection-yolo8n/toTest/2.jpeg"
    img = cv2.imread(inp)
    H, W, C = img.shape
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(img_gray, 180, 255, cv2.THRESH_BINARY)
    img_rgb = cv2.cvtColor(thresh, cv2.COLOR_GRAY2RGB)
    
    results = model(img_rgb, verbose=False, conf=0.25, imgsz=1280)
    result = results[0]
    if result.obb is None or len(result.obb) == 0:
        print("No boxes found in 2.jpeg")
        return
        
    reader = easyocr.Reader(['en'])
    xywhr = result.obb.xywhr.cpu().numpy()
    print(f"Found {len(xywhr)} bounding boxes in 2.jpeg")
    
    for idx, (cx, cy, w, h, r) in enumerate(xywhr):
        angle_deg = math.degrees(r)
        crop = rectify_crop(
            img_rgb,
            bbox_metrics={'cx': cx, 'cy': cy, 'w': w, 'h': h, 'angle': angle_deg},
            buffer_percent=0.20
        )
        ocr_res = reader.readtext(crop)
        texts = [x[1] for x in ocr_res]
        print(f"Box {idx}: Text = {texts} | cx={cx:.1f}, cy={cy:.1f}, w={w:.1f}, h={h:.1f}")
        out_path = f"d:/Internship/OCR_PDF/BoudningBoxCleaning/scratch/2_crop_{idx}.png"
        cv2.imwrite(out_path, cv2.cvtColor(crop, cv2.COLOR_RGB2BGR))

if __name__ == "__main__":
    inspect_2()
