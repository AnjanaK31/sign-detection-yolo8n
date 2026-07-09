import os
import sys
import cv2
import numpy as np
import torch
from PIL import Image

sys.path.insert(0, "d:/Internship/OCR_PDF/internt-ocrmodel")

from pipeline import load_yolo_model
from classifier import SymbolClassifier
from rectifier import rectify_crop
import line_cleaner

# Paths
yolo_path = "../YOLO_expression_best.pt"
classifier_path = "../sign-detection-yolo8n/classifier_best.pt"
eval_page_path = "eval_output/eval_page.png"

# Load page
page_img = Image.open(eval_page_path)
img_bgr = cv2.cvtColor(np.array(page_img), cv2.COLOR_RGB2BGR)

# Run YOLO
yolo_model = load_yolo_model(yolo_path)
results = yolo_model(img_bgr, verbose=False, conf=0.25, imgsz=1280)
result = results[0]

if result.obb is not None and len(result.obb) > 4:
    xywhr = result.obb.xywhr.cpu().numpy()
    cx, cy, w, h, r = xywhr[4]
    angle_deg = np.degrees(r)
    
    # Extract standard and big crops
    crop_std = rectify_crop(
        page_img,
        bbox_metrics={'cx': cx, 'cy': cy, 'w': w, 'h': h, 'angle': angle_deg},
        buffer_percent=0.10
    )
    crop_big = rectify_crop(
        page_img,
        bbox_metrics={'cx': cx, 'cy': cy, 'w': w, 'h': h, 'angle': angle_deg},
        buffer_percent=0.40
    )
    
    thresh_std = cv2.bitwise_not(np.array(crop_std.convert("L")))
    _, thresh_std = cv2.threshold(thresh_std, 127, 255, cv2.THRESH_BINARY)
    
    skel = line_cleaner.skeletonize(thresh_std)
    branches, junctions = line_cleaner.split_skeleton_into_branches(skel)
    
    # Run branch classification with border_margin = 2
    H_std, W_std = thresh_std.shape
    border_margin = 2  # TEST value
    
    print(f"Testing border_margin = {border_margin}")
    for idx, branch_pts in enumerate(branches):
        touches_border = np.any((branch_pts[:, 0] < border_margin) | 
                                (branch_pts[:, 0] > W_std - 1 - border_margin) | 
                                (branch_pts[:, 1] < border_margin) | 
                                (branch_pts[:, 1] > H_std - 1 - border_margin))
        shape_type, score = line_cleaner.classify_branch_shape(branch_pts)
        print(f"Branch {idx}: len={len(branch_pts)}, type={shape_type}, score={score:.2f}, border={touches_border}")

else:
    print("Could not find crop 4 in page detections.")
