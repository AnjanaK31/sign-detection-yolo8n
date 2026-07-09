import cv2
import numpy as np
import os
import torch
from PIL import Image

import sys
sys.path.insert(0, "d:/Internship/OCR_PDF/internt-ocrmodel")

from data_gen import SyntheticDataGenerator
from pipeline import load_yolo_model, process_page
from classifier import SymbolClassifier

# Generate a single page
generator = SyntheticDataGenerator()
bg_dir = "temp_debug_bg"
generator.generate_backgrounds(bg_dir, count=1)
bg_path = os.path.join(bg_dir, "bg_0.png")
page_img, page_colored, labels, gt_details = generator.generate_full_page(bg_path, num_annotations=15)

yolo_model = load_yolo_model("../YOLO_expression_best.pt")
classifier = SymbolClassifier(model_path="../sign-detection-yolo8n/classifier_best.pt", device="cpu")

# Run with low conf to get all detections
annotated_img, detections = process_page(
    page_img,
    yolo_model,
    classifier,
    conf_threshold=0.15,
    output_dir="debug_output",
    gt_colored_pil=page_colored
)

print("\nDetailed Detection analysis:")
for d in detections:
    idx = d["idx"]
    yolo_conf = d["yolo_conf"]
    pred_text = d["text"]
    
    # Let's find if there is a matching ground truth item by overlap
    # gt_details has "corners" and "text"
    cx, cy = d["center"]
    best_gt = None
    min_dist = 99999.0
    for gt in gt_details:
        # compute distance from gt center
        gt_corners = np.array(gt["corners"])
        gt_cx, gt_cy = np.mean(gt_corners, axis=0)
        dist = np.linalg.norm([cx - gt_cx, cy - gt_cy])
        if dist < min_dist:
            min_dist = dist
            best_gt = gt
            
    gt_text = best_gt["text"] if (best_gt and min_dist < 150) else "None (CAD/Background)"
    
    tpr, ldr = "-", "-"
    if "evaluation" in d:
        tpr = f"{d['evaluation']['text_preservation_rate']*100:.2f}%"
        ldr = f"{d['evaluation']['line_deletion_rate']*100:.2f}%"
        
    print(f"Crop {idx}: YOLO Conf={yolo_conf:.3f} | Pred Text='{pred_text}' | GT Text='{gt_text}' (dist={min_dist:.1f}) | TPR={tpr} | LDR={ldr}")

import shutil
shutil.rmtree(bg_dir)
