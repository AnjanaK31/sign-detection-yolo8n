import os
import argparse
import json
import torch
import cv2
import numpy as np
import math
from PIL import Image, ImageDraw, ImageFont
from ultralytics import YOLO

# Import custom modules
from rectifier import rectify_crop
from classifier import SymbolClassifier, CLASSES
from preprocessor import full_preprocess
from line_cleaner import clean_patch_lines, evaluate_cleaning

CLASS_TO_CHAR = {
    '0': '0', '1': '1', '2': '2', '3': '3', '4': '4',
    '5': '5', '6': '6', '7': '7', '8': '8', '9': '9',
    'plus_minus': '+/-',
    'diameter': 'DIA',
    'radius': 'R',
    'Rz': 'Rz',
    'Ra': 'Ra',
    'perpendicular': 'PERP',
    'parallel': 'PARA',
    'circularity': 'CIRC',
    'true_position': 'TP',
    'arrow': 'Arrow',
    'comma': ','
}

def find_gt_colored_page(input_path, page_index):
    if not input_path:
        return None
    # Try direct replacement
    if not input_path.lower().endswith(".pdf"):
        dir_name = os.path.dirname(input_path)
        base_name_no_ext = os.path.splitext(os.path.basename(input_path))[0]
        candidate_path = os.path.join(dir_name, f"{base_name_no_ext}_gt_colored.png")
        if os.path.exists(candidate_path):
            return Image.open(candidate_path)
    
    # If PDF, try to find matching page file in standard search paths
    import re
    pdf_match = re.search(r'blueprint_(\d+)', os.path.basename(input_path))
    if pdf_match:
        pdf_idx = int(pdf_match.group(1))
        page_idx = pdf_idx * 5 + page_index
        for root, dirs, files in os.walk("."):
            for f in files:
                if f == f"page_{page_idx}_gt_colored.png":
                    return Image.open(os.path.join(root, f))
                    
    # Fallback search: just find any matching page index file
    for root, dirs, files in os.walk("."):
        for f in files:
            if f == f"page_{page_index}_gt_colored.png":
                return Image.open(os.path.join(root, f))
    return None

def load_yolo_model(yolo_path):
    print(f"Loading YOLOv8-OBB model from: {yolo_path}")
    return YOLO(yolo_path)

def preprocess_image(img_pil):
    return full_preprocess(img_pil)

def process_page(img_pil, yolo_model, classifier, conf_threshold=0.25, output_dir="output_pipeline", gt_colored_pil=None):
    # Phase 1 Preprocessing
    preprocessed_img = preprocess_image(img_pil)
    
    img_np = np.array(preprocessed_img)
    img_bgr = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
    
    # Run YOLOv8-OBB inference (imgsz=1280)
    results = yolo_model(img_bgr, verbose=False, conf=conf_threshold, imgsz=1280)
    result = results[0]
    
    # Calculate full-page connected components and yolo mask to protect symbol-internal lines
    gray_page = cv2.cvtColor(np.array(preprocessed_img.convert("RGB")), cv2.COLOR_RGB2GRAY)
    _, thresh_page = cv2.threshold(gray_page, 127, 255, cv2.THRESH_BINARY_INV)
    num_labels, labels_im, stats, _ = cv2.connectedComponentsWithStats(thresh_page)
    
    protected_labels = set()
    if result.obb is not None and len(result.obb) > 0:
        xyxyxyxy_page = result.obb.xyxyxyxy.cpu().numpy()
        yolo_mask = np.zeros(thresh_page.shape, dtype=np.uint8)
        for corners in xyxyxyxy_page:
            pts = np.array(corners, dtype=np.int32)
            cv2.fillPoly(yolo_mask, [pts], 255)
            
        font_threshold = 35
        for i in range(1, num_labels):
            w_island = stats[i, cv2.CC_STAT_WIDTH]
            h_island = stats[i, cv2.CC_STAT_HEIGHT]
            
            if w_island > font_threshold or h_island > font_threshold:
                island_mask = (labels_im == i)
                total_pixels = np.sum(island_mask)
                inside_pixels = np.sum(yolo_mask[island_mask] > 0)
                
                if total_pixels > 0 and (inside_pixels / total_pixels) > 0.85:
                    protected_labels.add(i)
                    
        print(f"Page-level protection: {len(protected_labels)} out of {num_labels - 1} islands protected (contain symbol components).")
    
    detections = []
    page_eval_metrics = []
    
    annotated_img = preprocessed_img.copy()
    draw = ImageDraw.Draw(annotated_img)
    
    font_choices = [
        "C:\\Windows\\Fonts\\seguisym.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/freefont/FreeSans.ttf",
        "DejaVuSans.ttf",
        "C:\\Windows\\Fonts\\arial.ttf",
        "C:\\Windows\\Fonts\\calibri.ttf",
        "C:\\Windows\\Fonts\\segoeui.ttf"
    ]
    font = None
    for path in font_choices:
        if os.path.exists(path):
            try:
                font = ImageFont.truetype(path, 20)
                break
            except:
                pass
    if font is None:
        font = ImageFont.load_default()
        
    if result.obb is not None and len(result.obb) > 0:
        xywhr = result.obb.xywhr.cpu().numpy()  # cx, cy, w, h, r
        xyxyxyxy = result.obb.xyxyxyxy.cpu().numpy()  # corners: TL, TR, BR, BL
        yolo_confs = result.obb.conf.cpu().numpy()
        
        # Create directories for reviews and crops
        review_dir = os.path.join(output_dir, "reviews")
        crop_dir = os.path.join(output_dir, "crops")
        os.makedirs(review_dir, exist_ok=True)
        os.makedirs(crop_dir, exist_ok=True)
        
        for idx in range(len(xywhr)):
            cx, cy, w, h, r = xywhr[idx]
            corners = xyxyxyxy[idx]
            yolo_conf = yolo_confs[idx]
            angle_deg = math.degrees(r)
            
            # Extract standard crop (20% buffer to prevent boundary clipping)
            crop_pil_std = rectify_crop(
                preprocessed_img,
                bbox_metrics={'cx': cx, 'cy': cy, 'w': w, 'h': h, 'angle': angle_deg},
                buffer_percent=0.20
            )
            
            # Extract bigger crop (40% buffer) for line tracing
            crop_pil_big = rectify_crop(
                preprocessed_img,
                bbox_metrics={'cx': cx, 'cy': cy, 'w': w, 'h': h, 'angle': angle_deg},
                buffer_percent=0.40
            )
            
            # Apply straight & curved line deletion and get review image
            bbox_metrics = {'cx': cx, 'cy': cy, 'w': w, 'h': h, 'angle': angle_deg}
            cleaned_crop, review_img, clean_details = clean_patch_lines(
                crop_pil_std, crop_pil_big,
                bbox_metrics=bbox_metrics,
                labels_im=labels_im,
                protected_labels=protected_labels
            )
            
            # Save crops for inspection
            crop_pil_std.save(os.path.join(crop_dir, f"crop_{idx}_original.png"))
            cleaned_crop.save(os.path.join(crop_dir, f"crop_{idx}_cleaned.png"))
            
            # Save reviewer image (Red = Text, Blue = Lines)
            cv2.imwrite(os.path.join(review_dir, f"crop_{idx}_review.png"), cv2.cvtColor(review_img, cv2.COLOR_RGB2BGR))
            
            # If gt_colored_pil is available, run evaluation
            evaluation_metrics = None
            if gt_colored_pil is not None:
                try:
                    # Rectify the colored ground truth crop (matching 20% buffer)
                    crop_pil_gt_col = rectify_crop(
                        gt_colored_pil,
                        bbox_metrics={'cx': cx, 'cy': cy, 'w': w, 'h': h, 'angle': angle_deg},
                        buffer_percent=0.20
                    )
                    
                    # Convert to RGB array
                    gt_rgb = np.array(crop_pil_gt_col.convert("RGB"))
                    
                    # Extract ground truth masks
                    # Pure Red for Text: R > 200, G < 100, B < 100
                    gt_text_mask = ((gt_rgb[:, :, 0] > 200) & (gt_rgb[:, :, 1] < 100) & (gt_rgb[:, :, 2] < 100)).astype(np.uint8) * 255
                    # Pure Blue for Line: R < 100, G < 100, B > 200
                    gt_line_mask = ((gt_rgb[:, :, 0] < 100) & (gt_rgb[:, :, 1] < 100) & (gt_rgb[:, :, 2] > 200)).astype(np.uint8) * 255
                    
                    # Formulate prediction masks
                    cleaned_np = np.array(cleaned_crop.convert("L"))
                    _, pred_text_mask = cv2.threshold(cleaned_np, 127, 255, cv2.THRESH_BINARY_INV)
                    
                    orig_np = np.array(crop_pil_std.convert("L"))
                    _, orig_thresh = cv2.threshold(orig_np, 127, 255, cv2.THRESH_BINARY_INV)
                    
                    # Intersect ground truth masks with orig_thresh to only evaluate on pixels present in the crop
                    gt_text_mask_clean = cv2.bitwise_and(gt_text_mask, orig_thresh)
                    gt_line_mask_clean = cv2.bitwise_and(gt_line_mask, orig_thresh)
                    
                    pred_line_mask = cv2.subtract(orig_thresh, pred_text_mask)
                    
                    # Run evaluate_cleaning
                    evaluation_metrics = evaluate_cleaning(
                        pred_text_mask, pred_line_mask, gt_text_mask_clean, gt_line_mask_clean
                    )
                    page_eval_metrics.append(evaluation_metrics)
                except Exception as e:
                    print(f"Warning: Failed to evaluate crop {idx} due to error: {e}")
            
            # Run MobileNetV3 character recognition on cleaned crop
            pred_text, class_conf, rectified_char_pil, char_details = classifier.predict_expression(cleaned_crop)
            
            det_info = {
                "idx": idx,
                "yolo_conf": float(yolo_conf),
                "text": pred_text,
                "confidence": float(class_conf),
                "rotation_degrees": float(angle_deg),
                "center": [float(cx), float(cy)],
                "size": [float(w), float(h)],
                "corners": corners.tolist(),
                "cleaning_details": clean_details
            }
            if evaluation_metrics is not None:
                det_info["evaluation"] = evaluation_metrics
                
            if class_conf >= 0.7:
                detections.append(det_info)
                
                # Draw OBB bounding box (green)
                poly_pts = [(float(pt[0]), float(pt[1])) for pt in corners]
                draw.polygon(poly_pts, outline=(0, 200, 0), width=3)
                
                # Draw predicted label (red text)
                lx, ly = float(corners[0][0]), float(corners[0][1])
                label_text = f"{pred_text} ({class_conf:.2f})"
                draw.text((lx, ly - 25), label_text, font=font, fill=(220, 0, 0))
                
    # Calculate page-level averages of evaluation metrics
    if page_eval_metrics:
        avg_tpr = np.mean([e["text_preservation_rate"] for e in page_eval_metrics])
        avg_ldr = np.mean([e["line_deletion_rate"] for e in page_eval_metrics])
        avg_f1 = np.mean([e["f1_score"] for e in page_eval_metrics])
        print("\n" + "="*50)
        print("PAGE-LEVEL LINE DELETION METRICS EVALUATION")
        print("="*50)
        print(f"Total Bboxes Evaluated: {len(page_eval_metrics)}")
        print(f"Average Text Preservation Rate (TPR): {avg_tpr*100:.2f}%")
        print(f"Average Line Deletion Rate (LDR):     {avg_ldr*100:.2f}%")
        print(f"Average Harmonic F1 Score:            {avg_f1*100:.2f}%")
        print("="*50 + "\n")
        
    return annotated_img, detections

def run_pipeline(input_path, yolo_path, classifier_path, output_dir, conf_threshold=0.25, poppler_path=None):
    os.makedirs(output_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Running pipeline on device: {device}")
    
    yolo_model = load_yolo_model(yolo_path)
    classifier = SymbolClassifier(model_path=classifier_path, device=device)
    
    pages = []
    is_pdf = input_path.lower().endswith(".pdf")
    
    if is_pdf:
        print(f"Reading PDF from: {input_path} (converting at 300 DPI)...")
        try:
            from pdf2image import convert_from_path
            kwargs = {"dpi": 300}
            if poppler_path:
                kwargs["poppler_path"] = poppler_path
            pages = convert_from_path(input_path, **kwargs)
            print(f"Successfully loaded {len(pages)} pages from PDF.")
        except ImportError:
            print("ERROR: 'pdf2image' is required to process PDF files.")
            return
        except Exception as e:
            print(f"ERROR: Failed to load PDF pages. Details: {e}")
            return
    else:
        print(f"Reading Image from: {input_path}")
        try:
            pages = [Image.open(input_path)]
        except Exception as e:
            print(f"ERROR: Failed to open image. Details: {e}")
            return
            
    all_detections = []
    annotated_pages = []
    base_name = os.path.splitext(os.path.basename(input_path))[0]
    
    for i, page_img in enumerate(pages):
        print(f"Processing page {i + 1}/{len(pages)}...")
        gt_colored_img = find_gt_colored_page(input_path, i)
        if gt_colored_img is not None:
            print(f"Loaded corresponding ground truth colored image for page {i + 1}.")
        annotated_img, page_dets = process_page(
            page_img, yolo_model, classifier, conf_threshold, output_dir, gt_colored_pil=gt_colored_img
        )
        
        page_save_path = os.path.join(output_dir, f"{base_name}_page_{i}_annotated.png")
        annotated_img.save(page_save_path)
        print(f"Saved annotated page image to: {page_save_path}")
        
        annotated_pages.append(annotated_img)
        all_detections.append({
            "page": i + 1,
            "detections": page_dets
        })
        
    json_path = os.path.join(output_dir, f"{base_name}_results.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(all_detections, f, indent=4)
    print(f"Saved JSON detection report to: {json_path}")
    
    if is_pdf and len(annotated_pages) > 0:
        pdf_save_path = os.path.join(output_dir, f"{base_name}_annotated.pdf")
        annotated_pages[0].save(
            pdf_save_path,
            save_all=True,
            append_images=annotated_pages[1:],
            format="PDF"
        )
        print(f"Saved compiled annotated PDF to: {pdf_save_path}")
        
    print("\n--- Pipeline execution complete! ---")
    print(f"Total pages processed: {len(pages)}")
    total_dets = sum(len(p["detections"]) for p in all_detections)
    print(f"Total symbols detected and recognized: {total_dets}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Orchestrator Pipeline for Blueprint Symbol Recognition")
    parser.add_argument("--input", required=True, help="Path to input image or PDF file")
    parser.add_argument("--yolo", default="../sign-detection-yolo8n/YOLO_expression_best.pt", help="Path to trained YOLOv8-OBB model (.pt)")
    parser.add_argument("--classifier", default="../sign-detection-yolo8n/classifier_best.pt", help="Path to trained MobileNetV3 classifier model (.pt)")
    parser.add_argument("--output", default="output_pipeline", help="Directory to save annotated outputs and JSON")
    parser.add_argument("--conf", type=float, default=0.25, help="Confidence threshold for YOLOv8 detection")
    parser.add_argument("--poppler-path", default=None, help="Windows path to poppler bin folder (optional)")
    args = parser.parse_args()
    
    run_pipeline(
        input_path=args.input,
        yolo_path=args.yolo,
        classifier_path=args.classifier,
        output_dir=args.output,
        conf_threshold=args.conf,
        poppler_path=args.poppler_path
    )
