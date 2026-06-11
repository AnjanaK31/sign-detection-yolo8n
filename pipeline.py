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

# Class to character mapping for display
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

def load_yolo_model(yolo_path):
    print(f"Loading YOLOv8-OBB model from: {yolo_path}")
    return YOLO(yolo_path)

def preprocess_image(img_pil):
    """Phase 1: Converts image to Grayscale and applies Adaptive Thresholding."""
    img_np = np.array(img_pil.convert("RGB"))
    gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
    thresh = cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2
    )
    return Image.fromarray(cv2.cvtColor(thresh, cv2.COLOR_GRAY2RGB))

def process_page(img_pil, yolo_model, classifier, conf_threshold=0.25):
    """Processes a single page: preprocesses, runs YOLO-OBB, rectifies, classifies, and annotates."""
    # Phase 1 Preprocessing
    preprocessed_img = preprocess_image(img_pil)
    
    # YOLO expectations: BGR image
    img_np = np.array(preprocessed_img)
    img_bgr = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
    
    # Run YOLOv8-OBB inference (imgsz=1280 per requirements)
    results = yolo_model(img_bgr, verbose=False, conf=conf_threshold, imgsz=1280)
    result = results[0]
    
    detections = []
    
    # Create PIL Image copy for drawing annotations (to support Unicode fonts)
    annotated_img = preprocessed_img.copy()
    draw = ImageDraw.Draw(annotated_img)
    
    # Load unicode font
    font_choices = [
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
        # Get coordinates in xywhr and xyxyxyxy
        xywhr = result.obb.xywhr.cpu().numpy()  # cx, cy, w, h, r
        xyxyxyxy = result.obb.xyxyxyxy.cpu().numpy()  # corners: TL, TR, BR, BL
        yolo_confs = result.obb.conf.cpu().numpy()
        
        for idx in range(len(xywhr)):
            cx, cy, w, h, r = xywhr[idx]
            corners = xyxyxyxy[idx]
            yolo_conf = yolo_confs[idx]
            angle_deg = math.degrees(r)
            
            # Phase 3 Rectification: use cv2.getRotationMatrix2D and cv2.warpAffine with 5-10% buffer
            crop_pil = rectify_crop(
                preprocessed_img,
                bbox_metrics={'cx': cx, 'cy': cy, 'w': w, 'h': h, 'angle': angle_deg},
                buffer_percent=0.08
            )
            
            # Diagnostic: Save the rectified crop to inspect what the classifier sees
            debug_dir = "debug_crops"
            os.makedirs(debug_dir, exist_ok=True)
            crop_save_path = os.path.join(debug_dir, f"crop_idx_{idx}_ang_{int(angle_deg)}.png")
            crop_pil.save(crop_save_path)
            
            # Phase 4 Character Recognition: MobileNetV3
            class_name, class_conf = classifier.predict(crop_pil)
            
            # Filtering: Apply a 0.9 confidence floor to exclude low-probability results
            if class_conf >= 0.9:
                char_display = CLASS_TO_CHAR.get(class_name, class_name)
                
                detections.append({
                    "yolo_conf": float(yolo_conf),
                    "pred_class": class_name,
                    "pred_char": char_display,
                    "class_confidence": float(class_conf),
                    "rotation_degrees": float(angle_deg),
                    "center": [float(cx), float(cy)],
                    "size": [float(w), float(h)],
                    "corners": corners.tolist()
                })
                
                # Draw OBB bounding box (green)
                poly_pts = [(float(pt[0]), float(pt[1])) for pt in corners]
                draw.polygon(poly_pts, outline=(0, 200, 0), width=3)
                
                # Draw predicted label (red text)
                lx, ly = float(corners[0][0]), float(corners[0][1])
                label_text = f"{char_display} ({class_conf:.2f})"
                draw.text((lx, ly - 25), label_text, font=font, fill=(220, 0, 0))
                
    return annotated_img, detections

def run_pipeline(input_path, yolo_path, classifier_path, output_dir, conf_threshold=0.25, poppler_path=None):
    """Orchestrates the complete PDF/Image character extraction and recognition pipeline."""
    os.makedirs(output_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Running pipeline on device: {device}")
    
    # Load models
    yolo_model = load_yolo_model(yolo_path)
    classifier = SymbolClassifier(model_path=classifier_path, device=device)
    
    # 1. Phase 1: Load pages (PDF to high quality image at min 300 DPI)
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
            print(f"Successfully loaded {len(pages)} pages from PDF at 300 DPI.")
        except ImportError:
            print("ERROR: 'pdf2image' is required to process PDF files. Please install it.")
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
            
    # 2. Process all pages
    all_detections = []
    annotated_pages = []
    
    base_name = os.path.splitext(os.path.basename(input_path))[0]
    
    for i, page_img in enumerate(pages):
        print(f"Processing page {i + 1}/{len(pages)}...")
        annotated_img, page_dets = process_page(page_img, yolo_model, classifier, conf_threshold)
        
        # Save preprocessed & annotated page
        page_save_path = os.path.join(output_dir, f"{base_name}_page_{i}_annotated.png")
        annotated_img.save(page_save_path)
        print(f"Saved annotated page image to: {page_save_path}")
        
        annotated_pages.append(annotated_img)
        all_detections.append({
            "page": i + 1,
            "detections": page_dets
        })
        
    # 3. Save JSON results report
    json_path = os.path.join(output_dir, f"{base_name}_results.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(all_detections, f, indent=4)
    print(f"Saved JSON detection report to: {json_path}")
    
    # 4. If PDF input, compile annotated pages back into PDF
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
    parser.add_argument("--yolo", default="yolo_obb_project/symbol_obb_train/weights/best.pt", help="Path to trained YOLOv8-OBB model (.pt)")
    parser.add_argument("--classifier", default="classifier_best.pt", help="Path to trained MobileNetV3 classifier model (.pt)")
    parser.add_argument("--output", default="output_pipeline", help="Directory to save annotated outputs and JSON")
    parser.add_argument("--conf", type=float, default=0.25, help="Confidence threshold for YOLOv8 detection")
    parser.add_argument("--poppler-path", default=None, help="Windows path to poppler bin folder (optional)")
    args = parser.parse_args()
    
    if not os.path.exists(args.yolo):
        print(f"WARNING: Trained YOLO weights not found at '{args.yolo}'. Run train_yolo.py first.")
    if not os.path.exists(args.classifier):
        print(f"WARNING: Trained MobileNetV3 classifier weights not found at '{args.classifier}'. Run train_classifier.py first.")
        
    run_pipeline(
        input_path=args.input,
        yolo_path=args.yolo,
        classifier_path=args.classifier,
        output_dir=args.output,
        conf_threshold=args.conf,
        poppler_path=args.poppler_path
    )
