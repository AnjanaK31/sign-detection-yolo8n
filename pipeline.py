import os
import argparse
import json
import torch
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from ultralytics import YOLO

# Import custom rectifier
from rectifier import rectify_crop

def load_yolo_model(yolo_path):
    """Loads the YOLOv8-OBB model."""
    print(f"Loading YOLOv8-OBB model from: {yolo_path}")
    return YOLO(yolo_path)

def process_image(img_pil, yolo_model, ocr_reader, conf_threshold=0.25):
    """Processes a single page image: runs OBB detection, rectifies, OCRs in four rotations, and annotates."""
    img_np = np.array(img_pil.convert("RGB"))
    # YOLO expectations: BGR image for OpenCV operations
    img_bgr = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
    
    # Run YOLOv8-OBB inference
    results = yolo_model(img_bgr, verbose=False, conf=conf_threshold)
    result = results[0]
    
    detections = []
    
    # Create PIL Image copy for drawing annotations (to support Unicode fonts)
    annotated_img = img_pil.convert("RGB")
    draw = ImageDraw.Draw(annotated_img)
    
    # Try to load a unicode font
    font_choices = [
        "C:\\Windows\\Fonts\\arial.ttf",
        "C:\\Windows\\Fonts\\calibri.ttf",
        "C:\\Windows\\Fonts\\segoeui.ttf"
    ]
    font = None
    for path in font_choices:
        if os.path.exists(path):
            try:
                font = ImageFont.truetype(path, 24)
                break
            except:
                pass
    if font is None:
        font = ImageFont.load_default()
    
    if result.obb is not None:
        # Get corner coordinates, class indices, and confidences from YOLO
        # xyxyxyxy shape: (N, 4, 2)
        boxes_corners = result.obb.xyxyxyxy.cpu().numpy()
        yolo_confs = result.obb.conf.cpu().numpy()
        
        for idx in range(len(boxes_corners)):
            corners = boxes_corners[idx] # (4, 2) -> TL, TR, BR, BL
            yolo_conf = yolo_confs[idx]
            
            # Calculate dynamic bounding box width and height
            w_val = np.linalg.norm(corners[0] - corners[1])
            h_val = np.linalg.norm(corners[0] - corners[3])
            tw = int(max(w_val, 1))
            th = int(max(h_val, 1))
            
            # 1. Rectify/Deskew the crop using Affine transformation
            crop_pil = rectify_crop(img_pil, corners, target_size=(tw, th))
            
            # Ensure it is RGB
            if crop_pil.mode != "RGB":
                crop_pil = crop_pil.convert("RGB")
            
            # 2. OCR in four rotations to find best orientation
            best_text = ""
            best_conf = -1.0
            best_rot = 0
            
            rotations = [0, 90, 180, 270]
            for rot in rotations:
                if rot == 0:
                    rot_img = crop_pil
                else:
                    rot_img = crop_pil.rotate(rot, expand=True)
                
                rot_np = np.array(rot_img)
                ocr_results = ocr_reader.readtext(rot_np)
                
                if ocr_results:
                    texts = [res[1] for res in ocr_results]
                    confs = [res[2] for res in ocr_results]
                    mean_conf = sum(confs) / len(confs)
                    full_text = " ".join(texts)
                    
                    if mean_conf > best_conf:
                        best_conf = mean_conf
                        best_text = full_text
                        best_rot = rot
            
            # Fallback if no text detected in any orientation
            if best_conf < 0:
                best_text = "UNKNOWN"
                best_conf = 0.0
                best_rot = 0
            
            detections.append({
                "yolo_conf": float(yolo_conf),
                "ocr_text": best_text,
                "ocr_confidence": float(best_conf),
                "detected_rotation": best_rot,
                "corners": corners.tolist()
            })
            
            # 3. Draw annotations
            # Draw oriented box outline (in green) using Pillow polygon
            poly_pts = [(float(pt[0]), float(pt[1])) for pt in corners]
            draw.polygon(poly_pts, outline=(0, 200, 0), width=4)
            
            # Draw label (in red) using Pillow text
            lx, ly = float(corners[0][0]), float(corners[0][1])
            label_text = f"{best_text} ({best_conf:.2f})"
            draw.text((lx, ly - 35), label_text, font=font, fill=(220, 0, 0))
            
    return annotated_img, detections

def run_pipeline(input_path, yolo_path, output_dir, conf_threshold=0.25, poppler_path=None):
    """Main orchestration function to run the full document equation detection and OCR pipeline."""
    os.makedirs(output_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Running pipeline on device: {device}")
    
    yolo_model = load_yolo_model(yolo_path)
    
    # Initialize EasyOCR
    import easyocr
    print("Initializing EasyOCR reader...")
    ocr_reader = easyocr.Reader(['en'], gpu=torch.cuda.is_available())
    
    # 1. Determine file type and load pages
    pages = []
    is_pdf = input_path.lower().endswith(".pdf")
    
    if is_pdf:
        print(f"Reading PDF from: {input_path}")
        try:
            from pdf2image import convert_from_path
            kwargs = {}
            if poppler_path:
                kwargs["poppler_path"] = poppler_path
            pages = convert_from_path(input_path, **kwargs)
            print(f"Successfully loaded {len(pages)} pages from PDF.")
        except ImportError:
            print("ERROR: 'pdf2image' is required to process PDF files. Please install it.")
            return
        except Exception as e:
            print(f"ERROR: Failed to load PDF pages. Details: {e}")
            print("Make sure poppler is installed and added to PATH (or specify --poppler-path).")
            return
    else:
        # Load as a single image
        print(f"Reading Image from: {input_path}")
        try:
            pages = [Image.open(input_path)]
        except Exception as e:
            print(f"ERROR: Failed to open image. Details: {e}")
            return
            
    # 2. Process all pages
    all_detections = []
    annotated_pages = []
    
    for i, page_img in enumerate(pages):
        print(f"Processing page {i + 1}/{len(pages)}...")
        annotated_img, page_dets = process_image(page_img, yolo_model, ocr_reader, conf_threshold)
        
        # Save annotated page image
        base_name = os.path.splitext(os.path.basename(input_path))[0]
        page_save_path = os.path.join(output_dir, f"{base_name}_page_{i}_annotated.png")
        annotated_img.save(page_save_path)
        print(f"Saved annotated page image to: {page_save_path}")
        
        annotated_pages.append(annotated_img)
        all_detections.append({
            "page": i + 1,
            "detections": page_dets
        })
        
    # 3. Save JSON results summary
    json_path = os.path.join(output_dir, f"{base_name}_results.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(all_detections, f, indent=4)
    print(f"Saved JSON detection report to: {json_path}")
    
    # 4. If PDF, compile annotated pages back into a PDF
    if is_pdf and len(annotated_pages) > 0:
        pdf_save_path = os.path.join(output_dir, f"{base_name}_annotated.pdf")
        annotated_pages[0].save(
            pdf_save_path, 
            save_all=True, 
            append_images=annotated_pages[1:]
        )
        print(f"Saved compiled annotated PDF to: {pdf_save_path}")
        
    print("\n--- Pipeline execution complete! ---")
    print(f"Total pages processed: {len(pages)}")
    total_dets = sum(len(p["detections"]) for p in all_detections)
    print(f"Total equations detected and OCR'd: {total_dets}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Orchestrator Pipeline for Equation Detection and OCR")
    parser.add_argument("--input", required=True, help="Path to input image or PDF file")
    parser.add_argument("--yolo", default="yolo_obb_project/symbol_obb_train/weights/best.pt", help="Path to trained YOLOv8-OBB model (.pt)")
    parser.add_argument("--output", default="output_pipeline", help="Directory to save annotated outputs and JSON")
    parser.add_argument("--conf", type=float, default=0.25, help="Confidence threshold for YOLOv8 detection")
    parser.add_argument("--poppler-path", default=None, help="Windows path to poppler bin folder (optional)")
    args = parser.parse_args()
    
    if not os.path.exists(args.yolo):
        print(f"WARNING: Trained YOLO weights not found at '{args.yolo}'. Run train_yolo.py first.")
        
    run_pipeline(
        input_path=args.input,
        yolo_path=args.yolo,
        output_dir=args.output,
        conf_threshold=args.conf,
        poppler_path=args.poppler_path
    )
