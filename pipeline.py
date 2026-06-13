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
from classifier import SymbolClassifier, CLASSES, pil_to_base64

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

def group_characters_into_expressions(detections):
    """
    Groups individual character detections into expressions based on geometric proximity and alignment.
    Detections should be a list of dicts, each representing a character detection:
    {
        'id': int,
        'yolo_conf': float,
        'pred_char': str,
        'class_confidence': float,
        'rotation_degrees': float,
        'center': [cx, cy],
        'size': [w, h],
        'corners': [[x1, y1], [x2, y2], [x3, y3], [x4, y4]],
        'char_crop_b64': str
    }
    Returns a list of grouped detections matching the expression schema.
    """
    if not detections:
        return []

    num_dets = len(detections)
    adj = {i: [] for i in range(num_dets)}

    for i in range(num_dets):
        for j in range(i + 1, num_dets):
            det_a = detections[i]
            det_b = detections[j]

            # 1. Angle compatibility (modulo 180 degrees)
            theta_a = det_a['rotation_degrees']
            theta_b = det_b['rotation_degrees']
            
            angle_diff = abs(theta_a - theta_b) % 180
            angle_diff = min(angle_diff, 180 - angle_diff)

            if angle_diff > 15.0:
                continue

            # Use average angle for directional projections
            avg_angle_rad = math.radians((theta_a + theta_b) / 2.0)
            cos_a = math.cos(avg_angle_rad)
            sin_a = math.sin(avg_angle_rad)

            # Direction vectors (u along the line, v perpendicular to the line)
            u = np.array([cos_a, sin_a])
            v = np.array([-sin_a, cos_a])

            # Vector from center A to center B
            ca = np.array(det_a['center'])
            cb = np.array(det_b['center'])
            d_vec = cb - ca

            # Projected distances
            d_parallel = abs(np.dot(d_vec, u))
            d_perp = abs(np.dot(d_vec, v))

            # Character sizes
            w_a, h_a = det_a['size']
            w_b, h_b = det_b['size']
            max_w = max(w_a, w_b)
            max_h = max(h_a, h_b)

            # Alignment criteria:
            # - Perpendicular distance (offset) must be small relative to character height
            # - Parallel distance (separation) must be small relative to character width
            perp_thresh = max_h * 0.6
            parallel_thresh = max_w * 2.2

            if d_perp <= perp_thresh and d_parallel <= parallel_thresh:
                adj[i].append(j)
                adj[j].append(i)

    # Find connected components
    visited = set()
    components = []

    for i in range(num_dets):
        if i not in visited:
            comp = []
            queue = [i]
            visited.add(i)
            while queue:
                curr = queue.pop(0)
                comp.append(curr)
                for neighbor in adj[curr]:
                    if neighbor not in visited:
                        visited.add(neighbor)
                        queue.append(neighbor)
            components.append(comp)

    grouped_detections = []

    for comp in components:
        comp_dets = [detections[idx] for idx in comp]

        if len(comp_dets) == 1:
            det = comp_dets[0]
            grouped_detections.append({
                "id": len(grouped_detections),
                "yolo_conf": det["yolo_conf"],
                "pred_class": det["pred_char"],
                "pred_char": det["pred_char"],
                "class_confidence": det["class_confidence"],
                "rotation_degrees": det["rotation_degrees"],
                "center": det["center"],
                "size": det["size"],
                "corners": det["corners"],
                "char_details": [{
                    "char": det["pred_char"],
                    "confidence": det["class_confidence"],
                    "image": det.get("char_crop_b64", "")
                }]
            })
            continue

        # Multiple characters: sort along the text line
        avg_angle = np.mean([d['rotation_degrees'] for d in comp_dets])
        avg_angle_rad = math.radians(avg_angle)
        cos_a = math.cos(avg_angle_rad)
        sin_a = math.sin(avg_angle_rad)
        u = np.array([cos_a, sin_a])
        v = np.array([-sin_a, cos_a])

        # Project centers and sort left-to-right
        projected = []
        for det in comp_dets:
            proj_val = np.dot(np.array(det['center']), u)
            projected.append((proj_val, det))

        projected.sort(key=lambda x: x[0])
        sorted_dets = [p[1] for p in projected]

        # Concatenate string values
        expr_str = "".join([d["pred_char"] for d in sorted_dets])
        avg_conf = np.mean([d["class_confidence"] for d in sorted_dets])
        avg_yolo_conf = np.mean([d["yolo_conf"] for d in sorted_dets])

        # Compute oriented bounding box enclosing all character boxes
        all_pts = []
        for d in sorted_dets:
            all_pts.extend(d['corners'])
        all_pts = np.array(all_pts)

        proj_u = np.dot(all_pts, u)
        proj_v = np.dot(all_pts, v)

        min_u, max_u = np.min(proj_u), np.max(proj_u)
        min_v, max_v = np.min(proj_v), np.max(proj_v)

        mid_u = (min_u + max_u) / 2.0
        mid_v = (min_v + max_v) / 2.0
        center_pt = mid_u * u + mid_v * v

        w_expr = max_u - min_u
        h_expr = max_v - min_v

        # Corners: TL, TR, BR, BL
        half_w = w_expr / 2.0
        half_h = h_expr / 2.0
        corners_expr = [
            (center_pt - half_w * u - half_h * v).tolist(),
            (center_pt + half_w * u - half_h * v).tolist(),
            (center_pt + half_w * u + half_h * v).tolist(),
            (center_pt - half_w * u + half_h * v).tolist()
        ]

        char_details = []
        for d in sorted_dets:
            char_details.append({
                "char": d["pred_char"],
                "confidence": d["class_confidence"],
                "image": d.get("char_crop_b64", "")
            })

        grouped_detections.append({
            "id": len(grouped_detections),
            "yolo_conf": float(avg_yolo_conf),
            "pred_class": expr_str,
            "pred_char": expr_str,
            "class_confidence": float(avg_conf),
            "rotation_degrees": float(avg_angle),
            "center": [float(center_pt[0]), float(center_pt[1])],
            "size": [float(w_expr), float(h_expr)],
            "corners": corners_expr,
            "char_details": char_details
        })

    return grouped_detections

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
    
    temp_detections = []
    
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
            
            # Phase 4 Character Recognition: MobileNetV3 (predict single character)
            class_name, char_conf = classifier.predict(crop_pil)
            char_display = CLASS_TO_CHAR.get(class_name, class_name)
            char_crop_b64 = pil_to_base64(crop_pil)
            
            temp_detections.append({
                "yolo_conf": float(yolo_conf),
                "pred_char": char_display,
                "class_confidence": float(char_conf),
                "rotation_degrees": float(angle_deg),
                "center": [float(cx), float(cy)],
                "size": [float(w), float(h)],
                "corners": corners.tolist(),
                "char_crop_b64": char_crop_b64
            })
            
    # Group the individual character detections into expressions
    detections = group_characters_into_expressions(temp_detections)
    
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
        
    # Annotate grouped expressions and slice their visual crop images
    for idx, expr in enumerate(detections):
        corners = expr["corners"]
        expr_str = expr["pred_char"]
        expr_conf = expr["class_confidence"]
        
        # Rectify the full expression region to get a clean visual crop of the expression
        expr_crop_pil = rectify_crop(
            preprocessed_img,
            pts=np.array(corners, dtype=np.float32),
            buffer_percent=0.08
        )
        expr["crop_image_b64"] = pil_to_base64(expr_crop_pil)
        
        # Draw OBB bounding box (green)
        poly_pts = [(float(pt[0]), float(pt[1])) for pt in corners]
        draw.polygon(poly_pts, outline=(0, 200, 0), width=3)
        
        # Draw predicted label (red text)
        lx, ly = float(corners[0][0]), float(corners[0][1])
        label_text = f"{expr_str} ({expr_conf:.2f})"
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
    # Dynamically find the best trained YOLO model path
    default_yolo = "runs/obb/runs/obb/trained_on_1000_pdfs-2/weights/best.pt"
    if not os.path.exists(default_yolo):
        default_yolo = "runs/obb/yolo_obb_project/symbol_obb_train-2/weights/best.pt"
    if not os.path.exists(default_yolo):
        default_yolo = "runs/obb/yolo_obb_project/symbol_obb_train/weights/best.pt"
    if not os.path.exists(default_yolo):
        default_yolo = "yolo_obb_project/symbol_obb_train/weights/best.pt"

    default_classifier = "classifier_best_bigdatset.pt"
    if not os.path.exists(default_classifier):
        default_classifier = "classifier_best.pt"

    parser = argparse.ArgumentParser(description="Orchestrator Pipeline for Blueprint Symbol Recognition")
    parser.add_argument("--input", required=True, help="Path to input image or PDF file")
    parser.add_argument("--yolo", default=default_yolo, help="Path to trained YOLOv8-OBB model (.pt)")
    parser.add_argument("--classifier", default=default_classifier, help="Path to trained MobileNetV3 classifier model (.pt)")
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
