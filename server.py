import os
import json
import base64
import asyncio
import io
import math
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import torch
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from shapely.geometry import Polygon

# Import existing pipeline components
from pipeline import load_yolo_model, preprocess_image
from rectifier import rectify_crop
from classifier import SymbolClassifier, CLASSES
from pipeline import CLASS_TO_CHAR
from preprocessor import full_preprocess, preprocess_stages
from visualize_pipeline import get_padded_crop

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.on_event("startup")
def startup_event():
    print("Loading models during startup...")
    get_models()
    print("Models loaded successfully!")

# Global models (loaded on startup or first request)
yolo_model = None
classifier = None

# Cache for rendered PDF pages to avoid re-rendering
# Format: { filename: [ PIL.Image, ... ] }
pdf_pages_cache = {}

# Cache for ground truth database to avoid reloading 100MB JSON on each request
gt_db = None

def get_gt_db():
    global gt_db
    if gt_db is None:
        gt_db_path = os.path.join("dataset_yolo", "ground_truth.json")
        if os.path.exists(gt_db_path):
            try:
                print("Loading ground truth database in memory...")
                with open(gt_db_path, "r") as f:
                    gt_db = json.load(f)
                print("Ground truth database loaded!")
            except Exception as e:
                print(f"Error loading ground truth database: {e}")
                gt_db = {}
        else:
            gt_db = {}
    return gt_db

def get_angle_from_corners(corners):
    """Calculates box rotation angle in degrees from its 4 corners (TL, TR, BR, BL order)."""
    p1 = corners[0]
    p2 = corners[1]
    dx = p2[0] - p1[0]
    dy = p2[1] - p1[1]
    angle = math.degrees(math.atan2(dy, dx))
    return angle


# Cache for YOLO boxes to avoid running YOLO multiple times
# Format: { "filename_pagenum": [ boxes ] }
yolo_boxes_cache = {}

def get_models():
    global yolo_model, classifier
    if yolo_model is None:
        yolo_path = "weights/best.pt"
        if not os.path.exists(yolo_path):
            yolo_path = "runs/obb/runs/obb/trained_on_1000_pdfs-2/weights/best.pt"
        if not os.path.exists(yolo_path):
            yolo_path = "runs/obb/yolo_obb_project/symbol_obb_train-2/weights/best.pt"
        if not os.path.exists(yolo_path):
            yolo_path = "runs/obb/yolo_obb_project/symbol_obb_train/weights/best.pt"
        yolo_model = load_yolo_model(yolo_path)
    if classifier is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        clf_path = "classifier_best_bigdatset.pt"
        if not os.path.exists(clf_path):
            clf_path = "classifier_best.pt"
        classifier = SymbolClassifier(model_path=clf_path, device=device)
    return yolo_model, classifier

def pil_to_base64(img: Image.Image, format="PNG") -> str:
    buffered = io.BytesIO()
    img.save(buffered, format=format)
    img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")
    return f"data:image/{format.lower()};base64,{img_str}"

def load_pdf_pages(file_path: str, dpi: int = 150):
    # Try fitz (PyMuPDF) first as it is much faster on Windows and has no poppler dependency
    try:
        import fitz
        doc = fitz.open(file_path)
        pages = []
        zoom = dpi / 72
        mat = fitz.Matrix(zoom, zoom)
        for page_idx in range(len(doc)):
            page = doc[page_idx]
            pix = page.get_pixmap(matrix=mat, colorspace=fitz.csRGB)
            img_data = pix.tobytes("png")
            img = Image.open(io.BytesIO(img_data)).convert("RGB")
            pages.append(img)
        doc.close()
        print(f"Successfully loaded {len(pages)} pages using fitz.")
        return pages
    except Exception as e:
        print(f"fitz loading failed, falling back to pdf2image: {e}")
        from pdf2image import convert_from_path
        return convert_from_path(file_path, dpi=dpi)

@app.get("/pdfs")
def list_pdfs():
    pdfs_dir = "pdfs"
    if not os.path.exists(pdfs_dir):
        return {"pdfs": []}
    files = [f for f in os.listdir(pdfs_dir) if f.lower().endswith(".pdf")]
    return {"pdfs": sorted(files)}

@app.get("/pdf/load")
def load_pdf(filename: str):
    global pdf_pages_cache
    if not filename:
        return {"error": "No filename provided"}
        
    file_path = os.path.join("pdfs", filename)
    if not os.path.exists(file_path):
        return {"error": "File not found"}
        
    try:
        if filename in pdf_pages_cache:
            pages = pdf_pages_cache[filename]
        else:
            pages = load_pdf_pages(file_path, dpi=150)
            pdf_pages_cache[filename] = pages
            
        pages_meta = []
        for i, page_img in enumerate(pages):
            pages_meta.append({
                "page_num": i + 1,
                "width": page_img.width,
                "height": page_img.height,
                "image": pil_to_base64(page_img, format="JPEG")
            })
            
        return {
            "filename": filename,
            "total_pages": len(pages),
            "pages": pages_meta
        }
    except Exception as e:
        return {"error": str(e)}

@app.get("/pdf/page/yolo")
def run_yolo_step(filename: str, page_num: int):
    global pdf_pages_cache, yolo_boxes_cache
    if not filename or not page_num:
        return {"error": "Filename and page_num are required"}
        
    if filename not in pdf_pages_cache:
        file_path = os.path.join("pdfs", filename)
        if not os.path.exists(file_path):
            return {"error": "File not found"}
        pdf_pages_cache[filename] = load_pdf_pages(file_path, dpi=150)
        
    pages = pdf_pages_cache[filename]
    if page_num < 1 or page_num > len(pages):
        return {"error": f"Invalid page number {page_num} (total pages: {len(pages)})"}
        
    page_img = pages[page_num - 1]
    
    # Preprocess
    preprocessed_img = full_preprocess(page_img)
    img_np = np.array(preprocessed_img)
    img_bgr = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
    
    yolo, _ = get_models()
    results = yolo(img_bgr, verbose=False, conf=0.25, imgsz=1280)
    result = results[0]
    
    boxes_data = []
    if result.obb is not None and len(result.obb) > 0:
        xywhr = result.obb.xywhr.cpu().numpy()
        xyxyxyxy = result.obb.xyxyxyxy.cpu().numpy()
        yolo_confs = result.obb.conf.cpu().numpy()
        
        for idx in range(len(xywhr)):
            cx, cy, w, h, r = xywhr[idx]
            corners = xyxyxyxy[idx]
            yolo_conf = yolo_confs[idx]
            angle_deg = math.degrees(r)
            
            boxes_data.append({
                "id": idx,
                "cx": float(cx),
                "cy": float(cy),
                "w": float(w),
                "h": float(h),
                "angle": float(angle_deg),
                "corners": corners.tolist(),
                "conf": float(yolo_conf)
            })
            
    cache_key = f"{filename}_{page_num}"
    yolo_boxes_cache[cache_key] = boxes_data
    
    return {
        "filename": filename,
        "page_num": page_num,
        "boxes": boxes_data
    }

@app.get("/pdf/page/process_crops")
def process_crops_step(filename: str, page_num: int):
    global pdf_pages_cache, yolo_boxes_cache
    if not filename or not page_num:
        return {"error": "Filename and page_num are required"}
        
    if filename not in pdf_pages_cache:
        return {"error": "PDF pages not loaded. Run load PDF first."}
        
    pages = pdf_pages_cache[filename]
    page_img = pages[page_num - 1]
    
    cache_key = f"{filename}_{page_num}"
    if cache_key not in yolo_boxes_cache:
        run_yolo_step(filename, page_num)
        
    boxes = yolo_boxes_cache.get(cache_key, [])
    
    # Morphological stages
    raw_np, gray_np, thresh_np, cleaned_np = preprocess_stages(page_img)
    cleaned_pil = Image.fromarray(cv2.cvtColor(cleaned_np, cv2.COLOR_GRAY2RGB))
    
    # Find matching ground truth key
    pdf_base = os.path.splitext(filename)[0]
    expected_keys = [
        f"{pdf_base}_page_{page_num}.png",
        f"{pdf_base}_page_{page_num}.jpg",
        f"{pdf_base}_page_{page_num}.jpeg",
        f"{pdf_base}_page_{page_num}",
        f"{pdf_base}"
    ]
    
    gt_list = []
    matched_key = None
    
    gt_db = get_gt_db()
    for ek in expected_keys:
        if ek in gt_db:
            gt_list = gt_db[ek]
            matched_key = ek
            break
            
    if not matched_key:
        # Fallback to search keys containing both the pdf_base and page_num
        for key in gt_db.keys():
            key_base = os.path.splitext(key)[0]
            if pdf_base in key_base and f"page_{page_num}" in key_base:
                gt_list = gt_db[key]
                matched_key = key
                break
                
    scale_x, scale_y = 1.0, 1.0
    if matched_key:
        # Search for the original image in the workspace to get its dimensions
        orig_img_path = None
        for root, dirs, files in os.walk("dataset_yolo"):
            if matched_key in files:
                orig_img_path = os.path.join(root, matched_key)
                break
        
        if orig_img_path:
            try:
                with Image.open(orig_img_path) as orig_img:
                    orig_w, orig_h = orig_img.width, orig_img.height
                if orig_w != page_img.width or orig_h != page_img.height:
                    scale_x = page_img.width / orig_w
                    scale_y = page_img.height / orig_h
                    print(f"Scaling ground truth coordinates from {orig_w}x{orig_h} to {page_img.width}x{page_img.height} (Scale X: {scale_x:.3f}, Y: {scale_y:.3f})")
            except Exception as e:
                print(f"Warning: Could not check original image dimensions for scaling: {e}")
                
    _, cls_model = get_models()
    
    detections = []
    
    for box in boxes:
        cx = box["cx"]
        cy = box["cy"]
        w = box["w"]
        h = box["h"]
        angle_deg = box["angle"]
        corners = box["corners"]
        yolo_conf = box["conf"]
        
        # 1. Extract raw padded crop
        raw_crop_np = get_padded_crop(raw_np, cx, cy, w, h, pad_factor=2.0)
        raw_crop_pil = Image.fromarray(raw_crop_np)
        
        # 1.5. Extract gray padded crop
        gray_crop_np = get_padded_crop(cv2.cvtColor(gray_np, cv2.COLOR_GRAY2RGB), cx, cy, w, h, pad_factor=2.0)
        gray_crop_pil = Image.fromarray(gray_crop_np)
        
        # 2. Extract thresholded padded crop
        thresh_crop_np = get_padded_crop(cv2.cvtColor(thresh_np, cv2.COLOR_GRAY2RGB), cx, cy, w, h, pad_factor=2.0)
        thresh_crop_pil = Image.fromarray(thresh_crop_np)
        
        # 3. Extract cleaned padded crop
        cleaned_crop_np = get_padded_crop(cv2.cvtColor(cleaned_np, cv2.COLOR_GRAY2RGB), cx, cy, w, h, pad_factor=2.0)
        cleaned_crop_pil = Image.fromarray(cleaned_crop_np)
        
        # 4. Rectified crop (upright/deskewed)
        rect_crop_pil = rectify_crop(
            cleaned_pil,
            bbox_metrics={"cx": cx, "cy": cy, "w": w, "h": h, "angle": angle_deg},
            buffer_percent=0.08
        )
        
        # Run classification
        class_name, class_conf = cls_model.predict(rect_crop_pil)
        char_display = CLASS_TO_CHAR.get(class_name, class_name)
        
        # Match with Ground Truth
        gt_match = None
        best_iou = 0.0
        best_gt_idx = -1
        
        for g_idx, g in enumerate(gt_list):
            g_corners_scaled = [[pt[0] * scale_x, pt[1] * scale_y] for pt in g["corners"]]
            iou = calculate_iou(corners, g_corners_scaled)
            if iou > best_iou:
                best_iou = iou
                best_gt_idx = g_idx
                
        is_class_correct = None
        is_orient_correct = None
        gt_expected_char = None
        gt_angle = None
        
        if best_iou > 0.4:
            gt_match = gt_list[best_gt_idx]
            gt_class_idx = gt_match["class"]
            gt_class_name = CLASSES[gt_class_idx]
            gt_expected_char = CLASS_TO_CHAR.get(gt_class_name, gt_class_name)
            is_class_correct = (class_name == gt_class_name)
            
            scaled_gt_corners = [[pt[0] * scale_x, pt[1] * scale_y] for pt in gt_match["corners"]]
            gt_angle = get_angle_from_corners(scaled_gt_corners)
            ang_diff = (angle_deg - gt_angle) % 180
            if ang_diff > 90:
                ang_diff -= 180
            is_orient_correct = (abs(ang_diff) < 20.0)
            
        detections.append({
            "id": box["id"],
            "yolo_conf": yolo_conf,
            "pred_class": class_name,
            "pred_char": char_display,
            "class_confidence": float(class_conf),
            "rotation_degrees": angle_deg,
            "corners": corners,
            "is_class_correct": is_class_correct,
            "is_orient_correct": is_orient_correct,
            "gt_expected_char": gt_expected_char,
            "gt_angle": float(gt_angle) if gt_angle is not None else None,
            "best_iou": float(best_iou),
            "crops": {
                "raw": pil_to_base64(raw_crop_pil, format="PNG"),
                "gray": pil_to_base64(gray_crop_pil, format="PNG"),
                "thresh": pil_to_base64(thresh_crop_pil, format="PNG"),
                "clean": pil_to_base64(cleaned_crop_pil, format="PNG"),
                "rect": pil_to_base64(rect_crop_pil, format="PNG")
            }
        })
        
    return {
        "filename": filename,
        "page_num": page_num,
        "detections": detections
    }

@app.websocket("/ws/process")
async def websocket_process(websocket: WebSocket):
    await websocket.accept()
    
    try:
        data = await websocket.receive_json()
        filename = data.get("filename")
        skip_animations = data.get("skip_animations", False)
        
        if not filename:
            await websocket.send_json({"error": "No filename provided"})
            return
            
        file_path = os.path.join("pdfs", filename)
        if not os.path.exists(file_path):
            await websocket.send_json({"error": "File not found"})
            return
            
        yolo, cls_model = get_models()
        
        # Load PDF
        from pdf2image import convert_from_path
        await websocket.send_json({"step": "loading_pdf", "message": f"Loading {filename}..."})
        # Use poppler_path if needed on windows, assuming it's in PATH or set it here
        # For this example, assuming poppler is in PATH
        pages = convert_from_path(file_path, dpi=200) # Lower DPI for faster web display
        
        await websocket.send_json({"step": "pdf_loaded", "total_pages": len(pages)})
        
        all_results = []
        
        for i, page_img in enumerate(pages):
            await websocket.send_json({"step": "page_start", "page_num": i + 1})
            
            # Preprocess
            preprocessed_img = preprocess_image(page_img)
            b64_page = pil_to_base64(preprocessed_img)
            await websocket.send_json({
                "step": "page_image", 
                "page_num": i + 1,
                "image": b64_page,
                "width": preprocessed_img.width,
                "height": preprocessed_img.height
            })
            
            if not skip_animations:
                await asyncio.sleep(0.5) # Pause to let UI show image
                
            img_np = np.array(preprocessed_img)
            img_bgr = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
            
            # YOLO
            results = yolo(img_bgr, verbose=False, conf=0.25, imgsz=1280)
            result = results[0]
            
            page_detections = []
            
            if result.obb is not None and len(result.obb) > 0:
                xywhr = result.obb.xywhr.cpu().numpy()
                xyxyxyxy = result.obb.xyxyxyxy.cpu().numpy()
                yolo_confs = result.obb.conf.cpu().numpy()
                
                temp_chars = []
                for idx in range(len(xywhr)):
                    cx, cy, w, h, r = xywhr[idx]
                    corners = xyxyxyxy[idx]
                    yolo_conf = yolo_confs[idx]
                    angle_deg = math.degrees(r)
                    
                    crop_pil = rectify_crop(
                        preprocessed_img,
                        bbox_metrics={'cx': cx, 'cy': cy, 'w': w, 'h': h, 'angle': angle_deg},
                        buffer_percent=0.08
                    )
                    
                    # Predict single character
                    class_name, char_conf = cls_model.predict(crop_pil)
                    char_display = CLASS_TO_CHAR.get(class_name, class_name)
                    char_crop_b64 = pil_to_base64(crop_pil)
                    
                    temp_chars.append({
                        "yolo_conf": float(yolo_conf),
                        "pred_class": class_name,
                        "pred_char": char_display,
                        "class_confidence": float(char_conf),
                        "rotation_degrees": float(angle_deg),
                        "center": [float(cx), float(cy)],
                        "size": [float(w), float(h)],
                        "corners": corners.tolist(),
                        "char_crop_b64": char_crop_b64,
                        "char_details": []
                    })
                
                # Use individual detections directly
                page_detections = temp_chars
                
                # Send YOLO bounding boxes to UI
                boxes_data = []
                for idx, expr in enumerate(page_detections):
                    boxes_data.append({
                        "id": idx,
                        "corners": expr["corners"],
                        "conf": expr["yolo_conf"]
                    })
                
                await websocket.send_json({
                    "step": "yolo_done",
                    "page_num": i + 1,
                    "boxes": boxes_data
                })
                
                if not skip_animations:
                    await asyncio.sleep(1.0) # Let UI draw boxes
                
                # Process each crop
                for idx, expr in enumerate(page_detections):
                    corners = expr["corners"]
                    
                    # Rectify the expression region
                    expr_crop_pil = rectify_crop(
                        preprocessed_img,
                        pts=np.array(corners, dtype=np.float32),
                        buffer_percent=0.08
                    )
                    expr_crop_b64 = pil_to_base64(expr_crop_pil)
                    
                    det = {
                        "id": idx,
                        "yolo_conf": expr["yolo_conf"],
                        "pred_class": expr["pred_class"],
                        "pred_char": expr["pred_char"],
                        "class_confidence": expr["class_confidence"],
                        "rotation_degrees": expr["rotation_degrees"],
                        "center": expr["center"],
                        "size": expr["size"],
                        "corners": corners,
                        "char_details": expr["char_details"]
                    }
                    
                    await websocket.send_json({
                        "step": "crop_processed",
                        "page_num": i + 1,
                        "detection": det,
                        "crop_image": expr_crop_b64
                    })
                    
                    if not skip_animations:
                        await asyncio.sleep(0.1) # Animate OCR result appearing
            
            all_results.append({
                "page": i + 1,
                "detections": page_detections
            })
            
            await websocket.send_json({
                "step": "page_done",
                "page_num": i + 1
            })
            
        await websocket.send_json({
            "step": "complete",
            "results": all_results
        })
        
    except WebSocketDisconnect:
        print("Client disconnected")
    except Exception as e:
        await websocket.send_json({"error": str(e)})
        
@app.get("/val-images")
def list_val_images():
    val_dir = os.path.join("dataset_yolo", "images", "val")
    if not os.path.exists(val_dir):
        return {"images": []}
    files = [f for f in os.listdir(val_dir) if f.lower().endswith((".png", ".jpg", ".jpeg"))]
    return {"images": sorted(files)}

def calculate_iou(box1, box2):
    poly1 = Polygon(box1)
    poly2 = Polygon(box2)
    if not poly1.is_valid:
        poly1 = poly1.buffer(0)
    if not poly2.is_valid:
        poly2 = poly2.buffer(0)
    inter = poly1.intersection(poly2).area
    union = poly1.area + poly2.area - inter
    return inter / union if union > 0 else 0

@app.websocket("/ws/validate")
async def websocket_validate(websocket: WebSocket):
    await websocket.accept()
    try:
        data = await websocket.receive_json()
        filename = data.get("filename")
        if not filename:
            await websocket.send_json({"error": "No filename provided"})
            return
            
        img_path = os.path.join("dataset_yolo", "images", "val", filename)
        if not os.path.exists(img_path):
            await websocket.send_json({"error": "Image not found"})
            return
            
        yolo, _ = get_models()
        
        # Load Image
        page_img = Image.open(img_path).convert("RGB")
        width, height = page_img.width, page_img.height
        
        b64_page = pil_to_base64(page_img)
        await websocket.send_json({
            "step": "image_loaded",
            "image": b64_page,
            "width": width,
            "height": height
        })
        
        # Ground Truth from JSON Database
        gt_db = get_gt_db()
        gt_boxes = []
        if filename in gt_db:
            for item in gt_db[filename]:
                gt_boxes.append(item["corners"])
                        
        await websocket.send_json({"step": "gt_loaded", "gt_boxes": gt_boxes})
        
        # Predictions
        img_np = np.array(page_img)
        img_bgr = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
        results = yolo(img_bgr, verbose=False, conf=0.25, imgsz=1280)
        result = results[0]
        
        pred_boxes = []
        if result.obb is not None and len(result.obb) > 0:
            xyxyxyxy = result.obb.xyxyxyxy.cpu().numpy()
            yolo_confs = result.obb.conf.cpu().numpy()
            for idx in range(len(xyxyxyxy)):
                pred_boxes.append({
                    "id": idx,
                    "corners": xyxyxyxy[idx].tolist(),
                    "conf": float(yolo_confs[idx])
                })
                
        # Calculate Metrics
        true_positives = 0
        sum_iou = 0.0
        matched_gt = set()
        
        for p in pred_boxes:
            best_iou = 0
            best_gt = -1
            for j, g in enumerate(gt_boxes):
                if j in matched_gt: continue
                iou = calculate_iou(p["corners"], g)
                if iou > best_iou:
                    best_iou = iou
                    best_gt = j
            if best_iou > 0.5:
                true_positives += 1
                sum_iou += best_iou
                matched_gt.add(best_gt)
                
        total_gt = len(gt_boxes)
        total_pred = len(pred_boxes)
        false_positives = total_pred - true_positives
        false_negatives = total_gt - true_positives
        avg_iou = sum_iou / true_positives if true_positives > 0 else 0.0
        
        stats = {
            "total_gt": total_gt,
            "total_pred": total_pred,
            "true_positives": true_positives,
            "false_positives": false_positives,
            "false_negatives": false_negatives,
            "avg_iou": avg_iou
        }
        
        await websocket.send_json({
            "step": "validation_done",
            "pred_boxes": pred_boxes,
            "stats": stats
        })
        
    except WebSocketDisconnect:
        pass
    except Exception as e:
        await websocket.send_json({"error": str(e)})

@app.get("/reports")
def list_reports():
    reports_dir = "output_pipeline"
    if not os.path.exists(reports_dir):
        return {"reports": []}
    dirs = [d for d in os.listdir(reports_dir) if os.path.isdir(os.path.join(reports_dir, d))]
    return {"reports": sorted(dirs)}

@app.get("/report/load")
def load_report(name: str):
    report_path = os.path.join("output_pipeline", name, "pipeline_report.json")
    img_path = os.path.join("output_pipeline", name, "page_with_bboxes.png")
    
    if not os.path.exists(report_path):
        return {"error": f"Report '{name}' not found."}
        
    try:
        with open(report_path, "r", encoding="utf-8") as f:
            report_data = json.load(f)
        
        img_b64 = ""
        if os.path.exists(img_path):
            with Image.open(img_path) as img:
                img_b64 = pil_to_base64(img, format="PNG")
                
        return {
            "report": report_data,
            "image": img_b64
        }
    except Exception as e:
        return {"error": str(e)}

@app.get("/report/crop")
def load_report_crop(image_name: str, cx: float, cy: float, w: float, h: float, angle: float):
    # 1. Search for the source image in the workspace
    orig_img_path = None
    for search_dir in ["toTest", "dataset_yolo"]:
        if os.path.exists(search_dir):
            for root, dirs, files in os.walk(search_dir):
                if image_name in files:
                    orig_img_path = os.path.join(root, image_name)
                    break
            if orig_img_path:
                break
                
    if not orig_img_path:
        return {"error": f"Source image '{image_name}' not found."}
        
    try:
        page_img = Image.open(orig_img_path).convert("RGB")
        
        # Morphological stages
        raw_np, gray_np, thresh_np, cleaned_np = preprocess_stages(page_img)
        cleaned_pil = Image.fromarray(cv2.cvtColor(cleaned_np, cv2.COLOR_GRAY2RGB))
        
        # 1. Extract raw padded crop
        raw_crop_np = get_padded_crop(raw_np, cx, cy, w, h, pad_factor=2.0)
        raw_crop_pil = Image.fromarray(raw_crop_np)
        
        # 2. Extract gray padded crop
        gray_crop_np = get_padded_crop(cv2.cvtColor(gray_np, cv2.COLOR_GRAY2RGB), cx, cy, w, h, pad_factor=2.0)
        gray_crop_pil = Image.fromarray(gray_crop_np)
        
        # 3. Extract thresholded padded crop
        thresh_crop_np = get_padded_crop(cv2.cvtColor(thresh_np, cv2.COLOR_GRAY2RGB), cx, cy, w, h, pad_factor=2.0)
        thresh_crop_pil = Image.fromarray(thresh_crop_np)
        
        # 4. Extract cleaned padded crop
        cleaned_crop_np = get_padded_crop(cv2.cvtColor(cleaned_np, cv2.COLOR_GRAY2RGB), cx, cy, w, h, pad_factor=2.0)
        cleaned_crop_pil = Image.fromarray(cleaned_crop_np)
        
        # 5. Rectified crop (upright/deskewed)
        rect_crop_pil = rectify_crop(
            cleaned_pil,
            bbox_metrics={"cx": cx, "cy": cy, "w": w, "h": h, "angle": angle},
            buffer_percent=0.08
        )
        
        return {
            "crops": {
                "raw": pil_to_base64(raw_crop_pil, format="PNG"),
                "gray": pil_to_base64(gray_crop_pil, format="PNG"),
                "thresh": pil_to_base64(thresh_crop_pil, format="PNG"),
                "clean": pil_to_base64(cleaned_crop_pil, format="PNG"),
                "rect": pil_to_base64(rect_crop_pil, format="PNG")
            }
        }
    except Exception as e:
        return {"error": str(e)}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
