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

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global models (loaded on startup or first request)
yolo_model = None
classifier = None

def get_models():
    global yolo_model, classifier
    if yolo_model is None:
        yolo_path = "runs/obb/yolo_obb_project/symbol_obb_train-2/weights/best.pt"
        if not os.path.exists(yolo_path):
            yolo_path = "runs/obb/yolo_obb_project/symbol_obb_train/weights/best.pt"
        yolo_model = load_yolo_model(yolo_path)
    if classifier is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        classifier = SymbolClassifier(model_path="classifier_best.pt", device=device)
    return yolo_model, classifier

def pil_to_base64(img: Image.Image, format="PNG") -> str:
    buffered = io.BytesIO()
    img.save(buffered, format=format)
    img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")
    return f"data:image/{format.lower()};base64,{img_str}"

@app.get("/pdfs")
def list_pdfs():
    pdfs_dir = "pdfs"
    if not os.path.exists(pdfs_dir):
        return {"pdfs": []}
    files = [f for f in os.listdir(pdfs_dir) if f.lower().endswith(".pdf")]
    return {"pdfs": sorted(files)}

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
                
                # Send YOLO bounding boxes to UI
                boxes_data = []
                for idx in range(len(xywhr)):
                    boxes_data.append({
                        "id": idx,
                        "corners": xyxyxyxy[idx].tolist(),
                        "conf": float(yolo_confs[idx])
                    })
                
                await websocket.send_json({
                    "step": "yolo_done",
                    "page_num": i + 1,
                    "boxes": boxes_data
                })
                
                if not skip_animations:
                    await asyncio.sleep(1.0) # Let UI draw boxes
                
                # Process each crop
                for idx in range(len(xywhr)):
                    cx, cy, w, h, r = xywhr[idx]
                    corners = xyxyxyxy[idx]
                    angle_deg = math.degrees(r)
                    
                    crop_pil = rectify_crop(
                        preprocessed_img,
                        bbox_metrics={'cx': cx, 'cy': cy, 'w': w, 'h': h, 'angle': angle_deg},
                        buffer_percent=0.08
                    )
                    
                    class_name, class_conf = cls_model.predict(crop_pil)
                    
                    char_display = CLASS_TO_CHAR.get(class_name, class_name)
                    det = {
                        "id": idx,
                        "yolo_conf": float(yolo_confs[idx]),
                        "pred_class": class_name,
                        "pred_char": char_display,
                        "class_confidence": float(class_conf),
                        "rotation_degrees": float(angle_deg),
                        "center": [float(cx), float(cy)],
                        "size": [float(w), float(h)],
                        "corners": corners.tolist()
                    }
                    page_detections.append(det)
                    
                    await websocket.send_json({
                        "step": "crop_processed",
                        "page_num": i + 1,
                        "detection": det,
                        "crop_image": pil_to_base64(crop_pil) if not skip_animations else None
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
        gt_db_path = os.path.join("dataset_yolo", "ground_truth.json")
        gt_boxes = []
        if os.path.exists(gt_db_path):
            with open(gt_db_path, "r") as f:
                gt_db = json.load(f)
                
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

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
