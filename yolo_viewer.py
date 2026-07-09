import os
import json
import base64
import math
import io
import cv2
import numpy as np
import torch
import uvicorn
import subprocess
from PIL import Image, ImageDraw, ImageFont
from fastapi import FastAPI, HTTPException, Response, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

# Import custom pipeline components in the same workspace directory
from rectifier import rectify_crop
from classifier import SymbolClassifier, CLASSES
from pipeline import CLASS_TO_CHAR, load_yolo_model
from preprocessor import full_preprocess, preprocess_stages

app = FastAPI(title="YOLO-OBB and PaddleOCR Interactive Viewer API")

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global model pointers for cache loading
yolo_model = None
classifier_model = None

def get_models():
    """Lazy loads the YOLO and Classifier models to save resources and ensure correct pathways."""
    global yolo_model, classifier_model
    
    if yolo_model is None:
        yolo_path = "runs/obb/runs/obb/yolo_1000_pdfs_100_epochs/weights/best.pt"
        if not os.path.exists(yolo_path):
            # Fallbacks
            fallbacks = [
                "runs/obb/runs/obb/trained_on_1000_pdfs-2/weights/best.pt",
                "yolov8n-obb.pt",
                "weights/best.pt"
            ]
            for path in fallbacks:
                if os.path.exists(path):
                    yolo_path = path
                    break
        print(f"Loading YOLOv8-OBB model from: {yolo_path}")
        yolo_model = load_yolo_model(yolo_path)

    if classifier_model is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        clf_path = "classifier_best_updated.pt"
        if not os.path.exists(clf_path):
            fallbacks = [
                "classifier_best_bigdatset.pt",
                "classifier_best.pt"
            ]
            for path in fallbacks:
                if os.path.exists(path):
                    clf_path = path
                    break
        print(f"Loading MobileNetV3 Classifier from: {clf_path} (Device: {device})")
        classifier_model = SymbolClassifier(model_path=clf_path, device=device)
        
    return yolo_model, classifier_model

class PredictRequest(BaseModel):
    dir_path: str
    filename: str
    model_type: str = "yolo" # "yolo" | "paddle"

class ClearCacheRequest(BaseModel):
    dir_path: str
    filename: str
    model_type: str = "yolo" # "yolo" | "paddle"

def ensure_paddle_crops(dir_path: str, filename: str, boxes: list):
    """Crops each bounding box from the original image, saves them to a BBforOCR folder,
    and saves the full image with all bounding boxes overlayed."""
    if not boxes:
        return
        
    bb_dir = os.path.join(dir_path, "BBforOCR")
    os.makedirs(bb_dir, exist_ok=True)
    
    img_path = os.path.join(dir_path, filename)
    if not os.path.exists(img_path):
        print(f"Source image not found for cropping: {img_path}")
        return
        
    try:
        with Image.open(img_path) as img:
            # Proper alpha-compositing to a white background to remove transparent ghost layers
            if img.mode in ("RGBA", "LA") or (img.mode == "P" and "transparency" in img.info):
                background = Image.new("RGB", img.size, (255, 255, 255))
                mask = img.split()[3] if img.mode == "RGBA" else img.convert("RGBA").split()[3]
                background.paste(img, mask=mask)
                img_rgb = background
            else:
                img_rgb = img.convert("RGB")
            base_name = os.path.splitext(filename)[0]
            
            # 1. Crop and save individual boxes
            for box in boxes:
                box_id = box.get("id")
                cx = box.get("cx")
                cy = box.get("cy")
                w = box.get("w")
                h = box.get("h")
                angle = box.get("angle")
                
                if cx is None or cy is None or w is None or h is None or angle is None:
                    continue
                
                # Crop and rectify standard (10%) and big (40%) regions
                crop_pil_std = rectify_crop(
                    img_rgb,
                    bbox_metrics={"cx": cx, "cy": cy, "w": w, "h": h, "angle": angle},
                    buffer_percent=0.10
                )
                crop_pil_big = rectify_crop(
                    img_rgb,
                    bbox_metrics={"cx": cx, "cy": cy, "w": w, "h": h, "angle": angle},
                    buffer_percent=0.40
                )
                # Clean CAD lines and noise from the crop
                _, _, _, s_clean = clean_cad_lines_boundary_tracing_stages(crop_pil_std, crop_pil_big)
                crop_pil = Image.fromarray(s_clean).convert("RGB")
                
                crop_name = f"{base_name}_crop_{box_id}.png"
                crop_path = os.path.join(bb_dir, crop_name)
                crop_pil.save(crop_path)
                
            print(f"Successfully saved {len(boxes)} PaddleOCR crops to {bb_dir}")
            
            # 2. Draw all bounding boxes on the original image and save it
            annotated_img = img_rgb.copy()
            draw = ImageDraw.Draw(annotated_img)
            
            # Load font
            font_choices = [
                "C:\\Windows\\Fonts\\seguisym.ttf",
                "C:\\Windows\\Fonts\\arial.ttf",
                "C:\\Windows\\Fonts\\calibri.ttf",
                "C:\\Windows\\Fonts\\segoeui.ttf"
            ]
            font = None
            for path in font_choices:
                if os.path.exists(path):
                    try:
                        font = ImageFont.truetype(path, 14)
                        break
                    except:
                        pass
            if font is None:
                font = ImageFont.load_default()
                
            for box in boxes:
                corners = box.get("corners")
                if not corners:
                    continue
                    
                # Draw polygon outline (green)
                poly_pts = [(float(pt[0]), float(pt[1])) for pt in corners]
                draw.polygon(poly_pts, outline=(0, 200, 0), width=2)
                
                # Draw text label (red text)
                lx, ly = float(corners[0][0]), float(corners[0][1])
                text = box.get("char_display", "Text")
                conf = box.get("confidence", 1.0)
                label_text = f"{text} ({conf:.2f})"
                
                # Draw small background box for text legibility
                try:
                    text_bbox = draw.textbbox((lx, ly - 18), label_text, font=font)
                    draw.rectangle(text_bbox, fill=(255, 255, 255))
                except AttributeError:
                    pass
                draw.text((lx, ly - 18), label_text, font=font, fill=(220, 0, 0))
                
            annotated_name = f"{base_name}_annotated.png"
            annotated_path = os.path.join(bb_dir, annotated_name)
            annotated_img.save(annotated_path)
            print(f"Saved annotated image to {annotated_path}")
            
    except Exception as e:
        print(f"Error ensuring paddle crops for {filename}: {e}")

def run_paddle_detect_only(image_path: str):
    import urllib.request
    import urllib.error
    url = "http://127.0.0.1:8001/detect"
    data = json.dumps({"image_path": image_path}).encode("utf-8")
    req_http = urllib.request.Request(
        url,
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST"
    )
    with urllib.request.urlopen(req_http, timeout=180) as response:
        return json.loads(response.read().decode("utf-8"))

def run_paddle_recognize_crop(crop_b64: str):
    import urllib.request
    import urllib.error
    url = "http://127.0.0.1:8001/recognize_crop"
    data = json.dumps({"image_b64": crop_b64}).encode("utf-8")
    req_http = urllib.request.Request(
        url,
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST"
    )
    with urllib.request.urlopen(req_http, timeout=180) as response:
        return json.loads(response.read().decode("utf-8"))

def run_paddle_detection(image_path: str):
    """Tries to fetch predictions from the persistent PaddleOCR background service (port 8001).
    Falls back to running a separate python subprocess if the service is not active."""
    import urllib.request
    import urllib.error
    
    url = "http://127.0.0.1:8001/predict"
    data = json.dumps({"image_path": image_path}).encode("utf-8")
    
    req_http = urllib.request.Request(
        url,
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST"
    )
    
    try:
        print("Sending prediction request to persistent PaddleOCR service...")
        with urllib.request.urlopen(req_http, timeout=180) as response:
            return json.loads(response.read().decode("utf-8"))
    except Exception as e:
        print(f"Persistent PaddleOCR service error or timeout ({e}). Falling back to subprocess...")
        
        
        # Subprocess fallback logic
        current_dir = os.path.dirname(os.path.abspath(__file__))
        python_exe = os.path.abspath(os.path.join(current_dir, "..", "PaddleOCR", "venv", "Scripts", "python.exe"))
        script_path = os.path.abspath(os.path.join(current_dir, "paddle_detect.py"))
        
        if not os.path.exists(python_exe):
            raise Exception(f"PaddleOCR python executable not found at: {python_exe}")
        if not os.path.exists(script_path):
            raise Exception(f"PaddleOCR detection script not found at: {script_path}")
            
        env = os.environ.copy()
        env["FLAGS_enable_onednn"] = "0"
        env["PADDLE_PDX_ENABLE_MKLDNN_BYDEFAULT"] = "False"
        env["PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK"] = "True"
        
        cmd = [python_exe, script_path, image_path]
        print(f"Executing PaddleOCR Subprocess: {' '.join(cmd)}")
        
        res = subprocess.run(cmd, capture_output=True, env=env)
        
        if res.returncode != 0:
            err_msg = res.stderr.decode('utf-8', errors='replace')
            raise Exception(f"PaddleOCR subprocess failed (exit code {res.returncode}): {err_msg}")
            
        output_str = res.stdout.decode('utf-8', errors='replace')
        
        # Extract JSON array from stdout
        start_idx = output_str.find('[')
        end_idx = output_str.rfind(']')
        
        if start_idx == -1 or end_idx == -1:
            raise Exception(f"Invalid PaddleOCR subprocess output (JSON array not found): {output_str}")
            
        json_str = output_str[start_idx:end_idx + 1]
        return json.loads(json_str)

@app.get("/api/images")
def get_images(dir: str):
    """Scans the provided directory path for images and checks if predictions cache exists."""
    if not os.path.exists(dir) or not os.path.isdir(dir):
        return {"error": f"Directory path does not exist or is invalid: {dir}"}
        
    try:
        valid_extensions = {".png", ".jpg", ".jpeg", ".bmp", ".tiff", ".webp"}
        files = []
        
        # Look for predictions subfolder
        pred_dir = os.path.join(dir, "predictions")
        
        for name in os.listdir(dir):
            if os.path.isfile(os.path.join(dir, name)):
                ext = os.path.splitext(name)[1].lower()
                if ext in valid_extensions:
                    # Check YOLO cache existence
                    yolo_cache = os.path.join(pred_dir, f"{name}.json")
                    predicted_yolo = os.path.exists(yolo_cache)
                    
                    # Check PaddleOCR cache existence
                    paddle_cache = os.path.join(pred_dir, f"paddle_{name}.json")
                    predicted_paddle = os.path.exists(paddle_cache)
                    
                    files.append({
                        "filename": name,
                        "predicted_yolo": predicted_yolo,
                        "predicted_paddle": predicted_paddle
                    })
                    
        # Sort files alphabetically
        files.sort(key=lambda x: x["filename"])
        return {"images": files}
    except Exception as e:
        return {"error": f"Failed to read directory: {str(e)}"}

@app.get("/api/image-file")
def get_image_file(filepath: str, cleaned: bool = False):
    """Serves the raw image or the preprocessed/cleaned version."""
    if not os.path.exists(filepath):
        raise HTTPException(status_code=404, detail="Image file not found")
        
    if cleaned:
        try:
            # Process drawing (line subtraction, adaptive thresholding) dynamically
            with Image.open(filepath) as img:
                img_rgb = img.convert("RGB")
                cleaned_img = full_preprocess(img_rgb)
                
                # Output to binary buffer
                buf = io.BytesIO()
                cleaned_img.save(buf, format="JPEG", quality=90)
                buf.seek(0)
                
                return StreamingResponse(buf, media_type="image/jpeg")
        except Exception as e:
            print(f"Error preprocessing image: {e}")
            raise HTTPException(status_code=500, detail=f"Failed to preprocess image: {str(e)}")
    
    return FileResponse(filepath)

import asyncio

@app.websocket("/api/ws/paddle")
async def ws_paddle(websocket: WebSocket):
    import time
    await websocket.accept()
    try:
        data = await websocket.receive_json()
        dir_path = data.get("dir_path")
        filename = data.get("filename")
        
        if not dir_path or not filename:
            await websocket.send_json({"error": "dir_path and filename are required"})
            await websocket.close()
            return
            
        img_path = os.path.join(dir_path, filename)
        if not os.path.exists(img_path):
            await websocket.send_json({"error": f"Image file not found: {img_path}"})
            await websocket.close()
            return
            
        pred_dir = os.path.join(dir_path, "predictions")
        cache_path = os.path.join(pred_dir, f"paddle_{filename}.json")
        
        # 1. Check if cached predictions exist
        if os.path.exists(cache_path):
            try:
                print(f"WS: Loading cached paddle predictions for: {filename}")
                with open(cache_path, "r", encoding="utf-8") as f:
                    cached_data = json.load(f)
                
                boxes = cached_data.get("boxes", [])
                
                # Send the detection boxes first (with empty/placeholder text display)
                det_boxes = []
                for box in boxes:
                    det_boxes.append({
                        **box,
                        "char_display": "Pending...",
                        "class_confidence": 0.0
                    })
                
                await websocket.send_json({
                    "type": "detection",
                    "width": cached_data.get("width"),
                    "height": cached_data.get("height"),
                    "boxes": det_boxes
                })
                
                # Stream recognition results one-by-one with a small delay
                for box in boxes:
                    await asyncio.sleep(0.05)  # 50ms simulation delay
                    await websocket.send_json({
                        "type": "recognition",
                        "id": box["id"],
                        "char_display": box.get("char_display", "Text"),
                        "class_confidence": box.get("class_confidence", 1.0)
                    })
                    
                # Ensure crops exist in BBforOCR folder
                ensure_paddle_crops(dir_path, filename, boxes)
                
                await websocket.send_json({
                    "type": "complete",
                    "det_time": cached_data.get("det_time", 0.0),
                    "rec_time": cached_data.get("rec_time", 0.0),
                    "total_time": cached_data.get("total_time", 0.0)
                })
                await websocket.close()
                return
            except Exception as e:
                print(f"WS: Cache corrupt or failed to send: {e}, running live model...")
                
        # 2. Run live PaddleOCR
        total_start = time.perf_counter()
        
        # A. Preprocess: solid-white alpha-compositing
        temp_clean_path = os.path.join(pred_dir, f"temp_clean_{filename}")
        os.makedirs(pred_dir, exist_ok=True)
        
        with Image.open(img_path) as page_pil:
            width, height = page_pil.width, page_pil.height
            if page_pil.mode in ("RGBA", "LA") or (page_pil.mode == "P" and "transparency" in page_pil.info):
                background = Image.new("RGB", page_pil.size, (255, 255, 255))
                mask = page_pil.split()[3] if page_pil.mode == "RGBA" else page_pil.convert("RGBA").split()[3]
                background.paste(page_pil, mask=mask)
                img_rgb = background
            else:
                img_rgb = page_pil.convert("RGB")
            img_rgb.save(temp_clean_path)
            
        # Try detection via persistent server first
        boxes_data = []
        is_persistent = True
        det_start = time.perf_counter()
        try:
            print("WS: Sending detection request to persistent PaddleOCR service...")
            boxes_data = await asyncio.to_thread(run_paddle_detect_only, temp_clean_path)
        except Exception as e:
            print(f"WS: Persistent PaddleOCR service not available ({e}). Running full model via subprocess fallback...")
            is_persistent = False
        det_time = time.perf_counter() - det_start
            
        # Clean up temporary clean image file
        try:
            if os.path.exists(temp_clean_path):
                os.remove(temp_clean_path)
        except Exception as e:
            print(f"WS: Failed to delete temp clean image: {e}")
            
        if not is_persistent:
            # Subprocess fallback: get all data at once, then stream
            try:
                sub_start = time.perf_counter()
                full_boxes = await asyncio.to_thread(run_paddle_detection, img_path)
                sub_duration = time.perf_counter() - sub_start
                det_time = sub_duration * 0.3
                rec_time = sub_duration * 0.7
                
                # Send detection boxes first
                det_boxes = []
                for box in full_boxes:
                    det_boxes.append({
                        **box,
                        "char_display": "Pending...",
                        "class_confidence": 0.0
                    })
                await websocket.send_json({
                    "type": "detection",
                    "width": width,
                    "height": height,
                    "boxes": det_boxes
                })
                
                # Stream recognition results one-by-one with a small delay
                for box in full_boxes:
                    await asyncio.sleep(0.1)  # 100ms visual delay
                    await websocket.send_json({
                        "type": "recognition",
                        "id": box["id"],
                        "char_display": box.get("char_display", "Text"),
                        "class_confidence": box.get("class_confidence", 1.0)
                    })
                    
                total_time = time.perf_counter() - total_start
                response_data = {
                    "filepath": img_path,
                    "width": width,
                    "height": height,
                    "boxes": full_boxes,
                    "det_time": det_time,
                    "rec_time": rec_time,
                    "total_time": total_time
                }
                with open(cache_path, "w", encoding="utf-8") as f:
                    json.dump(response_data, f, indent=4)
                    
                ensure_paddle_crops(dir_path, filename, full_boxes)
                await websocket.send_json({
                    "type": "complete",
                    "det_time": det_time,
                    "rec_time": rec_time,
                    "total_time": total_time
                })
                await websocket.close()
                return
            except Exception as sub_err:
                print(f"WS: Subprocess fallback failed: {sub_err}")
                await websocket.send_json({"error": f"PaddleOCR subprocess failed: {sub_err}"})
                await websocket.close()
                return
                
        # Persistent Server is active! We have boxes_data from /detect (with char_display='Pending')
        # Send detection boxes to client
        await websocket.send_json({
            "type": "detection",
            "width": width,
            "height": height,
            "boxes": boxes_data
        })
        
        # Load the composite image in memory for cropping
        # We need a clean composite image (white background alpha-composited) to crop from.
        with Image.open(img_path) as page_pil:
            if page_pil.mode in ("RGBA", "LA") or (page_pil.mode == "P" and "transparency" in page_pil.info):
                background = Image.new("RGB", page_pil.size, (255, 255, 255))
                mask = page_pil.split()[3] if page_pil.mode == "RGBA" else page_pil.convert("RGBA").split()[3]
                background.paste(page_pil, mask=mask)
                cleaned_pil = background
            else:
                cleaned_pil = page_pil.convert("RGB")
                
        final_boxes = []
        rec_start = time.perf_counter()
        for box in boxes_data:
            # We crop and rectify using the scaled corners
            try:
                # Crop standard (10%) and big (40%) regions from image
                crop_pil_std = rectify_crop(
                    cleaned_pil,
                    bbox_metrics={
                        "cx": box["cx"],
                        "cy": box["cy"],
                        "w": box["w"],
                        "h": box["h"],
                        "angle": box["angle"]
                    },
                    buffer_percent=0.10
                )
                crop_pil_big = rectify_crop(
                    cleaned_pil,
                    bbox_metrics={
                        "cx": box["cx"],
                        "cy": box["cy"],
                        "w": box["w"],
                        "h": box["h"],
                        "angle": box["angle"]
                    },
                    buffer_percent=0.40
                )
                
                # Clean CAD lines and noise
                _, _, _, s_clean = clean_cad_lines_boundary_tracing_stages(crop_pil_std, crop_pil_big)
                clean_pil = Image.fromarray(s_clean).convert("RGB")
                
                # Convert cleaned PIL crop to Base64 PNG
                buf = io.BytesIO()
                clean_pil.save(buf, format="PNG")
                crop_b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
                
                # Request recognition on the crop asynchronously
                rec_res = await asyncio.to_thread(run_paddle_recognize_crop, crop_b64)
                text = rec_res.get("text", "")
                score = rec_res.get("score", 0.0)
                
                # Send update to UI
                await websocket.send_json({
                    "type": "recognition",
                    "id": box["id"],
                    "char_display": text,
                    "class_confidence": score
                })
                
                box_copy = {**box}
                box_copy["char_display"] = text
                box_copy["class_confidence"] = score
                box_copy["confidence"] = score
                final_boxes.append(box_copy)
                
            except Exception as box_err:
                print(f"WS: Error recognizing box {box['id']}: {box_err}")
                # Send fallback
                await websocket.send_json({
                    "type": "recognition",
                    "id": box["id"],
                    "char_display": "Error",
                    "class_confidence": 0.0
                })
                box_copy = {**box}
                box_copy["char_display"] = "Error"
                box_copy["class_confidence"] = 0.0
                box_copy["confidence"] = 0.0
                final_boxes.append(box_copy)
                
        rec_time = time.perf_counter() - rec_start
        total_time = time.perf_counter() - total_start
        
        # Cache final predictions
        response_data = {
            "filepath": img_path,
            "width": width,
            "height": height,
            "boxes": final_boxes,
            "det_time": det_time,
            "rec_time": rec_time,
            "total_time": total_time
        }
        with open(cache_path, "w", encoding="utf-8") as f:
            json.dump(response_data, f, indent=4)
        print(f"WS: Saved PaddleOCR predictions cache to: {cache_path}")
        
        # Save crops in BBforOCR folder
        ensure_paddle_crops(dir_path, filename, final_boxes)
        
        await websocket.send_json({
            "type": "complete",
            "det_time": det_time,
            "rec_time": rec_time,
            "total_time": total_time
        })
        await websocket.close()
        
    except WebSocketDisconnect:
        print("WS: Client disconnected")
    except Exception as e:
        import traceback
        print(f"WS error: {e}\n{traceback.format_exc()}")
        try:
            await websocket.send_json({"error": str(e)})
            await websocket.close()
        except:
            pass

@app.post("/api/predict")
def predict(req: PredictRequest):
    """Checks cache or triggers YOLO OBB / PaddleOCR on request."""
    img_path = os.path.join(req.dir_path, req.filename)
    if not os.path.exists(img_path):
        raise HTTPException(status_code=404, detail="Source image file not found")

    pred_dir = os.path.join(req.dir_path, "predictions")
    
    # Cache files are named differently depending on model selection
    if req.model_type == "paddle":
        cache_path = os.path.join(pred_dir, f"paddle_{req.filename}.json")
    else:
        cache_path = os.path.join(pred_dir, f"{req.filename}.json")

    # 1. Load from cache if available
    if os.path.exists(cache_path):
        try:
            print(f"Loading cached {req.model_type} predictions for: {req.filename}")
            with open(cache_path, "r", encoding="utf-8") as f:
                cached_data = json.load(f)
            
            # Ensure crops exist in BBforOCR folder if model is paddle
            if req.model_type == "paddle":
                ensure_paddle_crops(req.dir_path, req.filename, cached_data.get("boxes", []))
                
            return cached_data
        except Exception as e:
            print(f"Cache corrupt for {req.filename}, running model: {e}")

    # 2. Run inference if not cached
    try:
        if req.model_type == "paddle":
            print(f"Executing PaddleOCR detection via subprocess on: {req.filename}")
            
            # 1. Composite image to a white background to remove transparent ghost layers
            # and write it to a temporary clean file for PaddleOCR to detect on.
            temp_clean_path = os.path.join(pred_dir, f"temp_clean_{req.filename}")
            os.makedirs(pred_dir, exist_ok=True)
            
            with Image.open(img_path) as page_pil:
                width, height = page_pil.width, page_pil.height
                if page_pil.mode in ("RGBA", "LA") or (page_pil.mode == "P" and "transparency" in page_pil.info):
                    background = Image.new("RGB", page_pil.size, (255, 255, 255))
                    mask = page_pil.split()[3] if page_pil.mode == "RGBA" else page_pil.convert("RGBA").split()[3]
                    background.paste(page_pil, mask=mask)
                    img_rgb = background
                else:
                    img_rgb = page_pil.convert("RGB")
                img_rgb.save(temp_clean_path)
            
            # 2. Run sub-process detection on the clean composite image
            boxes_data = run_paddle_detection(temp_clean_path)
            
            # 3. Clean up the temporary clean image file
            try:
                if os.path.exists(temp_clean_path):
                    os.remove(temp_clean_path)
            except Exception as e:
                print(f"Failed to delete temp clean image: {e}")
            
            response_data = {
                "filepath": img_path,
                "width": width,
                "height": height,
                "boxes": boxes_data
            }
            
            # Cache output
            os.makedirs(pred_dir, exist_ok=True)
            with open(cache_path, "w", encoding="utf-8") as f:
                json.dump(response_data, f, indent=4)
            print(f"Saved PaddleOCR predictions cache to: {cache_path}")
            
            # Save crops in BBforOCR folder
            ensure_paddle_crops(req.dir_path, req.filename, boxes_data)
            
            return response_data
            
        else:
            # YOLO path
            # Lazy load models
            yolo, classifier = get_models()
            
            print(f"Executing YOLO detection model on: {req.filename}")
            with Image.open(img_path) as page_pil:
                img_rgb = page_pil.convert("RGB")
                width, height = img_rgb.width, img_rgb.height
                
                # Preprocess using same logic as main pipeline
                preprocessed_img = full_preprocess(img_rgb)
                img_np = np.array(preprocessed_img)
                img_bgr = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
                
                # Run YOLO OBB detection
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
                        
                        # Crop and Rectify OBB box region
                        crop_pil = rectify_crop(
                            preprocessed_img,
                            bbox_metrics={"cx": cx, "cy": cy, "w": w, "h": h, "angle": angle_deg},
                            buffer_percent=0.08
                        )
                        
                        # Classify Crop
                        class_name, class_conf = classifier.predict(crop_pil)
                        char_display = CLASS_TO_CHAR.get(class_name, class_name)
                        
                        boxes_data.append({
                            "id": idx,
                            "cx": float(cx),
                            "cy": float(cy),
                            "w": float(w),
                            "h": float(h),
                            "angle": float(angle_deg),
                            "corners": corners.tolist(),
                            "confidence": float(yolo_conf),
                            "class_name": class_name,
                            "char_display": char_display,
                            "class_confidence": float(class_conf)
                        })
                
                # Structure response
                response_data = {
                    "filepath": img_path,
                    "width": width,
                    "height": height,
                    "boxes": boxes_data
                }
                
                # Cache output
                os.makedirs(pred_dir, exist_ok=True)
                with open(cache_path, "w", encoding="utf-8") as f:
                    json.dump(response_data, f, indent=4)
                print(f"Saved YOLO predictions cache to: {cache_path}")
                
                return response_data
            
    except Exception as e:
        import traceback
        err_msg = traceback.format_exc()
        print(f"Error running inference: {e}\n{err_msg}")
        raise HTTPException(status_code=500, detail=f"Inference error: {str(e)}")

def get_padded_crop(img_np: np.ndarray, cx: float, cy: float, w: float, h: float, pad_factor: float = 2.0) -> np.ndarray:
    """Extracts an axis-aligned square padded crop centered at (cx, cy)."""
    size = int(math.ceil(max(w, h) * pad_factor))
    half = size // 2
    ih, iw = img_np.shape[:2]
    x0 = max(0, int(cx) - half)
    y0 = max(0, int(cy) - half)
    x1 = min(iw, x0 + size)
    y1 = min(ih, y0 + size)
    crop = img_np[y0:y1, x0:x1]
    
    # Pad to square with white background (255)
    if img_np.ndim == 3:
        sq = np.ones((size, size, 3), dtype=np.uint8) * 255
    else:
        sq = np.ones((size, size), dtype=np.uint8) * 255
        
    ch, cw = crop.shape[:2]
    sq[:ch, :cw] = crop
    return sq

def skeletonize(img):
    """Applies morphological skeletonization to thinned 1-pixel width."""
    size = np.size(img)
    skel = np.zeros(img.shape, np.uint8)
    element = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))
    done = False
    temp_img = img.copy()
    
    while not done:
        eroded = cv2.erode(temp_img, element)
        temp = cv2.dilate(eroded, element)
        temp = cv2.subtract(temp_img, temp)
        skel = cv2.bitwise_or(skel, temp)
        temp_img = eroded.copy()
        
        zeros = size - cv2.countNonZero(temp_img)
        if zeros == size:
            done = True
            
    return skel

def get_junctions(skel):
    """Finds junction points where lines cross/meet in the skeleton.
    Junctions are thinned pixels with > 2 neighbors in their 8-neighborhood.
    """
    kernel = np.array([[1, 1, 1],
                       [1, 0, 1],
                       [1, 1, 1]], dtype=np.uint8)
    
    binary_skel = (skel > 0).astype(np.uint8)
    neighbor_count = cv2.filter2D(binary_skel, -1, kernel)
    junctions = (binary_skel == 1) & (neighbor_count > 2)
    return junctions.astype(np.uint8) * 255

def clean_cad_lines_boundary_tracing_stages(crop_pil_std, crop_pil_big):
    """Traces connected components crossing from the outer buffer region into the inner standard crop.
    Binarization is used ONLY internally for detection. The output (stage_clean) is the raw
    grayscale crop with detected line pixels painted white — no thresholding or speck removal applied.
    Returns: stage_thresh, stage_denoised, stage_lines, stage_clean as uint8 numpy arrays
    """
    img_std = np.array(crop_pil_std.convert("L"))
    img_big = np.array(crop_pil_big.convert("L"))
    
    H_std, W_std = img_std.shape
    H_big, W_big = img_big.shape
    
    # Calculate top-left offset of standard crop within the big crop
    dx = (W_big - W_std) // 2
    dy = (H_big - H_std) // 2
    
    # --- INTERNAL binarization only (used for component detection + Hough, NOT the output) ---
    _, thresh_std = cv2.threshold(img_std, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    _, thresh_big = cv2.threshold(img_big, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    
    # Stage 2: Show thresholded for visualization only
    stage_thresh = cv2.bitwise_not(thresh_std)
    
    # Stage 3: Bypass (no extra processing) — same as thresh for display
    stage_denoised = stage_thresh
    
    # Dilate thresh_big slightly to bridge small gaps and ensure continuity across the boundary
    dilation_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    thresh_big_dilated = cv2.dilate(thresh_big, dilation_kernel)
    
    # Connected component analysis on the dilated big crop
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(thresh_big_dilated, connectivity=8)
    
    # Define boundary (border) mask and inner mask inside the big crop
    boundary_mask = np.ones((H_big, W_big), dtype=np.uint8) * 255
    boundary_mask[dy:dy+H_std, dx:dx+W_std] = 0
    
    inner_mask = np.zeros((H_big, W_big), dtype=np.uint8)
    inner_mask[dy:dy+H_std, dx:dx+W_std] = 255
    
    # Build erase mask: pixels that belong to components crossing the outer boundary
    erase_mask_inner = np.zeros((H_std, W_std), dtype=np.uint8)
    for label in range(1, num_labels):
        comp_mask = (labels == label)
        has_boundary = np.any(comp_mask & (boundary_mask == 255))
        has_inner    = np.any(comp_mask & (inner_mask == 255))
        if has_boundary and has_inner:
            comp_inner = comp_mask[dy:dy+H_std, dx:dx+W_std]
            erase_mask_inner[comp_inner & (thresh_std > 0)] = 255
    
    # Hough line detection — confirms which crossing-component pixels are actual lines
    lines = cv2.HoughLinesP(thresh_std, 1, np.pi/180, threshold=20, minLineLength=20, maxLineGap=8)
    line_pixels = np.zeros_like(thresh_std)
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            cv2.line(line_pixels, (x1, y1), (x2, y2), 255, thickness=3)
    
    # Primary erase mask: crossing component AND confirmed Hough line
    erase_mask_hough = cv2.bitwise_and(erase_mask_inner, line_pixels)
    
    # Dilate the confirmed mask slightly to catch adjacent line fragments
    frag_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    erase_mask_expanded = cv2.dilate(erase_mask_hough, frag_kernel, iterations=1)
    
    # Full erase mask: crossing-component pixels that are near a confirmed line
    erase_mask_lines = cv2.bitwise_and(erase_mask_inner, erase_mask_expanded)
    
    # Stage 4: Visualize detected lines in red on the raw grayscale
    stage_lines = cv2.cvtColor(img_std, cv2.COLOR_GRAY2RGB)
    stage_lines[erase_mask_lines == 255] = [239, 68, 68]
    
    # Protect junction zones (where character strokes cross) using skeleton analysis
    skel = skeletonize(thresh_std)
    junctions = get_junctions(skel)
    protection_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    protected_zone = cv2.dilate(junctions, protection_kernel)
    
    # Remove protected pixels from erase mask
    erase_mask_clean = cv2.bitwise_and(erase_mask_lines, cv2.bitwise_not(protected_zone))
    
    # --- STAGE 5: Apply erase to the RAW grayscale — NO other preprocessing ---
    # Only confirmed line pixels are whited-out. Image is otherwise identical to the original crop.
    raw_clean = img_std.copy()
    raw_clean[erase_mask_clean == 255] = 255
    stage_clean = cv2.cvtColor(raw_clean, cv2.COLOR_GRAY2RGB)
    
    return stage_thresh, stage_denoised, stage_lines, stage_clean

@app.get("/api/crop")
def get_crop(filepath: str, cx: float, cy: float, w: float, h: float, angle: float, force_classify: bool = False, thumb_only: bool = False):
    """Returns a rectified crop and base64 strings of all intermediate preprocessing stages."""
    if not os.path.exists(filepath):
        raise HTTPException(status_code=404, detail="Source image file not found")
        
    try:
        with Image.open(filepath) as img:
            # Alpha compositing to white background
            if img.mode in ("RGBA", "LA") or (img.mode == "P" and "transparency" in img.info):
                background = Image.new("RGB", img.size, (255, 255, 255))
                mask = img.split()[3] if img.mode == "RGBA" else img.convert("RGBA").split()[3]
                background.paste(img, mask=mask)
                img_rgb = background
            else:
                img_rgb = img.convert("RGB")
                
            # Helper to convert PIL/numpy crop to base64
            def to_b64(crop_img):
                if isinstance(crop_img, np.ndarray):
                    crop_img = Image.fromarray(crop_img)
                buffered = io.BytesIO()
                crop_img.save(buffered, format="PNG")
                img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")
                return f"data:image/png;base64,{img_str}"

            if thumb_only:
                # Direct crop from raw RGB image (very fast)
                rect_crop_pil = rectify_crop(
                    img_rgb,
                    bbox_metrics={"cx": cx, "cy": cy, "w": w, "h": h, "angle": angle},
                    buffer_percent=0.08
                )
                return {"crop": to_b64(rect_crop_pil)}
            
            # Full crop details with pipeline preprocessing stages
            crop_pil_std = rectify_crop(
                img_rgb,
                bbox_metrics={"cx": cx, "cy": cy, "w": w, "h": h, "angle": angle},
                buffer_percent=0.10
            )
            crop_pil_big = rectify_crop(
                img_rgb,
                bbox_metrics={"cx": cx, "cy": cy, "w": w, "h": h, "angle": angle},
                buffer_percent=0.40
            )
            
            # Run local crop preprocessing stages
            s_thresh, s_denoised, s_lines, s_clean = clean_cad_lines_boundary_tracing_stages(crop_pil_std, crop_pil_big)
            
            # Convert raw crop to numpy array
            raw_crop_np = np.array(crop_pil_std.convert("RGB"))
            
            response = {
                "crop": to_b64(s_clean),
                "stages": {
                    "raw": to_b64(raw_crop_np),
                    "thresh": to_b64(s_thresh),
                    "denoised": to_b64(s_denoised),
                    "lines": to_b64(s_lines),
                    "clean": to_b64(s_clean)
                }
            }
            
            # If force_classify is set, run MobileNet classification on this crop
            if force_classify:
                _, classifier = get_models()
                # Run classifier on the final cleaned crop PIL
                clean_pil = Image.fromarray(s_clean).convert("RGB")
                class_name, class_conf = classifier.predict(clean_pil)
                char_display = CLASS_TO_CHAR.get(class_name, class_name)
                response.update({
                    "class_name": class_name,
                    "char_display": char_display,
                    "class_confidence": float(class_conf)
                })
                
            return response
            
    except Exception as e:
        import traceback
        err_msg = traceback.format_exc()
        print(f"Error getting crop stages: {e}\n{err_msg}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/clear-cache")
def clear_cache(req: ClearCacheRequest):
    """Deletes cached predictions for a specific image and model type to force recalculation."""
    if req.model_type == "paddle":
        cache_filename = f"paddle_{req.filename}.json"
    else:
        cache_filename = f"{req.filename}.json"
        
    cache_path = os.path.join(req.dir_path, "predictions", cache_filename)
    if os.path.exists(cache_path):
        try:
            os.remove(cache_path)
            print(f"Deleted cache file: {cache_path}")
            return {"success": True, "message": "Cache deleted successfully"}
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Failed to delete cache file: {str(e)}")
    return {"success": True, "message": "No cache file existed"}

# Serve Frontend Client (Mount static folder at /)
current_dir = os.path.dirname(os.path.abspath(__file__))
static_dir = os.path.join(current_dir, "static")
if os.path.exists(static_dir):
    app.mount("/", StaticFiles(directory=static_dir, html=True), name="static")

if __name__ == "__main__":
    # Launch on port 8000
    uvicorn.run(app, host="0.0.0.0", port=8000)
