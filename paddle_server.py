import os
# Disable oneDNN/MKLDNN in PaddleX & PaddlePaddle
os.environ["FLAGS_enable_onednn"] = "0"
os.environ["PADDLE_PDX_ENABLE_MKLDNN_BYDEFAULT"] = "False"
os.environ["PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK"] = "True"

import base64
import cv2
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from paddleocr import PaddleOCR
import numpy as np
import math

app = FastAPI(title="PaddleOCR Background Inference Service")

# Initialize PaddleOCR on startup once
print("Initializing persistent PaddleOCR model...")
ocr = PaddleOCR(
    lang="en",
    use_doc_orientation_classify=False,
    use_doc_unwarping=False,
    use_textline_orientation=False
)
print("PaddleOCR model initialized and ready!")

class PredictionRequest(BaseModel):
    image_path: str

def get_angle_from_corners(corners):
    """Calculates box rotation angle in degrees from its 4 corners (TL, TR, BR, BL order)."""
    p1 = corners[0]
    p2 = corners[1]
    dx = p2[0] - p1[0]
    dy = p2[1] - p1[1]
    angle = math.degrees(math.atan2(dy, dx))
    return angle

@app.post("/predict")
def predict_ocr(req: PredictionRequest):
    if not os.path.exists(req.image_path):
        raise HTTPException(status_code=404, detail=f"Image not found at path: {req.image_path}")
        
    try:
        result = ocr.predict(req.image_path)
        res_list = list(result)
        if len(res_list) == 0:
            return []
            
        ocr_result = res_list[0]
        polys = []
        texts = []
        scores = []
        
        if isinstance(ocr_result, dict):
            polys = ocr_result.get('dt_polys', [])
            texts = ocr_result.get('rec_texts', [])
            scores = ocr_result.get('rec_scores', [])
        elif hasattr(ocr_result, 'dt_polys'):
            polys = getattr(ocr_result, 'dt_polys', [])
            texts = getattr(ocr_result, 'rec_texts', [])
            scores = getattr(ocr_result, 'rec_scores', [])
            
        boxes_data = []
        for idx, poly in enumerate(polys):
            corners = poly.tolist() if hasattr(poly, 'tolist') else list(poly)
            
            # Calculate geometric details
            pts = np.array(corners, dtype=np.float32)
            cx, cy = np.mean(pts, axis=0)
            
            # Width & height
            w = np.linalg.norm(pts[0] - pts[1])
            h = np.linalg.norm(pts[0] - pts[3])
            
            # Scale corners outwards by 10% around the center
            # to make the bounding boxes 10% larger than predicted.
            scaled_pts = pts + (pts - np.array([cx, cy])) * 0.10
            corners = scaled_pts.tolist()
            w = w * 1.10
            h = h * 1.10
            cx, cy = np.mean(scaled_pts, axis=0)
            
            # Rotation angle
            angle = get_angle_from_corners(corners)
            
            score = 1.0
            if idx < len(scores):
                score = float(scores[idx])
                
            text = "Text"
            if idx < len(texts):
                text = texts[idx]
                
            boxes_data.append({
                "id": idx,
                "cx": float(cx),
                "cy": float(cy),
                "w": float(w),
                "h": float(h),
                "angle": float(angle),
                "corners": corners,
                "confidence": float(score),
                "class_name": "text",
                "char_display": text,
                "class_confidence": float(score)
            })
            
        return boxes_data
        
    except Exception as e:
        import traceback
        err_msg = traceback.format_exc()
        print(f"Prediction error: {e}\n{err_msg}")
        raise HTTPException(status_code=500, detail=str(e))

class DetectRequest(BaseModel):
    image_path: str

class RecognizeRequest(BaseModel):
    image_b64: str

@app.post("/detect")
def detect_only(req: DetectRequest):
    if not os.path.exists(req.image_path):
        raise HTTPException(status_code=404, detail=f"Image not found at path: {req.image_path}")
        
    try:
        # Load image via cv2
        img = cv2.imread(req.image_path)
        if img is None or img.size == 0:
            raise HTTPException(status_code=400, detail="Failed to load image")
            
        # Run detection only
        det_results = list(ocr.paddlex_pipeline.text_det_model([img]))
        if not det_results:
            return []
            
        polys = det_results[0]["dt_polys"]
        scores = det_results[0]["dt_scores"]
        
        boxes_data = []
        for idx, poly in enumerate(polys):
            corners = poly.tolist() if hasattr(poly, 'tolist') else list(poly)
            
            # Calculate geometric details
            pts = np.array(corners, dtype=np.float32)
            cx, cy = np.mean(pts, axis=0)
            
            # Width & height
            w = np.linalg.norm(pts[0] - pts[1])
            h = np.linalg.norm(pts[0] - pts[3])
            
            # Scale corners outwards by 10% around the center
            scaled_pts = pts + (pts - np.array([cx, cy])) * 0.10
            corners = scaled_pts.tolist()
            w = w * 1.10
            h = h * 1.10
            cx, cy = np.mean(scaled_pts, axis=0)
            
            # Rotation angle
            angle = get_angle_from_corners(corners)
            
            score = 1.0
            if idx < len(scores):
                score = float(scores[idx])
                
            boxes_data.append({
                "id": idx,
                "cx": float(cx),
                "cy": float(cy),
                "w": float(w),
                "h": float(h),
                "angle": float(angle),
                "corners": corners,
                "confidence": float(score),
                "class_name": "text",
                "char_display": "Pending",
                "class_confidence": 0.0
            })
            
        return boxes_data
        
    except Exception as e:
        import traceback
        err_msg = traceback.format_exc()
        print(f"Detection error: {e}\n{err_msg}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/recognize_crop")
def recognize_crop(req: RecognizeRequest):
    try:
        # Decode base64 to image
        img_bytes = base64.b64decode(req.image_b64.split(",")[-1])
        nparr = np.frombuffer(img_bytes, np.uint8)
        img_bgr = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if img_bgr is None or img_bgr.size == 0:
            raise HTTPException(status_code=400, detail="Invalid crop image data")
            
        # Run recognition only
        rec_results = list(ocr.paddlex_pipeline.text_rec_model([img_bgr]))
        if not rec_results:
            return {"text": "", "score": 0.0}
            
        text = rec_results[0]["rec_text"]
        score = float(rec_results[0]["rec_score"])
        return {"text": text, "score": score}
        
    except Exception as e:
        import traceback
        err_msg = traceback.format_exc()
        print(f"Recognition error: {e}\n{err_msg}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8001)
