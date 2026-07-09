import os
# Disable oneDNN/MKLDNN in PaddleX & PaddlePaddle
os.environ["FLAGS_enable_onednn"] = "0"
os.environ["PADDLE_PDX_ENABLE_MKLDNN_BYDEFAULT"] = "False"
os.environ["PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK"] = "True"

import sys
import json
import math
import numpy as np
from paddleocr import PaddleOCR

def get_angle_from_corners(corners):
    """Calculates box rotation angle in degrees from its 4 corners (TL, TR, BR, BL order)."""
    p1 = corners[0]
    p2 = corners[1]
    dx = p2[0] - p1[0]
    dy = p2[1] - p1[1]
    angle = math.degrees(math.atan2(dy, dx))
    return angle

def main():
    if len(sys.argv) < 2:
        print("ERROR: Image path argument is required")
        sys.exit(1)
        
    image_path = sys.argv[1]
    if not os.path.exists(image_path):
        print(f"ERROR: Image not found: {image_path}")
        sys.exit(1)
        
    try:
        # Initialize PaddleOCR with disabled document unwarping/orientation classification
        # to ensure that output coordinates map 1:1 without layout distortion.
        ocr = PaddleOCR(
            lang="en",
            use_doc_orientation_classify=False,
            use_doc_unwarping=False,
            use_textline_orientation=False
        )
        result = ocr.predict(image_path)
        
        res_list = list(result)
        if len(res_list) == 0:
            sys.stdout.buffer.write(json.dumps([]).encode('utf-8'))
            return
            
        ocr_result = res_list[0]
        boxes_data = []
        
        # Check if dt_polys exists in the result dictionary/object
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
            
        for idx, poly in enumerate(polys):
            # poly is a numpy array of 4 corners [[x0, y0], [x1, y1], [x2, y2], [x3, y3]]
            corners = poly.tolist() if hasattr(poly, 'tolist') else list(poly)
            
            # Calculate geometric details
            pts = np.array(corners, dtype=np.float32)
            cx, cy = np.mean(pts, axis=0)
            
            # Width: distance between TL and TR
            w = np.linalg.norm(pts[0] - pts[1])
            # Height: distance between TL and BL
            h = np.linalg.norm(pts[0] - pts[3])
            
            # Scale corners outwards by 10% around the center
            # to make the bounding boxes 10% larger than predicted.
            scaled_pts = pts + (pts - np.array([cx, cy])) * 0.10
            corners = scaled_pts.tolist()
            
            # Scale width and height as well
            w = w * 1.10
            h = h * 1.10
            cx, cy = np.mean(scaled_pts, axis=0)
            
            # Rotation angle
            angle = get_angle_from_corners(corners)
            
            # Get score/text if available
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
            
        # Write clean JSON output to stdout with utf-8 encoding
        sys.stdout.buffer.write(json.dumps(boxes_data, ensure_ascii=False).encode('utf-8'))
        
    except Exception as e:
        import traceback
        err_msg = traceback.format_exc()
        sys.stderr.write(f"ERROR: {str(e)}\n{err_msg}")
        sys.exit(1)

if __name__ == "__main__":
    main()
