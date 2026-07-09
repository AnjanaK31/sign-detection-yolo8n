import os
import sys
import json
import subprocess
import cv2
import numpy as np
from PIL import Image, ImageDraw

# Add LineRemovalNet and BoudningBoxCleaning to path
sys.path.append(os.path.abspath(r"d:\Internship\OCR_PDF\LineRemovalNet"))
sys.path.append(os.path.abspath(r"d:\Internship\OCR_PDF\BoudningBoxCleaning"))

from clean_page_expressions import clean_crop_lines, rectify_crop

def get_paddle_ocr_boxes(image_path, python_env, detect_script):
    """Runs paddle_detect.py as a subprocess and parses JSON output."""
    cmd = [python_env, detect_script, image_path]
    print(f"Running: {' '.join(cmd)}")
    result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if result.returncode != 0:
        stderr_str = result.stderr.decode('utf-8', errors='ignore')
        print(f"Error running paddle_detect.py:\n{stderr_str}")
        return []
    stdout_str = result.stdout.decode('utf-8', errors='ignore')
    try:
        return json.loads(stdout_str)
    except json.JSONDecodeError:
        print(f"Failed to parse JSON from stdout:\n{stdout_str}")
        return []

def main():
    python_paddle_env = r"d:\Internship\OCR_PDF\PaddleOCR\venv\Scripts\python.exe"
    paddle_detect_script = r"d:\Internship\OCR_PDF\sign-detection-yolo8n\paddle_detect.py"
    
    img_path = r"d:\Internship\OCR_PDF\sign-detection-yolo8n\toTest\Screenshot 2026-06-24 145555.png"
    output_dir = r"d:\Internship\OCR_PDF\sign-detection-yolo8n\toTest\paddle_crops_cleaned_heuristics_3rd"
    
    os.makedirs(output_dir, exist_ok=True)
    
    if not os.path.exists(img_path):
        print(f"Error: Image not found at {img_path}")
        return
        
    print(f"Loading image: {os.path.basename(img_path)}...")
    orig_img_bgr = cv2.imread(img_path)
    if orig_img_bgr is None:
        print(f"Failed to load image: {img_path}")
        return
        
    # Get OCR boxes
    boxes = get_paddle_ocr_boxes(img_path, python_paddle_env, paddle_detect_script)
    print(f"Found {len(boxes)} text regions on page.")
    
    processed_count = 0
    for idx, box in enumerate(boxes):
        cx = box['cx']
        cy = box['cy']
        w = box['w']
        h = box['h']
        angle = box['angle']
        text = box.get('char_display', 'Text')
        
        # Extract crops for Heuristic pipeline (20% and 40% buffers)
        crop_pil_std = rectify_crop(
            Image.fromarray(cv2.cvtColor(orig_img_bgr, cv2.COLOR_BGR2RGB)),
            bbox_metrics={'cx': cx, 'cy': cy, 'w': w, 'h': h, 'angle': angle},
            buffer_percent=0.20
        )
        crop_pil_big = rectify_crop(
            Image.fromarray(cv2.cvtColor(orig_img_bgr, cv2.COLOR_BGR2RGB)),
            bbox_metrics={'cx': cx, 'cy': cy, 'w': w, 'h': h, 'angle': angle},
            buffer_percent=0.40
        )
        
        # Run Heuristics Line Cleaning
        try:
            cleaned_pil_heur, _, _ = clean_crop_lines(crop_pil_std, crop_pil_big)
            
            # Concatenate original (crop_pil_std) and cleaned (cleaned_pil_heur) side-by-side
            w_std, h_std = crop_pil_std.size
            w_heur, h_heur = cleaned_pil_heur.size
            
            combined_w = w_std + w_heur + 10  # 10px divider
            combined_h = max(h_std, h_heur)
            
            # Create a white canvas
            combined_img = Image.new("RGB", (combined_w, combined_h), (255, 255, 255))
            
            # Paste original (left) and cleaned (right)
            combined_img.paste(crop_pil_std.convert("RGB"), (0, 0))
            combined_img.paste(cleaned_pil_heur.convert("RGB"), (w_std + 10, 0))
            
            # Draw a red boundary line (2px wide) in the middle of the divider
            draw = ImageDraw.Draw(combined_img)
            draw.line([(w_std + 5, 0), (w_std + 5, combined_h)], fill=(255, 0, 0), width=2)
            
            save_name = f"crop_{idx:03d}_cleaned.png"
            save_path = os.path.join(output_dir, save_name)
            combined_img.save(save_path)
            processed_count += 1
            print(f"  Saved crop {idx:02d} (Original + Cleaned) (Text: '{text}') -> {save_name}")
        except Exception as e:
            print(f"  Heuristics failed on crop {idx}: {e}")
            
    print(f"\nSuccessfully cleaned and saved {processed_count}/{len(boxes)} crops to: {output_dir}")

if __name__ == "__main__":
    main()
