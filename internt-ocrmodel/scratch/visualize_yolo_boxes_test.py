import os
import sys
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

sys.path.insert(0, "d:/Internship/OCR_PDF/internt-ocrmodel")

from pipeline import load_yolo_model

def run_yolo_boxes_only_visualization():
    test_dir = "d:/Internship/OCR_PDF/sign-detection-yolo8n/toTest"
    output_dir = os.path.join(test_dir, "predictions")
    os.makedirs(output_dir, exist_ok=True)
    
    # Locate YOLO model
    yolo_path = "../YOLO_expression_best.pt"
    if not os.path.exists(yolo_path):
        yolo_path = "D:/Internship/OCR_PDF/YOLO_expression_best.pt"
        
    if not os.path.exists(yolo_path):
        print(f"ERROR: YOLO model not found at {yolo_path}")
        return
        
    print(f"Loading YOLO model from: {yolo_path}...")
    yolo_model = load_yolo_model(yolo_path)
    
    # Supported image extensions
    valid_exts = (".png", ".jpg", ".jpeg", ".bmp", ".tiff")
    images = [f for f in os.listdir(test_dir) if f.lower().endswith(valid_exts)]
    
    if not images:
        print(f"No images found in {test_dir}")
        return
        
    print(f"Found {len(images)} images to process.")
    
    font_choices = [
        "C:\\Windows\\Fonts\\seguisym.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "C:\\Windows\\Fonts\\arial.ttf",
        "C:\\Windows\\Fonts\\calibri.ttf"
    ]
    font = None
    for path in font_choices:
        if os.path.exists(path):
            try:
                font = ImageFont.truetype(path, 16)
                break
            except:
                pass
    if font is None:
        font = ImageFont.load_default()
        
    for img_name in images:
        img_path = os.path.join(test_dir, img_name)
        print(f"Processing image: {img_name}...")
        
        try:
            page_img = Image.open(img_path)
            
            # Run YOLO OBB
            img_bgr = cv2.cvtColor(np.array(page_img.convert("RGB")), cv2.COLOR_RGB2BGR)
            results = yolo_model(img_bgr, verbose=False, conf=0.25, imgsz=1280)
            result = results[0]
            
            # Draw YOLO boxes on the original page canvas
            draw_img = page_img.convert("RGBA")
            draw = ImageDraw.Draw(draw_img)
            
            if result.obb is not None and len(result.obb) > 0:
                xyxyxyxy = result.obb.xyxyxyxy.cpu().numpy()
                yolo_confs = result.obb.conf.cpu().numpy()
                
                for idx, corners in enumerate(xyxyxyxy):
                    conf = yolo_confs[idx]
                    poly_pts = [(float(pt[0]), float(pt[1])) for pt in corners]
                    
                    # Draw OBB bounding box with semi-transparent cyan fill and outline
                    draw.polygon(poly_pts, fill=(0, 255, 255, 40), outline=(0, 255, 255, 255), width=2)
                    
                    # Draw box confidence label
                    lx, ly = float(corners[0][0]), float(corners[0][1])
                    draw.rectangle((lx - 2, ly - 20, lx + 75, ly), fill=(0, 150, 150, 230))
                    draw.text((lx + 2, ly - 18), f"CONF: {conf:.2f}", font=font, fill=(255, 255, 255))
            
            # Save final image
            base_name, _ = os.path.splitext(img_name)
            out_img_name = f"{base_name}_yolo_boxes.png"
            out_path = os.path.join(output_dir, out_img_name)
            draw_img.convert("RGB").save(out_path)
            print(f"  -> Saved output to: {out_path}")
            
            # Copy to current script folder
            dest = f"d:/Internship/OCR_PDF/internt-ocrmodel/scratch/{out_img_name}"
            draw_img.convert("RGB").save(dest)
            print(f"  -> Copied to: {dest}")
            
        except Exception as e:
            print(f"Error processing {img_name}: {e}")
            
    print("\nBounding box overlay complete!")

if __name__ == "__main__":
    run_yolo_boxes_only_visualization()
