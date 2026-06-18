import os
import easyocr
import cv2
import json
import numpy as np
from PIL import Image, ImageDraw, ImageFont

def load_font(size=14):
    for p in ["C:\\Windows\\Fonts\\seguisym.ttf",
              "C:\\Windows\\Fonts\\arial.ttf",
              "C:\\Windows\\Fonts\\calibri.ttf",
              "C:\\Windows\\Fonts\\segoeui.ttf"]:
        if os.path.exists(p):
            try:
                return ImageFont.truetype(p, size)
            except Exception:
                pass
    return ImageFont.load_default()

def run_easyocr(image_path, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    base = os.path.splitext(os.path.basename(image_path))[0]
    
    print(f"\n--- Running EasyOCR on {image_path} ---")
    
    # Initialize EasyOCR reader
    reader = easyocr.Reader(['en'], gpu=False)  # Run on CPU
    
    # Load image
    img_pil = Image.open(image_path).convert("RGB")
    draw = ImageDraw.Draw(img_pil, "RGBA")
    font = load_font(size=12)
    
    # Run EasyOCR readtext
    img_np = np.array(img_pil)
    results = reader.readtext(img_np)
    
    detections = []
    print(f"EasyOCR found {len(results)} text regions:")
    
    for idx, (bbox, text, conf) in enumerate(results):
        # bbox is a list of 4 points: [TL, TR, BR, BL]
        pts = [(float(pt[0]), float(pt[1])) for pt in bbox]
        
        # Save detection details
        detections.append({
            "id": idx + 1,
            "text": text,
            "confidence": float(conf),
            "bbox": pts
        })
        
        print(f"  #{idx+1}: '{text}' (conf: {conf:.2f})")
        
        # Draw red thin bounding box for EasyOCR (semi-transparent)
        draw.polygon(pts, outline=(255, 60, 60, 200), width=2)
        
        # Draw label text
        lx, ly = pts[0][0], pts[0][1] - 14
        draw.text((lx, ly), f"#{idx+1} '{text}' ({conf:.2f})", font=font, fill=(255, 0, 0, 255))
        
    # Save annotated image
    out_img_path = os.path.join(out_dir, f"{base}_easyocr.png")
    img_pil.save(out_img_path)
    print(f"Saved EasyOCR annotated image to: {out_img_path}")
    
    # Save JSON report
    out_json_path = os.path.join(out_dir, f"{base}_easyocr.json")
    with open(out_json_path, "w", encoding="utf-8") as f:
        json.dump(detections, f, indent=2, ensure_ascii=False)
    print(f"Saved EasyOCR JSON report to: {out_json_path}")
    
    return detections

def main():
    out_dir = "output_pipeline/visualizations"
    run_easyocr("toTest/1.jpeg", out_dir)
    run_easyocr("toTest/2.jpeg", out_dir)

if __name__ == "__main__":
    main()
