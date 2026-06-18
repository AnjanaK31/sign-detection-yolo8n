import os
import sys
import json
from PIL import Image, ImageDraw, ImageFont

try:
    import pytesseract
except ImportError:
    print("ERROR: pytesseract package is not installed.")
    sys.exit(1)

# List of common Tesseract installation paths on Windows to search
COMMON_PATHS = [
    r"C:\Program Files\Tesseract-OCR\tesseract.exe",
    r"C:\Program Files (x86)\Tesseract-OCR\tesseract.exe",
    r"C:\ProgramData\Tesseract-OCR\tesseract.exe",
    os.path.expandvars(r"%LOCALAPPDATA%\Tesseract-OCR\tesseract.exe")
]

def find_tesseract():
    # 1. Check if it's in PATH already
    import shutil
    in_path = shutil.which("tesseract")
    if in_path:
        return in_path
    
    # 2. Check common paths
    for p in COMMON_PATHS:
        if os.path.exists(p):
            return p
            
    return None

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

def run_tesseract(image_path, out_dir, tesseract_path):
    os.makedirs(out_dir, exist_ok=True)
    base = os.path.splitext(os.path.basename(image_path))[0]
    
    print(f"\n--- Running Tesseract on {image_path} ---")
    
    # Configure Tesseract path
    pytesseract.pytesseract.tesseract_cmd = tesseract_path
    print(f"Using Tesseract executable: {tesseract_path}")
    
    # Load image
    img_pil = Image.open(image_path).convert("RGB")
    draw = ImageDraw.Draw(img_pil, "RGBA")
    font = load_font(size=12)
    
    # Run image_to_data
    try:
        data = pytesseract.image_to_data(img_pil, output_type=pytesseract.Output.DICT)
    except Exception as e:
        print(f"ERROR: Tesseract run failed: {e}")
        return []
        
    detections = []
    num_boxes = len(data["text"])
    box_idx = 1
    
    for i in range(num_boxes):
        text = data["text"][i].strip()
        conf = float(data["conf"][i])
        
        # Filter out empty texts and noise
        if text and conf > 0:
            left = data["left"][i]
            top = data["top"][i]
            width = data["width"][i]
            height = data["height"][i]
            
            pts = [
                (left, top),
                (left + width, top),
                (left + width, top + height),
                (left, top + height)
            ]
            
            detections.append({
                "id": box_idx,
                "text": text,
                "confidence": conf / 100.0,
                "bbox": pts
            })
            
            print(f"  #{box_idx}: '{text}' (conf: {conf/100.0:.2f})")
            
            # Draw blue thin bounding box for Tesseract
            draw.polygon(pts, outline=(40, 100, 250, 200), width=2)
            
            # Draw label text
            lx, ly = pts[0][0], pts[0][1] - 14
            draw.text((lx, ly), f"#{box_idx} '{text}' ({conf/100.0:.2f})", font=font, fill=(0, 0, 255, 255))
            
            box_idx += 1
            
    # Save annotated image
    out_img_path = os.path.join(out_dir, f"{base}_tesseract.png")
    img_pil.save(out_img_path)
    print(f"Saved Tesseract annotated image to: {out_img_path}")
    
    # Save JSON report
    out_json_path = os.path.join(out_dir, f"{base}_tesseract.json")
    with open(out_json_path, "w", encoding="utf-8") as f:
        json.dump(detections, f, indent=2, ensure_ascii=False)
    print(f"Saved Tesseract JSON report to: {out_json_path}")
    
    return detections

def main():
    tesseract_path = find_tesseract()
    if not tesseract_path:
        print("\n" + "!"*60)
        print("Tesseract binary (tesseract.exe) was NOT found on your system!")
        print("Please download and install it from:")
        print("  https://github.com/UB-Mannheim/tesseract/wiki")
        print("After installation, make sure to add it to your PATH or edit this script")
        print("to set 'tesseract_path' to the correct location of 'tesseract.exe'.")
        print("!"*60 + "\n")
        sys.exit(1)
        
    out_dir = "output_pipeline/visualizations"
    run_tesseract("toTest/1.jpeg", out_dir, tesseract_path)
    run_tesseract("toTest/2.jpeg", out_dir, tesseract_path)

if __name__ == "__main__":
    main()
