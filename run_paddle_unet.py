import os
import sys
import json
import subprocess
import cv2
import numpy as np
import torch
from PIL import Image

# Add LineRemovalNet to path so we can import the UNet model
sys.path.append(os.path.abspath("../LineRemovalNet"))
from models.unet import UNet
from rectifier import rectify_crop

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


def pad_to_multiple(img, multiple=16):
    h, w = img.shape[:2]
    pad_h = (multiple - h % multiple) % multiple
    pad_w = (multiple - w % multiple) % multiple
    padded = cv2.copyMakeBorder(img, 0, pad_h, 0, pad_w, cv2.BORDER_CONSTANT, value=255)
    return padded, pad_h, pad_w

def clean_crop(model, device, crop_img, threshold_val=180):
    """Cleans a single crop using the U-Net model."""
    # Convert PIL Image or RGB/BGR numpy array to grayscale
    if isinstance(crop_img, Image.Image):
        img_gray = np.array(crop_img.convert("L"))
    else:
        if len(crop_img.shape) == 3:
            img_gray = cv2.cvtColor(crop_img, cv2.COLOR_BGR2GRAY)
        else:
            img_gray = crop_img.copy()

    orig_h, orig_w = img_gray.shape[:2]
    
    # Pad to multiple of 16
    padded_img, pad_h, pad_w = pad_to_multiple(img_gray, 16)
    
    # Scale to [0, 1] and add batch/channel dimensions: (1, 1, H, W)
    img_np = padded_img.astype(np.float32) / 255.0
    img_tensor = torch.from_numpy(img_np).unsqueeze(0).unsqueeze(0).to(device)
    
    with torch.no_grad():
        output_tensor = model(img_tensor)
        
    # Squeeze back and move to CPU
    output_np = output_tensor.squeeze().cpu().numpy()
    
    # Scale back to [0, 255]
    output_scaled = (output_np * 255.0).astype(np.uint8)
    
    # Crop back to original dimensions
    cropped_out = output_scaled[0:orig_h, 0:orig_w]
    
    # Re-apply threshold to make it a sharp binary mask
    _, final_thresh = cv2.threshold(cropped_out, threshold_val, 255, cv2.THRESH_BINARY)
    
    return final_thresh

def generate_dashboard(orig_dir, cleaned_dir, output_html, info_list):
    import base64
    
    def get_base64_img(img_path):
        if not os.path.exists(img_path):
            return ""
        try:
            with open(img_path, "rb") as f:
                data = f.read()
            b64 = base64.b64encode(data).decode('utf-8')
            ext = os.path.splitext(img_path)[1].lower().replace('.', '')
            if ext == 'jpg':
                ext = 'jpeg'
            return f"data:image/{ext};base64,{b64}"
        except Exception as e:
            print(f"Error encoding {img_path}: {e}")
            return ""

    cards_html = []
    for info in info_list:
        raw_path = info['raw_path']
        clean_path = info['clean_path']
        fname = os.path.basename(raw_path)
        
        raw_b64 = get_base64_img(raw_path)
        clean_b64 = get_base64_img(clean_path)
        
        if raw_b64 and clean_b64:
            card = f"""
            <div class="crop-card">
                <div class="crop-info">
                    <div class="crop-name">{fname}</div>
                    <div class="crop-meta">Image Source: {info['source_image']}</div>
                    <div class="crop-text">PaddleOCR Text: <span class="badge">{info['text']}</span></div>
                </div>
                <div class="panels">
                    <div class="panel">
                        <div class="panel-label">Raw PaddleOCR Crop (No Preprocessing)</div>
                        <div class="img-wrapper">
                            <img src="{raw_b64}" alt="Original">
                        </div>
                    </div>
                    <div class="panel">
                        <div class="panel-label">Cleaned (U-Net)</div>
                        <div class="img-wrapper" style="background: white;">
                            <img src="{clean_b64}" alt="Cleaned">
                        </div>
                    </div>
                </div>
            </div>"""
            cards_html.append(card)

    html_content = f"""<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <title>PaddleOCR Raw vs. U-Net Cleaned Crop Comparison</title>
    <link href="https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;600;800&display=swap" rel="stylesheet">
    <style>
        body {{
            font-family: 'Outfit', sans-serif;
            background: linear-gradient(135deg, #0f172a, #1e293b);
            color: #f1f5f9;
            margin: 0;
            padding: 40px 20px;
            min-height: 100vh;
        }}
        h1 {{
            text-align: center;
            font-weight: 800;
            font-size: 2.5rem;
            background: linear-gradient(to right, #38bdf8, #818cf8);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            margin-bottom: 10px;
        }}
        .subtitle {{
            text-align: center;
            color: #94a3b8;
            font-size: 1.1rem;
            margin-bottom: 40px;
        }}
        .container {{
            max-width: 1100px;
            margin: 0 auto;
        }}
        .crop-card {{
            background: rgba(30, 41, 59, 0.7);
            backdrop-filter: blur(10px);
            border: 1px solid rgba(255, 255, 255, 0.08);
            border-radius: 16px;
            box-shadow: 0 10px 25px -5px rgba(0, 0, 0, 0.3), 0 8px 10px -6px rgba(0, 0, 0, 0.3);
            padding: 24px;
            margin-bottom: 24px;
            display: flex;
            align-items: center;
            justify-content: space-between;
            gap: 20px;
            transition: all 0.3s ease;
        }}
        .crop-card:hover {{
            transform: translateY(-4px);
            border-color: rgba(56, 189, 248, 0.4);
            box-shadow: 0 20px 25px -5px rgba(56, 189, 248, 0.1), 0 10px 10px -5px rgba(0, 0, 0, 0.2);
        }}
        .crop-info {{
            flex: 1;
            min-width: 250px;
        }}
        .crop-name {{
            font-weight: 600;
            font-size: 1.25rem;
            color: #f8fafc;
            margin-bottom: 8px;
            word-break: break-all;
        }}
        .crop-meta {{
            font-size: 0.9rem;
            color: #64748b;
            margin-bottom: 12px;
        }}
        .crop-text {{
            font-size: 0.95rem;
            color: #cbd5e1;
        }}
        .badge {{
            background: #3b82f6;
            color: white;
            padding: 3px 8px;
            border-radius: 6px;
            font-weight: 600;
            font-family: monospace;
            font-size: 0.9rem;
        }}
        .panels {{
            display: flex;
            gap: 24px;
            flex-wrap: wrap;
        }}
        .panel {{
            text-align: center;
        }}
        .panel-label {{
            font-size: 0.75rem;
            color: #94a3b8;
            margin-bottom: 8px;
            text-transform: uppercase;
            letter-spacing: 1px;
            font-weight: 600;
        }}
        .img-wrapper {{
            background: #0f172a;
            padding: 12px;
            border-radius: 10px;
            border: 1px dashed rgba(255, 255, 255, 0.15);
            display: flex;
            align-items: center;
            justify-content: center;
            min-width: 120px;
            min-height: 60px;
            transition: border-color 0.2s;
        }}
        img {{
            max-height: 120px;
            max-width: 300px;
            object-fit: contain;
            display: block;
        }}
    </style>
</head>
<body>
    <div class="container">
        <h1>PaddleOCR Raw vs. U-Net Cleaned Crops</h1>
        <div class="subtitle">Showing crops extracted using PaddleOCR coordinates on original images directly, with zero preprocessing or noise reduction, cleaned via U-Net.</div>
        {"\n".join(cards_html)}
    </div>
</body>
</html>
"""
    with open(output_html, "w", encoding="utf-8") as f:
        f.write(html_content)
    print(f"Generated comparison dashboard: {output_html}")

def main():
    python_paddle_env = r"d:\Internship\OCR_PDF\PaddleOCR\venv\Scripts\python.exe"
    paddle_detect_script = r"paddle_detect.py"
    model_path = r"d:\Internship\OCR_PDF\LineRemovalNet\best_model.pth"
    
    test_images = [
        r"d:\Internship\OCR_PDF\sign-detection-yolo8n\toTest\1.jpeg",
        r"d:\Internship\OCR_PDF\sign-detection-yolo8n\toTest\2.jpeg"
    ]
    
    output_dir_raw = r"d:\Internship\OCR_PDF\sign-detection-yolo8n\toTest\paddle_crops_raw"
    output_dir_cleaned = r"d:\Internship\OCR_PDF\sign-detection-yolo8n\toTest\paddle_crops_cleaned"
    dashboard_path = r"d:\Internship\OCR_PDF\sign-detection-yolo8n\toTest\crop_comparison.html"
    
    os.makedirs(output_dir_raw, exist_ok=True)
    os.makedirs(output_dir_cleaned, exist_ok=True)
    
    # Initialize U-Net Model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device for U-Net inference: {device}")
    
    if not os.path.exists(model_path):
        print(f"Error: Model weights not found at {model_path}")
        return
        
    model = UNet(n_channels=1, n_classes=1, bilinear=False, base_channels=32).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    print("Loaded U-Net weights successfully!")
    
    info_list = []
    
    for img_path in test_images:
        if not os.path.exists(img_path):
            print(f"Warning: Test image not found at {img_path}")
            continue
            
        print(f"\nProcessing {os.path.basename(img_path)}...")
        
        # Load the original image (fully raw, no pre-processing or binarization)
        orig_img_bgr = cv2.imread(img_path)
        if orig_img_bgr is None:
            print(f"Failed to load image: {img_path}")
            continue
            
        # Get OCR boxes
        boxes = get_paddle_ocr_boxes(img_path, python_paddle_env, paddle_detect_script)
        print(f"Found {len(boxes)} text regions.")
        
        for idx, box in enumerate(boxes):
            cx = box['cx']
            cy = box['cy']
            w = box['w']
            h = box['h']
            angle = box['angle']
            text = box.get('char_display', 'Text')
            
            # Extract crop using rectify_crop with 10% buffer
            crop_pil = rectify_crop(
                orig_img_bgr,
                bbox_metrics={'cx': cx, 'cy': cy, 'w': w, 'h': h, 'angle': angle},
                buffer_percent=0.10
            )
            
            # Save raw crop
            raw_fname = f"{os.path.splitext(os.path.basename(img_path))[0]}_crop_{idx}.png"
            raw_path = os.path.join(output_dir_raw, raw_fname)
            cv2.imwrite(raw_path, crop_pil)
            
            # Run U-Net clean
            cleaned_img = clean_crop(model, device, crop_pil)
            clean_fname = f"{os.path.splitext(os.path.basename(img_path))[0]}_crop_{idx}.png"
            clean_path = os.path.join(output_dir_cleaned, clean_fname)
            cv2.imwrite(clean_path, cleaned_img)
            
            info_list.append({
                'raw_path': raw_path,
                'clean_path': clean_path,
                'source_image': os.path.basename(img_path),
                'text': text
            })
            
    print(f"\nProcessed {len(info_list)} crops in total.")
    generate_dashboard(output_dir_raw, output_dir_cleaned, dashboard_path, info_list)

if __name__ == "__main__":
    main()
