import os
import sys
import json
import subprocess
import cv2
import numpy as np
import torch
from PIL import Image

# Add LineRemovalNet and BoudningBoxCleaning to path
sys.path.append(os.path.abspath(r"d:\Internship\OCR_PDF\LineRemovalNet"))
sys.path.append(os.path.abspath(r"d:\Internship\OCR_PDF\BoudningBoxCleaning"))

from models.unet import UNet
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

def pad_to_multiple(img, multiple=16):
    h, w = img.shape[:2]
    pad_h = (multiple - h % multiple) % multiple
    pad_w = (multiple - w % multiple) % multiple
    padded = cv2.copyMakeBorder(img, 0, pad_h, 0, pad_w, cv2.BORDER_CONSTANT, value=255)
    return padded, pad_h, pad_w

def clean_crop_unet(model, device, crop_img, threshold_val=180):
    """Cleans a single crop using the U-Net model."""
    if isinstance(crop_img, Image.Image):
        img_gray = np.array(crop_img.convert("L"))
    else:
        if len(crop_img.shape) == 3:
            img_gray = cv2.cvtColor(crop_img, cv2.COLOR_BGR2GRAY)
        else:
            img_gray = crop_img.copy()

    orig_h, orig_w = img_gray.shape[:2]
    
    padded_img, pad_h, pad_w = pad_to_multiple(img_gray, 16)
    img_np = padded_img.astype(np.float32) / 255.0
    img_tensor = torch.from_numpy(img_np).unsqueeze(0).unsqueeze(0).to(device)
    
    with torch.no_grad():
        output_tensor = model(img_tensor)
        
    output_np = output_tensor.squeeze().cpu().numpy()
    output_scaled = (output_np * 255.0).astype(np.uint8)
    cropped_out = output_scaled[0:orig_h, 0:orig_w]
    _, final_thresh = cv2.threshold(cropped_out, threshold_val, 255, cv2.THRESH_BINARY)
    return final_thresh

def generate_dashboard(output_html, info_list):
    import base64
    
    def get_base64_img(img_path):
        if not os.path.exists(img_path):
            return ""
        try:
            with open(img_path, "rb") as f:
                data = f.read()
            b64 = base64.b64encode(data).decode('utf-8')
            return f"data:image/png;base64,{b64}"
        except Exception as e:
            print(f"Error encoding {img_path}: {e}")
            return ""

    cards_html = []
    for info in info_list:
        raw_b64 = get_base64_img(info['raw_path'])
        heur_b64 = get_base64_img(info['heur_path'])
        unet_b64 = get_base64_img(info['unet_path'])
        edge_b64 = get_base64_img(info['edge_path'])
        
        card = f"""
        <div class="crop-card">
            <div class="crop-info">
                <div class="crop-name">Crop {info['idx']}</div>
                <div class="crop-text">Detected Text: <span class="badge">{info['text']}</span></div>
            </div>
            <div class="panels">
                <div class="panel">
                    <div class="panel-label">Raw Crop</div>
                    <div class="img-wrapper">
                        <img src="{raw_b64}" alt="Original">
                    </div>
                </div>
                <div class="panel">
                    <div class="panel-label">Heuristic Cleaned</div>
                    <div class="img-wrapper" style="background: white;">
                        <img src="{heur_b64}" alt="Heuristic">
                    </div>
                </div>
                <div class="panel">
                    <div class="panel-label">U-Net Cleaned</div>
                    <div class="img-wrapper" style="background: white;">
                        <img src="{unet_b64}" alt="U-Net">
                    </div>
                </div>
                <div class="panel">
                    <div class="panel-label">Edge Detection</div>
                    <div class="img-wrapper" style="background: white;">
                        <img src="{edge_b64}" alt="Edge Detection">
                    </div>
                </div>
            </div>
        </div>"""
        cards_html.append(card)

    cards_joined = "\n".join(cards_html)
    html_content = f"""<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <title>PaddleOCR 3rd Image: Heuristics vs. U-Net comparison</title>
    <link href="https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;600;800&display=swap" rel="stylesheet">
    <style>
        body {{
            font-family: 'Outfit', sans-serif;
            background: linear-gradient(135deg, #0b0f19, #1e1b4b);
            color: #f1f5f9;
            margin: 0;
            padding: 40px 20px;
            min-height: 100vh;
        }}
        h1 {{
            text-align: center;
            font-weight: 800;
            font-size: 2.5rem;
            background: linear-gradient(to right, #38bdf8, #818cf8, #c084fc);
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
            max-width: 1200px;
            margin: 0 auto;
        }}
        .crop-card {{
            background: rgba(30, 41, 59, 0.6);
            backdrop-filter: blur(12px);
            border: 1px solid rgba(255, 255, 255, 0.08);
            border-radius: 16px;
            box-shadow: 0 10px 25px -5px rgba(0, 0, 0, 0.3);
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
            border-color: rgba(129, 140, 248, 0.5);
            box-shadow: 0 20px 25px -5px rgba(129, 140, 248, 0.15);
        }}
        .crop-info {{
            flex: 0.8;
            min-width: 200px;
        }}
        .crop-name {{
            font-weight: 600;
            font-size: 1.25rem;
            color: #f8fafc;
            margin-bottom: 8px;
        }}
        .crop-text {{
            font-size: 0.95rem;
            color: #cbd5e1;
        }}
        .badge {{
            background: #4f46e5;
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
            flex: 3;
            justify-content: flex-end;
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
            min-width: 220px;
            min-height: 80px;
        }}
        img {{
            max-height: 100px;
            max-width: 260px;
            object-fit: contain;
            display: block;
        }}
    </style>
</head>
<body>
    <div class="container">
        <h1>PaddleOCR Cleaning Performance: Heuristics vs. U-Net</h1>
        <div class="subtitle">Comparison of Heuristic Line Cleaning with Curved Line Handling versus U-Net Neural Network Line Removal on crops from the 3rd image.</div>
        {cards_joined}
    </div>
</body>
</html>
"""
    with open(output_html, "w", encoding="utf-8") as f:
        f.write(html_content)
    print(f"Generated comparison dashboard: {output_html}")

def main():
    python_paddle_env = r"d:\Internship\OCR_PDF\PaddleOCR\venv\Scripts\python.exe"
    paddle_detect_script = r"d:\Internship\OCR_PDF\sign-detection-yolo8n\paddle_detect.py"
    unet_model_path = r"d:\Internship\OCR_PDF\LineRemovalNet\best_model.pth"
    
    img_path = r"d:\Internship\OCR_PDF\sign-detection-yolo8n\toTest\Screenshot 2026-06-24 145555.png"
    
    # Outputs folders in scratch
    scratch_dir = r"d:\Internship\OCR_PDF\BoudningBoxCleaning\scratch"
    output_dir_raw = os.path.join(scratch_dir, "paddle_3rd_raw")
    output_dir_heur = os.path.join(scratch_dir, "paddle_3rd_heur")
    output_dir_unet = os.path.join(scratch_dir, "paddle_3rd_unet")
    output_dir_edge = os.path.join(scratch_dir, "paddle_3rd_edge")
    dashboard_path = os.path.join(scratch_dir, "paddle_comparison_3rd_image.html")
    
    os.makedirs(output_dir_raw, exist_ok=True)
    os.makedirs(output_dir_heur, exist_ok=True)
    os.makedirs(output_dir_unet, exist_ok=True)
    os.makedirs(output_dir_edge, exist_ok=True)
    
    # Initialize U-Net Model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device for U-Net: {device}")
    
    if not os.path.exists(unet_model_path):
        print(f"Error: U-Net Model weights not found at {unet_model_path}")
        return
        
    model = UNet(n_channels=1, n_classes=1, bilinear=False, base_channels=32).to(device)
    model.load_state_dict(torch.load(unet_model_path, map_location=device))
    model.eval()
    print("Loaded U-Net weights successfully!")
    
    if not os.path.exists(img_path):
        print(f"Error: Image not found at {img_path}")
        return
        
    print(f"\nProcessing {os.path.basename(img_path)}...")
    orig_img_bgr = cv2.imread(img_path)
    if orig_img_bgr is None:
        print(f"Failed to load image: {img_path}")
        return
        
    # Get OCR boxes
    boxes = get_paddle_ocr_boxes(img_path, python_paddle_env, paddle_detect_script)
    print(f"Found {len(boxes)} text regions.")
    
    info_list = []
    
    for idx, box in enumerate(boxes):
        cx = box['cx']
        cy = box['cy']
        w = box['w']
        h = box['h']
        angle = box['angle']
        text = box.get('char_display', 'Text')
        
        # 1. Extract raw crop (10% buffer)
        crop_raw = rectify_crop(
            orig_img_bgr,
            bbox_metrics={'cx': cx, 'cy': cy, 'w': w, 'h': h, 'angle': angle},
            buffer_percent=0.10
        )
        raw_fname = f"crop_{idx}.png"
        raw_path = os.path.join(output_dir_raw, raw_fname)
        cv2.imwrite(raw_path, crop_raw)
        
        # 2. Extract crops for Heuristic pipeline (20% and 40% buffers)
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
            heur_fname = f"crop_{idx}.png"
            heur_path = os.path.join(output_dir_heur, heur_fname)
            cleaned_pil_heur.save(heur_path)
        except Exception as e:
            print(f"Heuristics failed on crop {idx}: {e}")
            # Fallback to saving raw crop
            heur_fname = f"crop_{idx}.png"
            heur_path = os.path.join(output_dir_heur, heur_fname)
            cv2.imwrite(heur_path, crop_raw)
            
        # 3. Run U-Net Line Cleaning on the standard heuristic crop
        try:
            cleaned_unet_np = clean_crop_unet(model, device, crop_pil_std)
            unet_fname = f"crop_{idx}.png"
            unet_path = os.path.join(output_dir_unet, unet_fname)
            cv2.imwrite(unet_path, cleaned_unet_np)
        except Exception as e:
            print(f"U-Net failed on crop {idx}: {e}")
            # Fallback to saving raw crop
            unet_fname = f"crop_{idx}.png"
            unet_path = os.path.join(output_dir_unet, unet_fname)
            cv2.imwrite(unet_path, crop_raw)
            
        # 4. Run Edge Detection (Canny)
        try:
            crop_gray = cv2.cvtColor(crop_raw, cv2.COLOR_BGR2GRAY) if len(crop_raw.shape) == 3 else crop_raw.copy()
            edges = cv2.Canny(crop_gray, 50, 150)
            edges_inv = cv2.bitwise_not(edges)
            edge_fname = f"crop_{idx}.png"
            edge_path = os.path.join(output_dir_edge, edge_fname)
            cv2.imwrite(edge_path, edges_inv)
        except Exception as e:
            print(f"Edge detection failed on crop {idx}: {e}")
            edge_fname = f"crop_{idx}.png"
            edge_path = os.path.join(output_dir_edge, edge_fname)
            cv2.imwrite(edge_path, crop_raw)

        info_list.append({
            'idx': idx,
            'raw_path': raw_path,
            'heur_path': heur_path,
            'unet_path': unet_path,
            'edge_path': edge_path,
            'text': text
        })
        
    print(f"\nProcessed {len(info_list)} crops in total.")
    generate_dashboard(dashboard_path, info_list)
    print("Done!")

if __name__ == "__main__":
    main()
