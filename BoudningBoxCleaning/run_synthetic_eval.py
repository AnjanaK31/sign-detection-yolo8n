import os
import sys
import glob
import cv2
import numpy as np
import torch
import base64
from PIL import Image

# Add LineRemovalNet to path so we can import the UNet model
sys.path.append(os.path.abspath("../LineRemovalNet"))
from models.unet import UNet

def pad_to_multiple(img, multiple=16):
    h, w = img.shape[:2]
    pad_h = (multiple - h % multiple) % multiple
    pad_w = (multiple - w % multiple) % multiple
    padded = cv2.copyMakeBorder(img, 0, pad_h, 0, pad_w, cv2.BORDER_CONSTANT, value=255)
    return padded, pad_h, pad_w

def clean_crop(model, device, img_path, threshold_val=180):
    img_gray = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    if img_gray is None:
        return None
        
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

def generate_dashboard(inputs_dir, targets_dir, cleaned_dir, output_html, info_list):
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
        fname = info['filename']
        input_path = os.path.join(inputs_dir, fname)
        target_path = os.path.join(targets_dir, fname)
        clean_path = os.path.join(cleaned_dir, fname)
        
        input_b64 = get_base64_img(input_path)
        target_b64 = get_base64_img(target_path)
        clean_b64 = get_base64_img(clean_path)
        
        if input_b64 and target_b64 and clean_b64:
            card = f"""
            <div class="crop-card">
                <div class="crop-info">
                    <div class="crop-name">{fname}</div>
                    <div class="crop-meta">Synthetic Evaluation Crop</div>
                </div>
                <div class="panels">
                    <div class="panel">
                        <div class="panel-label">Dirty Input (CAD Lines)</div>
                        <div class="img-wrapper">
                            <img src="{input_b64}" alt="Input">
                        </div>
                    </div>
                    <div class="panel">
                        <div class="panel-label">Cleaned (U-Net)</div>
                        <div class="img-wrapper" style="background: white;">
                            <img src="{clean_b64}" alt="Cleaned">
                        </div>
                    </div>
                    <div class="panel">
                        <div class="panel-label">Target (Ground Truth)</div>
                        <div class="img-wrapper" style="background: white;">
                            <img src="{target_b64}" alt="Target">
                        </div>
                    </div>
                </div>
            </div>"""
            cards_html.append(card)

    html_content = f"""<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <title>U-Net Synthetic Model Evaluation</title>
    <link href="https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;600;800&display=swap" rel="stylesheet">
    <style>
        body {{
            font-family: 'Outfit', sans-serif;
            background: linear-gradient(135deg, #090d16, #111827);
            color: #f3f4f6;
            margin: 0;
            padding: 40px 20px;
            min-height: 100vh;
        }}
        h1 {{
            text-align: center;
            font-weight: 800;
            font-size: 2.6rem;
            background: linear-gradient(to right, #60a5fa, #a78bfa);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            margin-bottom: 8px;
        }}
        .subtitle {{
            text-align: center;
            color: #9ca3af;
            font-size: 1.15rem;
            margin-bottom: 40px;
        }}
        .container {{
            max-width: 1200px;
            margin: 0 auto;
        }}
        .crop-card {{
            background: rgba(17, 24, 39, 0.75);
            backdrop-filter: blur(12px);
            border: 1px solid rgba(255, 255, 255, 0.08);
            border-radius: 18px;
            box-shadow: 0 10px 30px rgba(0, 0, 0, 0.4);
            padding: 24px;
            margin-bottom: 28px;
            display: flex;
            align-items: center;
            justify-content: space-between;
            gap: 20px;
            transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
        }}
        .crop-card:hover {{
            transform: translateY(-5px);
            border-color: rgba(96, 165, 250, 0.5);
            box-shadow: 0 20px 35px rgba(96, 165, 250, 0.12), 0 10px 15px rgba(0, 0, 0, 0.3);
        }}
        .crop-info {{
            flex: 0.8;
            min-width: 200px;
        }}
        .crop-name {{
            font-weight: 600;
            font-size: 1.35rem;
            color: #f9fafb;
            margin-bottom: 6px;
        }}
        .crop-meta {{
            font-size: 0.95rem;
            color: #6b7280;
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
            font-size: 0.72rem;
            color: #9ca3af;
            margin-bottom: 8px;
            text-transform: uppercase;
            letter-spacing: 1.2px;
            font-weight: 600;
        }}
        .img-wrapper {{
            background: #0b0f19;
            padding: 12px;
            border-radius: 12px;
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
        <h1>U-Net Line Removal Evaluation</h1>
        <div class="subtitle">Side-by-side comparison on synthetic test pairs using the model trained with the text-preserving loss function.</div>
        {"\n".join(cards_html)}
    </div>
</body>
</html>
"""
    with open(output_html, "w", encoding="utf-8") as f:
        f.write(html_content)
    print(f"Generated synthetic comparison dashboard: {output_html}")

def main():
    model_path = r"d:\Internship\OCR_PDF\LineRemovalNet\best_model.pth"
    test_dir = r"d:\Internship\OCR_PDF\BoudningBoxCleaning\synthetic_dataset_v4_test"
    
    inputs_dir = os.path.join(test_dir, "inputs")
    targets_dir = os.path.join(test_dir, "targets")
    cleaned_dir = os.path.join(test_dir, "cleaned_unet")
    dashboard_path = os.path.join(test_dir, "synthetic_comparison.html")
    
    os.makedirs(cleaned_dir, exist_ok=True)
    
    # Initialize U-Net Model (Medium, base_channels=32)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device for U-Net evaluation: {device}")
    
    if not os.path.exists(model_path):
        print(f"Error: Model weights not found at {model_path}")
        return
        
    model = UNet(n_channels=1, n_classes=1, bilinear=False, base_channels=32).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    print("Loaded U-Net weights successfully!")
    
    input_paths = sorted(glob.glob(os.path.join(inputs_dir, "*.png")))
    info_list = []
    
    for path in input_paths:
        fname = os.path.basename(path)
        print(f"Processing {fname}...")
        
        # Run inference
        cleaned = clean_crop(model, device, path)
        if cleaned is not None:
            save_path = os.path.join(cleaned_dir, fname)
            cv2.imwrite(save_path, cleaned)
            info_list.append({'filename': fname})
            
    print(f"Processed {len(info_list)} crops.")
    generate_dashboard(inputs_dir, targets_dir, cleaned_dir, dashboard_path, info_list)

if __name__ == "__main__":
    main()
