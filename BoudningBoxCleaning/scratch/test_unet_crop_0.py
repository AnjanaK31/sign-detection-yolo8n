import sys
import os
import cv2
import numpy as np
import torch
from PIL import Image

sys.path.append(os.path.abspath(r"d:\Internship\OCR_PDF\LineRemovalNet"))
from models.unet import UNet

def pad_to_multiple(img, multiple=16):
    h, w = img.shape[:2]
    pad_h = (multiple - h % multiple) % multiple
    pad_w = (multiple - w % multiple) % multiple
    padded = cv2.copyMakeBorder(img, 0, pad_h, 0, pad_w, cv2.BORDER_CONSTANT, value=255)
    return padded, pad_h, pad_w

def clean_crop_unet(model, device, crop_path, threshold_val=180):
    img_gray = cv2.imread(crop_path, cv2.IMREAD_GRAYSCALE)
    if img_gray is None:
        print("Failed to load image")
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

# Load U-Net model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model_path = r"d:\Internship\OCR_PDF\LineRemovalNet\best_model.pth"
model = UNet(n_channels=1, n_classes=1, bilinear=False, base_channels=32).to(device)
model.load_state_dict(torch.load(model_path, map_location=device))
model.eval()

img_path = r"d:\Internship\OCR_PDF\sign-detection-yolo8n\Nived-dataset\dataset\dataset\crop_img_backup\intra_crop_0.jpg"
cleaned = clean_crop_unet(model, device, img_path)

if cleaned is not None:
    print("U-Net Cleaned Output:")
    h, w = cleaned.shape
    for y in range(h):
        line = ""
        for x in range(w):
            if cleaned[y, x] < 180:
                line += "#"
            else:
                line += "."
        print(line)
else:
    print("Failed to clean")
