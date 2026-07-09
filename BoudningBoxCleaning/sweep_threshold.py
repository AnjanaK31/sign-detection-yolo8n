import os
import glob
import cv2
import numpy as np
import torch
import sys
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim

sys.path.append(os.path.abspath("../LineRemovalNet"))
from models.unet import UNet

def pad_to_multiple(img, multiple=16):
    h, w = img.shape[:2]
    pad_h = (multiple - h % multiple) % multiple
    pad_w = (multiple - w % multiple) % multiple
    padded = cv2.copyMakeBorder(img, 0, pad_h, 0, pad_w, cv2.BORDER_CONSTANT, value=255)
    return padded, pad_h, pad_w

def get_unet_output(model, device, img_path):
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
    return cropped_out

def main():
    model_path = r"d:\Internship\OCR_PDF\LineRemovalNet\best_model.pth"
    test_dir = r"d:\Internship\OCR_PDF\BoudningBoxCleaning\synthetic_dataset_v4_test"
    inputs_dir = os.path.join(test_dir, "inputs")
    targets_dir = os.path.join(test_dir, "targets")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = UNet(n_channels=1, n_classes=1, bilinear=False, base_channels=32).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    
    input_paths = sorted(glob.glob(os.path.join(inputs_dir, "*.png")))
    
    # Pre-calculate U-Net outputs and load targets
    pairs = []
    for path in input_paths:
        fname = os.path.basename(path)
        target_path = os.path.join(targets_dir, fname)
        if os.path.exists(target_path):
            unet_out = get_unet_output(model, device, path)
            img_target = cv2.imread(target_path, cv2.IMREAD_GRAYSCALE)
            pairs.append((unet_out, img_target))
            
    print(f"{'Threshold':<10} | {'Avg PSNR (dB)':<15} | {'Avg SSIM':<10} | {'Avg MSE':<10}")
    print("-" * 55)
    
    best_threshold = 180
    best_ssim = 0.0
    
    for thresh in range(50, 250, 10):
        psnrs = []
        ssims = []
        mses = []
        for unet_out, img_target in pairs:
            _, thresh_img = cv2.threshold(unet_out, thresh, 255, cv2.THRESH_BINARY)
            
            mse_val = np.mean((thresh_img.astype(np.float32) - img_target.astype(np.float32)) ** 2)
            try:
                psnr_val = psnr(img_target, thresh_img, data_range=255)
            except Exception:
                psnr_val = 0.0
            try:
                ssim_val = ssim(img_target, thresh_img, data_range=255)
            except Exception:
                ssim_val = 0.0
                
            psnrs.append(psnr_val)
            ssims.append(ssim_val)
            mses.append(mse_val)
            
        avg_psnr = np.mean(psnrs)
        avg_ssim = np.mean(ssims)
        avg_mse = np.mean(mses)
        
        if avg_ssim > best_ssim:
            best_ssim = avg_ssim
            best_threshold = thresh
            
        print(f"{thresh:<10} | {avg_psnr:<15.2f} | {avg_ssim:<10.4f} | {avg_mse:<10.2f}")
        
    print("-" * 55)
    print(f"Optimal Threshold: {best_threshold} (Avg SSIM: {best_ssim:.4f})")

if __name__ == "__main__":
    main()
