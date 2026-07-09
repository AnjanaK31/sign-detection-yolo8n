import os
import glob
import cv2
import numpy as np
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim

def main():
    test_dir = r"d:\Internship\OCR_PDF\BoudningBoxCleaning\synthetic_dataset_v4_test"
    targets_dir = os.path.join(test_dir, "targets")
    cleaned_dir = os.path.join(test_dir, "cleaned_unet")
    
    cleaned_paths = sorted(glob.glob(os.path.join(cleaned_dir, "*.png")))
    
    if not cleaned_paths:
        print("No cleaned images found!")
        return
        
    all_psnr = []
    all_ssim = []
    all_mse = []
    
    print(f"{'Filename':<20} | {'PSNR (dB)':<10} | {'SSIM':<8} | {'MSE':<8}")
    print("-" * 55)
    
    for path in cleaned_paths:
        fname = os.path.basename(path)
        target_path = os.path.join(targets_dir, fname)
        
        if not os.path.exists(target_path):
            print(f"Target not found for {fname}")
            continue
            
        img_cleaned = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        img_target = cv2.imread(target_path, cv2.IMREAD_GRAYSCALE)
        
        # Calculate MSE
        mse_val = np.mean((img_cleaned.astype(np.float32) - img_target.astype(np.float32)) ** 2)
        
        # Calculate PSNR
        try:
            psnr_val = psnr(img_target, img_cleaned, data_range=255)
        except Exception:
            psnr_val = 0.0
            
        # Calculate SSIM
        try:
            ssim_val = ssim(img_target, img_cleaned, data_range=255)
        except Exception:
            ssim_val = 0.0
            
        all_psnr.append(psnr_val)
        all_ssim.append(ssim_val)
        all_mse.append(mse_val)
        
        print(f"{fname:<20} | {psnr_val:<10.2f} | {ssim_val:<8.4f} | {mse_val:<8.2f}")
        
    print("-" * 55)
    print(f"{'AVERAGES':<20} | {np.mean(all_psnr):<10.2f} | {np.mean(all_ssim):<8.4f} | {np.mean(all_mse):<8.2f}")

if __name__ == "__main__":
    main()
