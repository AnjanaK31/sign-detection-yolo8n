import os
import glob
import cv2
import numpy as np

crops_dir = r"d:\Internship\OCR_PDF\sign-detection-yolo8n\Nived-dataset\dataset\dataset\crop_img"
backup_dir = r"d:\Internship\OCR_PDF\sign-detection-yolo8n\Nived-dataset\dataset\dataset\crop_img_backup"

orig_crops = glob.glob(os.path.join(backup_dir, "*.jpg")) + glob.glob(os.path.join(backup_dir, "*.png"))
cleaned_crops = glob.glob(os.path.join(crops_dir, "*.jpg")) + glob.glob(os.path.join(crops_dir, "*.png"))

print(f"Crops in backup folder: {len(orig_crops)}")
print(f"Crops in active folder: {len(cleaned_crops)}")

if len(orig_crops) != len(cleaned_crops):
    print("WARNING: Count mismatch between active and backup folders!")
    
# Let's count how many files are actually binary-different
modified_count = 0
examples = []

for active_path in cleaned_crops:
    fname = os.path.basename(active_path)
    backup_path = os.path.join(backup_dir, fname)
    
    if os.path.exists(backup_path):
        img_active = cv2.imread(active_path, cv2.IMREAD_GRAYSCALE)
        img_backup = cv2.imread(backup_path, cv2.IMREAD_GRAYSCALE)
        
        if img_active is not None and img_backup is not None:
            # check if different
            diff = cv2.absdiff(img_active, img_backup)
            non_zero = cv2.countNonZero(diff)
            if non_zero > 0:
                modified_count += 1
                if len(examples) < 10:
                    examples.append((fname, non_zero))

print(f"Number of verified modified files: {modified_count}")
print("Sample modified files and number of pixels changed:")
for fname, diff_pixels in examples:
    print(f"  - {fname}: {diff_pixels} pixels changed")
