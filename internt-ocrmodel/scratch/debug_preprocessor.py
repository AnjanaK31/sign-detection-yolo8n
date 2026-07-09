import cv2
import numpy as np
import os
import sys
sys.path.insert(0, "d:/Internship/OCR_PDF/internt-ocrmodel")

from data_gen import SyntheticDataGenerator
from preprocessor import full_preprocess

generator = SyntheticDataGenerator()
bg_dir = "temp_prep_bg"
generator.generate_backgrounds(bg_dir, count=1)
bg_path = os.path.join(bg_dir, "bg_0.png")
page_img, page_colored, labels, gt_details = generator.generate_full_page(bg_path, num_annotations=5)

# Save original page
page_img.save("debug_output/page_original.png")

# Run preprocessor
preprocessed_page = full_preprocess(page_img)
preprocessed_page.save("debug_output/page_preprocessed.png")

# Let's crop the first symbol from both
if gt_details:
    gt = gt_details[0]
    corners = np.array(gt["corners"], dtype=np.int32)
    x_min, y_min = np.min(corners, axis=0) - 20
    x_max, y_max = np.max(corners, axis=0) + 20
    
    img_orig_np = np.array(page_img)
    img_prep_np = np.array(preprocessed_page)
    img_gt_np = np.array(page_colored)
    
    h, w = img_orig_np.shape[:2]
    x_min = max(0, x_min)
    y_min = max(0, y_min)
    x_max = min(w, x_max)
    y_max = min(h, y_max)
    
    crop_orig = img_orig_np[y_min:y_max, x_min:x_max]
    crop_prep = img_prep_np[y_min:y_max, x_min:x_max]
    crop_gt = img_gt_np[y_min:y_max, x_min:x_max]
    
    cv2.imwrite("debug_output/crop_orig.png", cv2.cvtColor(crop_orig, cv2.COLOR_RGB2BGR))
    cv2.imwrite("debug_output/crop_prep.png", cv2.cvtColor(crop_prep, cv2.COLOR_RGB2BGR))
    cv2.imwrite("debug_output/crop_gt_col.png", cv2.cvtColor(crop_gt, cv2.COLOR_RGB2BGR))
    print("Saved crop debug images to debug_output/")

import shutil
shutil.rmtree(bg_dir)
