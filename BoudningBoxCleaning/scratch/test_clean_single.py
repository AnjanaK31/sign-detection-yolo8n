import sys
import os
import cv2
import numpy as np
from PIL import Image

sys.path.append(os.path.abspath(r"d:\Internship\OCR_PDF\BoudningBoxCleaning"))
from clean_page_expressions import clean_crop_lines

img_path = r"d:\Internship\OCR_PDF\sign-detection-yolo8n\Nived-dataset\dataset\dataset\crop_img\intra_crop_0.jpg"
img = Image.open(img_path)

# Let's try calling clean_crop_lines with standard and big crops
# Since we only have the standard crop img, let's pass it as both standard and big crop
try:
    cleaned, erased_pts, review = clean_crop_lines(img, img)
    print("Execution completed successfully!")
    print(f"Erased pixels: {len(erased_pts)}")
except Exception as e:
    print(f"Failed: {e}")
