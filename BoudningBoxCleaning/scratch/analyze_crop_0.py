import cv2
import numpy as np
import os

img_path = r"d:\Internship\OCR_PDF\sign-detection-yolo8n\Nived-dataset\dataset\dataset\crop_img_backup\intra_crop_0.jpg"
img = cv2.imread(img_path)
h, w = img.shape[:2]
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
_, thresh = cv2.threshold(gray, 180, 255, cv2.THRESH_BINARY_INV)

print(f"Crop Dimensions: {w}x{h}")
num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(thresh, connectivity=8)
print(f"Connected components: {num_labels}")
for label in range(1, num_labels):
    cx = stats[label, cv2.CC_STAT_LEFT]
    cy = stats[label, cv2.CC_STAT_TOP]
    cw = stats[label, cv2.CC_STAT_WIDTH]
    ch = stats[label, cv2.CC_STAT_HEIGHT]
    area = stats[label, cv2.CC_STAT_AREA]
    print(f"  Component {label}: bbox=[x={cx}, y={cy}, w={cw}, h={ch}], area={area}")
