import cv2
import numpy as np

# Load original and cleaned Crop 5
crop_orig = cv2.imread("eval_output/crops/crop_5_original.png", cv2.IMREAD_GRAYSCALE)
crop_cleaned = cv2.imread("eval_output/crops/crop_5_cleaned.png", cv2.IMREAD_GRAYSCALE)

# Print ASCII of original Crop 5
print("="*60)
print("ORIGINAL CROP 5")
print("="*60)
h, w = crop_orig.shape
resized_orig = cv2.resize(cv2.threshold(crop_orig, 127, 255, cv2.THRESH_BINARY_INV)[1], (60, int(h * 60 / w * 0.45)), interpolation=cv2.INTER_AREA)
for row in resized_orig:
    print("".join(["#" if val > 50 else " " for val in row]))

# Print ASCII of cleaned Crop 5
print("\n" + "="*60)
print("CLEANED CROP 5 (CURRENT)")
print("="*60)
resized_cleaned = cv2.resize(cv2.threshold(crop_cleaned, 127, 255, cv2.THRESH_BINARY_INV)[1], (60, int(h * 60 / w * 0.45)), interpolation=cv2.INTER_AREA)
for row in resized_cleaned:
    print("".join(["#" if val > 50 else " " for val in row]))
