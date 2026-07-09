import cv2
import numpy as np
import os

gt_path = "eval_output/eval_page_gt_colored.png"
if not os.path.exists(gt_path):
    print("GT page does not exist!")
    exit(1)

img = cv2.imread(gt_path)  # loads in BGR
h, w, c = img.shape
print(f"Loaded GT colored page: {w}x{h}")

# Unique colors
colors, counts = np.unique(img.reshape(-1, 3), axis=0, return_counts=True)
sorted_idx = np.argsort(-counts)
print("Top 10 colors in BGR:")
for idx in sorted_idx[:10]:
    print(f"Color (BGR): {colors[idx]} -> Count: {counts[idx]}")

# Let's count red pixels and blue pixels in BGR:
# Red in BGR: B < 100, G < 100, R > 200
# Blue in BGR: B > 200, G < 100, R < 100
blue_mask = (img[:, :, 0] > 200) & (img[:, :, 1] < 100) & (img[:, :, 2] < 100)
red_mask = (img[:, :, 0] < 100) & (img[:, :, 1] < 100) & (img[:, :, 2] > 200)

print(f"Total red pixels (text): {np.sum(red_mask)}")
print(f"Total blue pixels (CAD): {np.sum(blue_mask)}")
