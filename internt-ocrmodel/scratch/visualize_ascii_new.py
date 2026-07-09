import cv2
import numpy as np

def print_ascii(img_path, label):
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        print(f"Could not load {img_path}")
        return
    
    # Threshold to binary
    _, thresh = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY_INV)
    
    h, w = thresh.shape
    new_w = 80
    new_h = int(h * (new_w / w) * 0.45)
    resized = cv2.resize(thresh, (new_w, new_h), interpolation=cv2.INTER_AREA)
    
    print("\n" + "="*80)
    print(f"ASCII Art for {label} ({w}x{h})")
    print("="*80)
    for y in range(new_h):
        line = ""
        for x in range(new_w):
            if resized[y, x] > 50:
                line += "#"
            else:
                line += " "
        print(line)
    print("="*80)

print_ascii("eval_output/crops/crop_4_original.png", "Original Crop 4")
print_ascii("eval_output/crops/crop_4_cleaned.png", "Cleaned Crop 4")
