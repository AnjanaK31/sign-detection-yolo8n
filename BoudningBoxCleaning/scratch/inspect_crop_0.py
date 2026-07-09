import cv2
import easyocr
import numpy as np

def inspect():
    f = "d:/Internship/OCR_PDF/BoudningBoxCleaning/toTest_outputs/crops/crop_0_comparison.png"
    img = cv2.imread(f)
    if img is None:
        print("Image not found")
        return
    H, W, C = img.shape
    w_sub = (W - 8) // 3
    print(f"Image shape: {img.shape}, w_sub: {w_sub}")
    
    orig_crop = img[:, :w_sub]
    review_crop = img[:, w_sub+4:2*w_sub+4]
    cleaned_crop = img[:, 2*w_sub+8:]
    
    reader = easyocr.Reader(['en'])
    print("Orig text:", reader.readtext(orig_crop))
    print("Review text:", reader.readtext(review_crop))
    print("Cleaned text:", reader.readtext(cleaned_crop))

if __name__ == "__main__":
    inspect()
