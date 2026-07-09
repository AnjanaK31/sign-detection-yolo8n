import cv2
import glob
import os

def find_exact_crop():
    directories = [
        "d:/Internship/OCR_PDF/BoudningBoxCleaning/crops/*.png",
        "d:/Internship/OCR_PDF/BoudningBoxCleaning/toTest_outputs/crops/*.png"
    ]
    
    import easyocr
    reader = easyocr.Reader(['en'])
    
    for pattern in directories:
        files = glob.glob(pattern)
        print(f"Searching pattern: {pattern} ({len(files)} files)")
        for f in files:
            img = cv2.imread(f)
            if img is None:
                continue
            H, W, C = img.shape
            w_sub = (W - 8) // 3
            if w_sub <= 0:
                continue
            orig_crop = img[:, :w_sub]
            
            results = reader.readtext(orig_crop)
            texts = [r[1] for r in results]
            text_str = " ".join(texts)
            
            if any(x in text_str for x in ["3.44", "83", "1.25", "1.2", "44"]):
                print(f"File: {f} | Extracted: {texts}")

if __name__ == "__main__":
    find_exact_crop()
