import cv2
import glob
import os
import pytesseract

# Reconfigure stdout for Windows unicode support
import sys
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

def search_tesseract():
    directories = [
        "d:/Internship/OCR_PDF/BoudningBoxCleaning/crops/*.png",
        "d:/Internship/OCR_PDF/BoudningBoxCleaning/toTest_outputs/crops/*.png"
    ]
    
    custom_config = r'--psm 6'
    
    for pattern in directories:
        files = glob.glob(pattern)
        print(f"\n--- Searching with Tesseract in: {pattern} ({len(files)} files) ---")
        for f in sorted(files):
            img = cv2.imread(f)
            if img is None:
                continue
            H, W, C = img.shape
            w_sub = (W - 8) // 3
            if w_sub <= 0:
                continue
            orig_crop = img[:, :w_sub]
            
            try:
                text = pytesseract.image_to_string(orig_crop, config=custom_config)
                lines = [line.strip() for line in text.split('\n') if line.strip()]
                text_str = " ".join(lines)
                if any(k in text_str for k in ["3.44", "83", "1.25", "3.4", "83 ±", "1.2", "44"]):
                    print(f"File: {f} | Lines: {lines}")
            except Exception as e:
                pass

if __name__ == "__main__":
    search_tesseract()
