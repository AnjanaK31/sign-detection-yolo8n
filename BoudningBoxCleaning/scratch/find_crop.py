import cv2
import numpy as np
import os
import glob

def find_344():
    # We will search in both 'crops' and 'toTest_outputs/crops'
    paths = [
        "d:/Internship/OCR_PDF/BoudningBoxCleaning/crops/*.png",
        "d:/Internship/OCR_PDF/BoudningBoxCleaning/toTest_outputs/crops/*.png"
    ]
    
    # Let's try importing pytesseract or easyocr to perform text matching
    try:
        import pytesseract
        print("pytesseract is available")
    except ImportError:
        pytesseract = None
        
    try:
        import easyocr
        reader = easyocr.Reader(['en'])
        print("easyocr is available")
    except ImportError:
        easyocr = None

    for pattern in paths:
        files = glob.glob(pattern)
        print(f"Checking {len(files)} files in pattern {pattern}...")
        for f in files:
            img = cv2.imread(f)
            if img is None:
                continue
            # Each crop comparison is a grid: original | separator | review | separator | cleaned
            # The original crop is the left-most sub-image. Let's split it.
            H, W, C = img.shape
            # The separator has width 4, and there are 3 sub-images + 2 separators.
            # Grid width = 3 * W_sub + 8. So W_sub = (W - 8) // 3.
            w_sub = (W - 8) // 3
            if w_sub <= 0:
                continue
            orig_crop = img[:, :w_sub]
            
            # Use OCR to search for "3" or "3.44"
            if easyocr:
                results = reader.readtext(orig_crop)
                text = " ".join([r[1] for r in results])
                if "3.44" in text or "44" in text:
                    print(f"Match found with easyocr in {f}: '{text}'")
            elif pytesseract:
                try:
                    text = pytesseract.image_to_string(orig_crop)
                    if "3.44" in text or "44" in text or "83" in text:
                        print(f"Match found with pytesseract in {f}: '{text.strip()}'")
                except Exception as e:
                    pass

if __name__ == "__main__":
    find_344()
