import cv2
import glob
import os
import easyocr

def inspect_all():
    directories = [
        "d:/Internship/OCR_PDF/BoudningBoxCleaning/crops/*.png",
        "d:/Internship/OCR_PDF/BoudningBoxCleaning/toTest_outputs/crops/*.png"
    ]
    
    reader = easyocr.Reader(['en'])
    
    for pattern in directories:
        files = glob.glob(pattern)
        print(f"\n--- Directory: {pattern} ---")
        for f in sorted(files, key=lambda x: int(os.path.basename(x).split('_')[1])):
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
            print(f"{os.path.basename(f)}: {texts}")

if __name__ == "__main__":
    inspect_all()
