import cv2
import glob
import numpy as np
import os

def find_matching_file():
    screenshot_path = "C:/Users/lalit/.gemini/antigravity-ide/brain/beaedbbc-8ffb-40cf-848f-d2b7b648f122/media__1782197394748.png"
    screen = cv2.imread(screenshot_path)
    if screen is None:
        print("Screenshot not found!")
        return
    
    # Preprocess screen (convert to gray)
    screen_gray = cv2.cvtColor(screen, cv2.COLOR_BGR2GRAY)
    sh_h, sh_w = screen_gray.shape
    print(f"Screenshot shape: {screen_gray.shape}")
    
    paths = [
        "d:/Internship/OCR_PDF/BoudningBoxCleaning/crops/*.png",
        "d:/Internship/OCR_PDF/BoudningBoxCleaning/toTest_outputs/crops/*.png"
    ]
    
    best_score = float('inf')
    best_file = None
    
    for pattern in paths:
        files = glob.glob(pattern)
        for f in files:
            img = cv2.imread(f)
            if img is None:
                continue
            img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            
            # Since the screenshot might be slightly resized or cropped, we can resize both to a standard size or template match
            # Actually, let's resize the image to the screenshot size and check MSE
            resized_img = cv2.resize(img_gray, (sh_w, sh_h))
            mse = np.mean((screen_gray - resized_img) ** 2)
            if mse < best_score:
                best_score = mse
                best_file = f
            
            # Print if it's very close
            if mse < 100:
                print(f"Very close match: {f} | MSE={mse:.2f}")
                
    print(f"Best matching file: {best_file} | MSE={best_score:.2f}")

if __name__ == "__main__":
    find_matching_file()
