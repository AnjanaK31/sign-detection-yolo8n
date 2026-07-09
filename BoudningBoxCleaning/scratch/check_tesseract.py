import pytesseract
import cv2

try:
    # Try running tesseract on a blank image
    import numpy as np
    img = np.zeros((100, 100), dtype=np.uint8) + 255
    res = pytesseract.image_to_string(img)
    print("Tesseract works! Output:", res)
except Exception as e:
    print("Tesseract error:", e)
