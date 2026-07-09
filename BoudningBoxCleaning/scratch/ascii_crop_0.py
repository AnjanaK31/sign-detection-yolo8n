import cv2

path = r"d:\Internship\OCR_PDF\sign-detection-yolo8n\Nived-dataset\dataset\dataset\crop_img_backup\intra_crop_0.jpg"
img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
if img is None:
    print("Failed to load image")
else:
    h, w = img.shape
    print(f"Dimensions: {w}x{h}")
    # Threshold it at 180 (black text on white background)
    for y in range(h):
        line = ""
        for x in range(w):
            val = img[y, x]
            if val < 180:
                line += "#"
            else:
                line += "."
        print(line)
