import os
import cv2
import numpy as np

# Resolve image.png relative to the script directory
script_dir = os.path.dirname(os.path.abspath(__file__))
image_path = os.path.join(script_dir, "image.png")
img = cv2.imread(image_path)
if img is None:
    raise FileNotFoundError(f"Could not load image at {image_path}. Please check if the file exists.")

# grayscale
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
cv2.imshow('gray', gray)

# binary
# ret, thresh = cv2.threshold(gray, 250, 255, cv2.THRESH_BINARY_INV)
thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 35, 180)
cv2.imshow('threshold', thresh)

# dilation
kernel = np.ones((1, 1), np.uint8)
img_dilation = cv2.dilate(thresh, kernel, iterations=1)
cv2.imshow('dilated', img_dilation)

# find contours
# cv2.findCountours() function changed from OpenCV3 to OpenCV4: now it have only two parameters instead of 3
cv2MajorVersion = cv2.__version__.split(".")[0]
# check for contours on thresh
if int(cv2MajorVersion) >= 4:
    ctrs, hier = cv2.findContours(img_dilation.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
else:
    im2, ctrs, hier = cv2.findContours(img_dilation.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# sort contours
sorted_ctrs = sorted(ctrs, key=lambda ctr: cv2.boundingRect(ctr)[0])

for i, ctr in enumerate(sorted_ctrs):
    # Get bounding box
    x, y, w, h = cv2.boundingRect(ctr)

    # Getting ROI
    roi = img[y:y + h, x:x + w]

    # show ROI
    # cv2.imshow('segment no:'+str(i),roi)
    cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 1)

    # if you want to save the letters without green bounding box, comment the line above
    if w > 5:
        cv2.imwrite('{}.png'.format(i), roi)

cv2.imshow('marked areas', img)

cv2.waitKey(0)
