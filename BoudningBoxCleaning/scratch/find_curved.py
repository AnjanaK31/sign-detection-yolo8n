import cv2
import numpy as np
import os
import glob

def find_curved():
    files = glob.glob("d:/Internship/OCR_PDF/BoudningBoxCleaning/synthetic_dataset_v3_test/inputs/*.png")
    for f in sorted(files):
        img = cv2.imread(f, cv2.IMREAD_GRAYSCALE)
        # Find if the image contains lines. We can do simple Hough line detection or check line properties.
        # Actually, let's just trace the coordinates of black pixels. A curved line will have multiple y-coordinates
        # for a single horizontal line structure. Let's do a simple check.
        H, W = img.shape
        binary = img < 127
        # Find the rows with black pixels
        row_sums = np.sum(binary, axis=1)
        # Let's count how many rows have black pixels.
        # Also, let's print the name of the file so we can view them.
        print(f"File: {os.path.basename(f)} | Size: {W}x{H}")

if __name__ == "__main__":
    find_curved()
