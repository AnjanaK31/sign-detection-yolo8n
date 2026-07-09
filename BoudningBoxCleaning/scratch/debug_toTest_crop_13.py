import cv2
import os

def split_panels():
    f = "d:/Internship/OCR_PDF/BoudningBoxCleaning/toTest_outputs/crops/crop_13_comparison.png"
    img = cv2.imread(f)
    if img is None:
        print("Image not found")
        return
    H, W, C = img.shape
    w_sub = (W - 8) // 3
    
    orig_crop = img[:, :w_sub]
    review_crop = img[:, w_sub+4:2*w_sub+4]
    cleaned_crop = img[:, 2*w_sub+8:]
    
    out_dir = "d:/Internship/OCR_PDF/BoudningBoxCleaning/scratch"
    os.makedirs(out_dir, exist_ok=True)
    
    cv2.imwrite(os.path.join(out_dir, "toTest_crop_13_orig.png"), orig_crop)
    cv2.imwrite(os.path.join(out_dir, "toTest_crop_13_review.png"), review_crop)
    cv2.imwrite(os.path.join(out_dir, "toTest_crop_13_cleaned.png"), cleaned_crop)
    print("Panels saved in scratch directory.")

if __name__ == "__main__":
    split_panels()
