import cv2
import math
import numpy as np
from ultralytics import YOLO

import sys
sys.path.append("d:/Internship/OCR_PDF/BoudningBoxCleaning")
from clean_page_expressions import rectify_crop, skeletonize, get_skeleton_junctions, split_skeleton_into_branches

def test():
    model = YOLO("D:/Internship/OCR_PDF/YOLO_expression_best.pt")
    img = cv2.imread("d:/Internship/OCR_PDF/sign-detection-yolo8n/toTest/2.jpeg")
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(img_gray, 180, 255, cv2.THRESH_BINARY)
    img_rgb = cv2.cvtColor(thresh, cv2.COLOR_GRAY2RGB)
    
    results = model(img_rgb, verbose=False, conf=0.25, imgsz=1280)
    cx, cy, w, h, r = results[0].obb.xywhr.cpu().numpy()[14]
    angle_deg = math.degrees(r)
    
    from PIL import Image
    crop_std = rectify_crop(Image.fromarray(img_rgb), {'cx': cx, 'cy': cy, 'w': w, 'h': h, 'angle': angle_deg}, 0.20)
    
    img_std = np.array(crop_std.convert("L"))
    _, thresh_std = cv2.threshold(img_std, 180, 255, cv2.THRESH_BINARY_INV)
    
    thickness_map = 2 * cv2.distanceTransform(thresh_std, cv2.DIST_L2, 5)
    skel = skeletonize(thresh_std)
    branches, _ = split_skeleton_into_branches(skel)
    
    for b_idx, branch_pts in enumerate(branches):
        thickness_val = np.mean([thickness_map[pt[1], pt[0]] for pt in branch_pts])
        print(f"Branch {b_idx}: length={len(branch_pts)}, x_range=[{np.min(branch_pts[:,0])}, {np.max(branch_pts[:,0])}], thickness_map mean={thickness_val:.2f}")

if __name__ == "__main__":
    test()
