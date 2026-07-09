import cv2
import math
import numpy as np
from ultralytics import YOLO
import sys

sys.path.append("d:/Internship/OCR_PDF/BoudningBoxCleaning")
from clean_page_expressions import rectify_crop, skeletonize, get_skeleton_junctions, split_skeleton_into_branches

def test_component_protection():
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
    crop_big = rectify_crop(Image.fromarray(img_rgb), {'cx': cx, 'cy': cy, 'w': w, 'h': h, 'angle': angle_deg}, 0.40)
    
    img_std = np.array(crop_std.convert("L"))
    img_big = np.array(crop_big.convert("L"))
    
    H_std, W_std = img_std.shape
    H_big, W_big = img_big.shape
    
    dx = (W_big - W_std) // 2
    dy = (H_big - H_std) // 2
    
    _, thresh_std = cv2.threshold(img_std, 180, 255, cv2.THRESH_BINARY_INV)
    _, thresh_big = cv2.threshold(img_big, 180, 255, cv2.THRESH_BINARY_INV)
    
    # We will compute the connected components on a slightly dilated version of thresh_big
    # (or just thresh_big) to find the labels.
    # Dilate thresh_big slightly (e.g. 3x3) to make sure minor gaps in lines are bridged
    dilated_big = cv2.dilate(thresh_big, cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3)))
    num_labels, labels_big, stats, centroids = cv2.connectedComponentsWithStats(dilated_big, connectivity=8)
    
    # For each component in labels_big, check if it extends to the outer buffer of big crop
    extends_outside = {}
    for label in range(1, num_labels):
        stat = stats[label]
        # stat is: [left, top, width, height, area]
        left = stat[cv2.CC_STAT_LEFT]
        top = stat[cv2.CC_STAT_TOP]
        width = stat[cv2.CC_STAT_WIDTH]
        height = stat[cv2.CC_STAT_HEIGHT]
        right = left + width
        bottom = top + height
        
        # Check if the component bounding box extends beyond the standard crop box
        ext = (left < dx or right > dx + W_std or top < dy or bottom > dy + H_std)
        extends_outside[label] = ext
        print(f"Big Component {label}: area={stat[cv2.CC_STAT_AREA]}, box=[x: {left}-{right}, y: {top}-{bottom}], extends_outside={ext}")
        
    skel = skeletonize(thresh_std)
    branches, _ = split_skeleton_into_branches(skel)
    
    border_margin = 3
    print("\n--- Testing Branch Protection ---")
    for b_idx, branch_pts in enumerate(branches):
        touches_border = np.any((branch_pts[:, 0] < border_margin) | 
                                (branch_pts[:, 0] > W_std - 1 - border_margin) | 
                                (branch_pts[:, 1] < border_margin) | 
                                (branch_pts[:, 1] > H_std - 1 - border_margin))
        
        if touches_border:
            # Find the border-touching points of the branch
            border_pts = branch_pts[(branch_pts[:, 0] < border_margin) | 
                                    (branch_pts[:, 0] > W_std - 1 - border_margin) | 
                                    (branch_pts[:, 1] < border_margin) | 
                                    (branch_pts[:, 1] > H_std - 1 - border_margin)]
            
            # Map them to big crop coordinates and check their labels
            labels_found = []
            for pt in border_pts:
                bx = pt[0] + dx
                by = pt[1] + dy
                if 0 <= bx < W_big and 0 <= by < H_big:
                    lbl = labels_big[by, bx]
                    if lbl > 0:
                        labels_found.append(lbl)
            
            # If any of the labels extends outside, then this branch is indeed part of a crossing line
            is_crossing_line = False
            if len(labels_found) > 0:
                is_crossing_line = any(extends_outside.get(lbl, False) for lbl in labels_found)
                
            print(f"Branch {b_idx}: length={len(branch_pts)}, x_range=[{np.min(branch_pts[:,0])}, {np.max(branch_pts[:,0])}], touches_border=True, is_crossing_line={is_crossing_line}")

if __name__ == "__main__":
    test_component_protection()
