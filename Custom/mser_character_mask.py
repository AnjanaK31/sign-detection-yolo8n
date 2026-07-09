import os
import cv2
import numpy as np

def generate_character_masks():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    image_path = os.path.join(script_dir, "image.png")
    
    print(f"Loading image from: {image_path}")
    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(f"Could not load image at {image_path}. Please check if the file exists.")
        
    h_img, w_img = img.shape[:2]
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Detect MSER regions
    mser = cv2.MSER_create(5, 30, 15000)
    regions, bboxes = mser.detectRegions(gray)
    
    # Filter bboxes to remove noise/borders using oriented bounding boxes
    filtered_char_boxes = [] # will contain (rect, w_oriented, h_oriented)
    for r in regions:
        rect = cv2.minAreaRect(r)
        box_pts = cv2.boxPoints(rect)
        v1 = box_pts[1] - box_pts[0]
        v2 = box_pts[2] - box_pts[1]
        L1 = np.linalg.norm(v1)
        L2 = np.linalg.norm(v2)
        
        if abs(v1[0]) > abs(v1[1]):
            w_o, h_o = L1, L2
        else:
            w_o, h_o = L2, L1
            
        if 5 <= w_o <= 80 and 5 <= h_o <= 80:
            filtered_char_boxes.append((rect, w_o, h_o))
            
    print(f"Total filtered candidate regions: {len(filtered_char_boxes)}")
    
    thresholds = [1.2, 1.5, 1.8]
    
    for thresh in thresholds:
        # Create an all-black background
        mask = np.zeros((h_img, w_img), dtype=np.uint8)
        char_count = 0
        
        for rect, w_o, h_o in filtered_char_boxes:
            ar = w_o / h_o if h_o != 0 else 0
            
            if ar <= thresh:
                char_count += 1
                box_pts = cv2.boxPoints(rect)
                box_pts = np.intp(box_pts)
                # Fill the oriented bounding box with white (255)
                cv2.drawContours(mask, [box_pts], 0, 255, -1)
                
        output_name = f"character_mask_thresh_{thresh:.1f}.png"
        output_path = os.path.join(script_dir, output_name)
        cv2.imwrite(output_path, mask)
        print(f"Saved binary character mask ({thresh}): {output_path}")
        print(f"  -> Highlighted {char_count} characters as white pixels.")

if __name__ == "__main__":
    generate_character_masks()
