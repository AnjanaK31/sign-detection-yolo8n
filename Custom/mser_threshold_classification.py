import os
import cv2
import numpy as np

def classify_by_thresholds():
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
            
    print(f"Total filtered candidate regions to classify: {len(filtered_char_boxes)}")
    
    thresholds = [1.2, 1.5, 1.8]
    
    for thresh in thresholds:
        img_visual = img.copy()
        char_count = 0
        wide_count = 0
        
        for rect, w_o, h_o in filtered_char_boxes:
            ar = w_o / h_o if h_o != 0 else 0
            
            box_pts = cv2.boxPoints(rect)
            box_pts = np.intp(box_pts)
            
            if ar <= thresh:
                # Classify as Character (Green)
                char_count += 1
                cv2.drawContours(img_visual, [box_pts], 0, (0, 255, 0), 1)
            else:
                # Classify as Wide Element / Horizontal noise / Fraction bar (Red)
                wide_count += 1
                cv2.drawContours(img_visual, [box_pts], 0, (0, 0, 255), 1)
                
        # Draw a legend on the top-left corner
        legend_bg = np.zeros((70, 320, 3), dtype=np.uint8) + 255
        img_visual[10:80, 10:330] = cv2.addWeighted(img_visual[10:80, 10:330], 0.3, legend_bg, 0.7, 0)
        
        cv2.putText(img_visual, f"Threshold (w/h): {thresh}", (20, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
        cv2.putText(img_visual, f"Character (w/h <= {thresh}): {char_count} (Green)", (20, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 150, 0), 1, cv2.LINE_AA)
        cv2.putText(img_visual, f"Wide / Non-Char (w/h > {thresh}): {wide_count} (Red)", (20, 70),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 200), 1, cv2.LINE_AA)
        
        output_name = f"classification_thresh_{thresh:.1f}.png"
        output_path = os.path.join(script_dir, output_name)
        cv2.imwrite(output_path, img_visual)
        print(f"Saved threshold classification image ({thresh}): {output_path}")
        print(f"  -> Characters: {char_count} ({char_count/len(filtered_char_boxes)*100:.1f}%)")
        print(f"  -> Wide Elements: {wide_count} ({wide_count/len(filtered_char_boxes)*100:.1f}%)")

if __name__ == "__main__":
    classify_by_thresholds()
