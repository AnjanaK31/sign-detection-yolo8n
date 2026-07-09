import os
import sys
import json
import cv2
import numpy as np

# Reconfigure stdout for Windows unicode support
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

def main():
    label_path = r"d:\Internship\OCR_PDF\INTRA_cleaning\Label.txt"
    img_path = r"d:\Internship\OCR_PDF\INTRA_cleaning\intra.png"
    output_path = r"d:\Internship\OCR_PDF\INTRA_cleaning\intra_marked.png"
    output_resized_path = r"d:\Internship\OCR_PDF\INTRA_cleaning\intra_marked_resized.png"
    
    print(f"Loading image from: {img_path}")
    img = cv2.imread(img_path)
    if img is None:
        print(f"ERROR: Could not load image at {img_path}")
        return
        
    h, w = img.shape[:2]
    marked_img = img.copy()
    
    if not os.path.exists(label_path):
        print(f"ERROR: Label file not found: {label_path}")
        return
        
    boxes = None
    with open(label_path, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split("\t")
            if len(parts) >= 2 and "intra.png" in parts[0]:
                boxes = json.loads(parts[1])
                break
                
    if boxes is None:
        print("ERROR: Did not find entry for intra.png in Label.txt")
        return
        
    print(f"Drawing {len(boxes)} bounding boxes on intra.png...")
    for idx, box in enumerate(boxes):
        transcription = box.get("transcription", "")
        points = box.get("points", [])
        
        if len(points) != 4:
            continue
            
        # Draw box
        pts_np = np.array(points, dtype=np.int32)
        cv2.polylines(marked_img, [pts_np], isClosed=True, color=(0, 0, 255), thickness=3)
        
        # Put text label
        tx = int(points[0][0])
        ty = int(points[0][1]) - 8
        cv2.putText(marked_img, f"{idx}: {transcription}", (tx, ty), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2, cv2.LINE_AA)
                    
    # Save the marked image
    print(f"Saving marked image to: {output_path}")
    cv2.imwrite(output_path, marked_img)
    
    # Save resized version
    target_h = 1200
    target_w = int(w * (target_h / h))
    marked_resized = cv2.resize(marked_img, (target_w, target_h))
    cv2.imwrite(output_resized_path, marked_resized)
    print("Success!")

if __name__ == "__main__":
    main()
