import os
import sys
import json
import cv2
import numpy as np

# Reconfigure stdout for Windows unicode support
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

def get_rotated_crop(img, points):
    pts = np.array(points, dtype=np.float32)
    # Calculate distances to get width and height
    w = int(round(np.linalg.norm(pts[0] - pts[1])))
    h = int(round(np.linalg.norm(pts[1] - pts[2])))
    
    if w <= 0: w = 1
    if h <= 0: h = 1
    
    # Destination points for warp
    dst_pts = np.array([[0, 0], [w-1, 0], [w-1, h-1], [0, h-1]], dtype=np.float32)
    M = cv2.getPerspectiveTransform(pts, dst_pts)
    warped = cv2.warpPerspective(img, M, (w, h))
    return warped

def main():
    label_path = r"d:\Internship\OCR_PDF\sign-detection-yolo8n\toTest\CLEANINTEST\Label.txt"
    img_path = r"d:\Internship\OCR_PDF\sign-detection-yolo8n\toTest\CLEANINTEST\intra.png"
    output_crops_dir = r"d:\Internship\OCR_PDF\sign-detection-yolo8n\toTest\CLEANINTEST\intra_crops"
    output_marked_path = r"d:\Internship\OCR_PDF\sign-detection-yolo8n\toTest\CLEANINTEST\intra_with_boxes.png"
    output_marked_resized_path = r"d:\Internship\OCR_PDF\sign-detection-yolo8n\toTest\CLEANINTEST\intra_with_boxes_resized.png"
    
    os.makedirs(output_crops_dir, exist_ok=True)
    
    print(f"Loading image from: {img_path}")
    img = cv2.imread(img_path)
    if img is None:
        print(f"ERROR: Could not load image at {img_path}")
        return
        
    h_img, w_img = img.shape[:2]
    marked_img = img.copy()
    
    # Read Label.txt and parse first line (intra.png)
    if not os.path.exists(label_path):
        print(f"ERROR: Label file not found: {label_path}")
        return
        
    intra_boxes = None
    with open(label_path, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split("\t")
            if len(parts) >= 2 and "intra.png" in parts[0]:
                intra_boxes = json.loads(parts[1])
                break
                
    if intra_boxes is None:
        print("ERROR: Did not find entry for intra.png in Label.txt")
        return
        
    print(f"Found {len(intra_boxes)} bounding boxes for intra.png.")
    
    for idx, box in enumerate(intra_boxes):
        transcription = box.get("transcription", "")
        points = box.get("points", [])
        
        if len(points) != 4:
            print(f"Warning: Box {idx} does not have 4 points: {points}")
            continue
            
        # Draw box on marked image
        pts_np = np.array(points, dtype=np.int32)
        cv2.polylines(marked_img, [pts_np], isClosed=True, color=(0, 0, 255), thickness=3)
        
        # Crop and save rotated box
        crop = get_rotated_crop(img, points)
        crop_name = f"crop_{idx}_{transcription.replace('/', '_').replace(':', '_').replace('°', 'deg')}.jpg"
        crop_path = os.path.join(output_crops_dir, crop_name)
        cv2.imwrite(crop_path, crop)
        
        # Put transcription label text near box
        # Find top-left coordinates for text placement
        tx = int(points[0][0])
        ty = int(points[0][1]) - 8
        cv2.putText(marked_img, f"{idx}: {transcription}", (tx, ty), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2, cv2.LINE_AA)
                    
        print(f"  Cropped & drawn Box {idx}: text='{transcription}' -> {crop_name}")
        
    # Save the marked full resolution image
    print(f"Saving marked full image to: {output_marked_path}")
    cv2.imwrite(output_marked_path, marked_img)
    
    # Save resized version
    target_h = 1200
    target_w = int(w_img * (target_h / h_img))
    marked_resized = cv2.resize(marked_img, (target_w, target_h))
    print(f"Saving marked resized image (1200px height) to: {output_marked_resized_path}")
    cv2.imwrite(output_marked_resized_path, marked_resized)
    print("Completed successfully!")

if __name__ == "__main__":
    main()
