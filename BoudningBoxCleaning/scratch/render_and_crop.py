import os
import sys
import json
import fitz  # PyMuPDF
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
    pdf_path = r"d:\Internship\OCR_PDF\sign-detection-yolo8n\toTest\CLEANINTEST\Intra v30 spec.pdf"
    label_path = r"d:\Internship\OCR_PDF\sign-detection-yolo8n\toTest\CLEANINTEST\Label.txt"
    output_crops_dir = r"d:\Internship\OCR_PDF\sign-detection-yolo8n\toTest\CLEANINTEST\intra_pdf_crops"
    output_rendered_path = r"d:\Internship\OCR_PDF\sign-detection-yolo8n\toTest\CLEANINTEST\intra_pdf_rendered.png"
    output_marked_path = r"d:\Internship\OCR_PDF\sign-detection-yolo8n\toTest\CLEANINTEST\intra_pdf_with_boxes.png"
    output_marked_resized_path = r"d:\Internship\OCR_PDF\sign-detection-yolo8n\toTest\CLEANINTEST\intra_pdf_with_boxes_resized.png"
    
    os.makedirs(output_crops_dir, exist_ok=True)
    
    print(f"Opening PDF: {pdf_path}")
    if not os.path.exists(pdf_path):
        print(f"ERROR: PDF file not found: {pdf_path}")
        return
        
    doc = fitz.open(pdf_path)
    print(f"Number of pages: {len(doc)}")
    page = doc[0]
    rect = page.rect
    print(f"Original Page dimensions: {rect.width}x{rect.height} points")
    
    # Target resolution matching Label.txt coordinates (3300x2550)
    target_w = 3300
    target_h = 2550
    
    # Calculate scale factor
    scale_x = target_w / rect.width
    scale_y = target_h / rect.height
    matrix = fitz.Matrix(scale_x, scale_y)
    
    print(f"Rendering page 0 at {target_w}x{target_h} resolution...")
    pix = page.get_pixmap(matrix=matrix)
    
    # Convert PyMuPDF pixmap to numpy RGB image
    # pix.samples contains the raw pixel data
    img_data = np.frombuffer(pix.samples, dtype=np.uint8).reshape((pix.h, pix.w, pix.n))
    if pix.n == 4:
        img_bgr = cv2.cvtColor(img_data, cv2.COLOR_RGBA2BGR)
    else:
        img_bgr = cv2.cvtColor(img_data, cv2.COLOR_RGB2BGR)
        
    # Save the rendered high quality image
    print(f"Saving rendered image to: {output_rendered_path}")
    cv2.imwrite(output_rendered_path, img_bgr)
    
    marked_img = img_bgr.copy()
    
    # Read Label.txt and parse entry for intra.png
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
        
    print(f"Found {len(intra_boxes)} bounding boxes. Commencing crop extraction...")
    
    for idx, box in enumerate(intra_boxes):
        transcription = box.get("transcription", "")
        points = box.get("points", [])
        
        if len(points) != 4:
            print(f"Warning: Box {idx} does not have 4 points: {points}")
            continue
            
        # Draw box on marked image
        pts_np = np.array(points, dtype=np.int32)
        cv2.polylines(marked_img, [pts_np], isClosed=True, color=(0, 0, 255), thickness=3)
        
        # Crop rotated box from rendered image
        crop = get_rotated_crop(img_bgr, points)
        crop_name = f"crop_{idx}_{transcription.replace('/', '_').replace(':', '_').replace('°', 'deg')}.png"
        crop_path = os.path.join(output_crops_dir, crop_name)
        cv2.imwrite(crop_path, crop)
        
        # Put transcription label text near box
        tx = int(points[0][0])
        ty = int(points[0][1]) - 8
        cv2.putText(marked_img, f"{idx}: {transcription}", (tx, ty), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2, cv2.LINE_AA)
                    
        print(f"  Cropped Box {idx}: text='{transcription}' -> {crop_name}")
        
    # Save the marked full resolution image
    print(f"Saving marked image to: {output_marked_path}")
    cv2.imwrite(output_marked_path, marked_img)
    
    # Save resized version
    marked_resized = cv2.resize(marked_img, (int(target_w * (1200 / target_h)), 1200))
    print(f"Saving marked resized image (1200px height) to: {output_marked_resized_path}")
    cv2.imwrite(output_marked_resized_path, marked_resized)
    print("Completed PDF high quality cropping successfully!")

if __name__ == "__main__":
    main()
