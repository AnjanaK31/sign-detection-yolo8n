import os
import argparse
import cv2
import numpy as np
import math
from PIL import Image
from pdf2image import convert_from_path

# Import existing pipeline components
from pipeline import load_yolo_model, preprocess_image
from rectifier import rectify_crop

def skeletonize(img):
    """Applies morphological skeletonization to thinned 1-pixel width."""
    size = np.size(img)
    skel = np.zeros(img.shape, np.uint8)
    element = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))
    done = False
    temp_img = img.copy()
    
    while not done:
        eroded = cv2.erode(temp_img, element)
        temp = cv2.dilate(eroded, element)
        temp = cv2.subtract(temp_img, temp)
        skel = cv2.bitwise_or(skel, temp)
        temp_img = eroded.copy()
        
        zeros = size - cv2.countNonZero(temp_img)
        if zeros == size:
            done = True
            
    return skel

def get_junctions(skel):
    """Finds junction points where lines cross/meet in the skeleton.
    Junctions are thinned pixels with > 2 neighbors in their 8-neighborhood.
    """
    kernel = np.array([[1, 1, 1],
                       [1, 0, 1],
                       [1, 1, 1]], dtype=np.uint8)
    
    binary_skel = (skel > 0).astype(np.uint8)
    neighbor_count = cv2.filter2D(binary_skel, -1, kernel)
    junctions = (binary_skel == 1) & (neighbor_count > 2)
    return junctions.astype(np.uint8) * 255

def clean_cad_lines_boundary_tracing(crop_pil_std, crop_pil_big):
    """Traces connected components crossing from the outer buffer region into the inner standard crop.
    Wipes out noise speckles and erases crossing straight line segments, protecting character junctions.
    """
    img_std = np.array(crop_pil_std.convert("L"))
    img_big = np.array(crop_pil_big.convert("L"))
    
    H_std, W_std = img_std.shape
    H_big, W_big = img_big.shape
    
    # Calculate top-left offset of standard crop within the big crop
    dx = (W_big - W_std) // 2
    dy = (H_big - H_std) // 2
    
    # Binarize (assuming white background, black drawings)
    _, thresh_std = cv2.threshold(img_std, 127, 255, cv2.THRESH_BINARY_INV)
    _, thresh_big = cv2.threshold(img_big, 127, 255, cv2.THRESH_BINARY_INV)
    
    # --- NOISE BARRIER ---
    # 1. Morphological Opening (2x2 kernel) to erase thin speckles/noise lines
    morph_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
    thresh_std = cv2.morphologyEx(thresh_std, cv2.MORPH_OPEN, morph_kernel)
    thresh_big = cv2.morphologyEx(thresh_big, cv2.MORPH_OPEN, morph_kernel)
    
    # 2. Connected Component Area Filtering (area < 15 pixels) to erase larger speckles
    num_labels_std, labels_std, stats_std, _ = cv2.connectedComponentsWithStats(thresh_std, connectivity=8)
    for label in range(1, num_labels_std):
        if stats_std[label, cv2.CC_STAT_AREA] < 15:
            thresh_std[labels_std == label] = 0
            
    num_labels_big_denoise, labels_big_denoise, stats_big_denoise, _ = cv2.connectedComponentsWithStats(thresh_big, connectivity=8)
    for label in range(1, num_labels_big_denoise):
        if stats_big_denoise[label, cv2.CC_STAT_AREA] < 15:
            thresh_big[labels_big_denoise == label] = 0
    # ---------------------
            
    # Connected component analysis on the denoised big crop
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(thresh_big, connectivity=8)
    
    # Define boundary (border) mask and inner mask inside the big crop
    boundary_mask = np.ones((H_big, W_big), dtype=np.uint8) * 255
    boundary_mask[dy:dy+H_std, dx:dx+W_std] = 0
    
    inner_mask = np.zeros((H_big, W_big), dtype=np.uint8)
    inner_mask[dy:dy+H_std, dx:dx+W_std] = 255
    
    # Mask for components that cross the boundary (inner standard crop space)
    erase_mask_inner = np.zeros((H_std, W_std), dtype=np.uint8)
    
    # Mask of crossing components inside the big crop
    crossing_mask_big = np.zeros((H_big, W_big), dtype=np.uint8)
    
    for label in range(1, num_labels):
        comp_mask = (labels == label)
        
        has_boundary = np.any(comp_mask & (boundary_mask == 255))
        has_inner = np.any(comp_mask & (inner_mask == 255))
        
        if has_boundary and has_inner:
            crossing_mask_big[comp_mask] = 255
            comp_inner = comp_mask[dy:dy+H_std, dx:dx+W_std]
            erase_mask_inner[comp_inner] = 255
            
    # Detect straight lines inside the standard crop (min length of 20 pixels to filter out short strokes)
    lines = cv2.HoughLinesP(thresh_std, 1, np.pi/180, threshold=20, minLineLength=20, maxLineGap=4)
    line_pixels = np.zeros_like(thresh_std)
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            # Draw line with thickness 2 to cover the stroke width
            cv2.line(line_pixels, (x1, y1), (x2, y2), 255, thickness=2)
            
    # Erase candidate pixels must belong to a crossing component AND a detected straight line
    erase_mask_lines = cv2.bitwise_and(erase_mask_inner, line_pixels)
    
    # Protect character intersections using skeleton junction detection
    skel = skeletonize(thresh_std)
    junctions = get_junctions(skel)
    protection_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    protected_zone = cv2.dilate(junctions, protection_kernel)
    
    # Do not erase protected zones
    erase_mask_clean = cv2.bitwise_and(erase_mask_lines, cv2.bitwise_not(protected_zone))
    
    # Clean the standard threshold
    cleaned_thresh_std = cv2.subtract(thresh_std, erase_mask_clean)
    cleaned_std = cv2.bitwise_not(cleaned_thresh_std)
    
    # Highlight lines in Red on both standard and big crops
    highlight_img_big = cv2.cvtColor(img_big, cv2.COLOR_GRAY2BGR)
    highlight_img_std = cv2.cvtColor(img_std, cv2.COLOR_GRAY2BGR)
    
    # Detect straight lines in the big crop (including standard crop area + outer boundary)
    line_pixels_big = np.zeros_like(thresh_big)
    lines_big = cv2.HoughLinesP(thresh_big, 1, np.pi/180, threshold=20, minLineLength=20, maxLineGap=4)
    if lines_big is not None:
        for line in lines_big:
            x1, y1, x2, y2 = line[0]
            cv2.line(line_pixels_big, (x1, y1), (x2, y2), 255, thickness=2)
            
    # Draw standard crop lines mapped onto the big crop to ensure continuity
    line_pixels_big[dy:dy+H_std, dx:dx+W_std] = cv2.bitwise_or(line_pixels_big[dy:dy+H_std, dx:dx+W_std], line_pixels)
            
    # Intersect crossing components with detected lines in big crop
    erase_mask_lines_big = cv2.bitwise_and(crossing_mask_big, line_pixels_big)
    highlight_img_big[erase_mask_lines_big == 255] = [0, 0, 255] # Red BGR
    
    # Highlight standard crop
    highlight_img_std[erase_mask_lines == 255] = [0, 0, 255] # Red BGR
    
    return cleaned_std, highlight_img_big, highlight_img_std

def run_test_pipeline(pdf_path, yolo_path, output_dir):
    print(f"Loading YOLOv8-OBB model from {yolo_path}...")
    yolo = load_yolo_model(yolo_path)
    
    print(f"Reading PDF: {pdf_path} (converting Page 1)...")
    pages = convert_from_path(pdf_path, dpi=200)
    if not pages:
        print("Error: Could not extract pages from PDF.")
        return
        
    page_img = pages[0]
    print("Preprocessing Page 1...")
    preprocessed_img = preprocess_image(page_img)
    
    img_np = np.array(preprocessed_img)
    img_bgr = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
    
    print("Running YOLO symbol detection...")
    results = yolo(img_bgr, verbose=False, conf=0.25, imgsz=1280)
    result = results[0]
    
    os.makedirs(output_dir, exist_ok=True)
    print(f"Saving comparison crops to: {output_dir}")
    
    if result.obb is not None and len(result.obb) > 0:
        xywhr = result.obb.xywhr.cpu().numpy()
        xyxyxyxy = result.obb.xyxyxyxy.cpu().numpy()
        
        print(f"Found {len(xywhr)} bounding boxes. Processing crops...")
        
        for idx in range(len(xywhr)):
            cx, cy, w, h, r = xywhr[idx]
            corners = xyxyxyxy[idx]
            angle_deg = math.degrees(r)
            
            # Extract standard crop (10% buffer)
            crop_pil_std = rectify_crop(
                preprocessed_img,
                bbox_metrics={'cx': cx, 'cy': cy, 'w': w, 'h': h, 'angle': angle_deg},
                buffer_percent=0.10
            )
            
            # Extract bigger crop (40% buffer)
            crop_pil_big = rectify_crop(
                preprocessed_img,
                bbox_metrics={'cx': cx, 'cy': cy, 'w': w, 'h': h, 'angle': angle_deg},
                buffer_percent=0.40
            )
            
            # Run boundary-crossing tracing and line cleaning
            cleaned_crop, highlight_big, highlight_std = clean_cad_lines_boundary_tracing(crop_pil_std, crop_pil_big)
            
            # Format outputs for visual grid:
            # 1. Standard Crop (BGR)
            orig_std_bgr = cv2.cvtColor(np.array(crop_pil_std.convert("L")), cv2.COLOR_GRAY2BGR)
            
            # 2. Big Crop with highlighted lines scaled to standard height
            H_std, W_std = orig_std_bgr.shape[:2]
            H_big, W_big = highlight_big.shape[:2]
            scale = H_std / H_big
            W_big_scaled = int(W_big * scale)
            highlight_big_scaled = cv2.resize(highlight_big, (W_big_scaled, H_std), interpolation=cv2.INTER_CUBIC)
            
            # 3. Cleaned standard crop (BGR)
            cleaned_bgr = cv2.cvtColor(cleaned_crop, cv2.COLOR_GRAY2BGR)
            
            # Add thin green separator lines between sections
            separator = np.ones((H_std, 3, 3), dtype=np.uint8) * 128
            separator[:, :, 1] = 200 # green tint
            
            # Stack horizontally: Standard original | Outer with highlighted crossings | Cleaned standard
            comparison = np.hstack((orig_std_bgr, separator, highlight_big_scaled, separator, cleaned_bgr))
            
            save_path = os.path.join(output_dir, f"crop_{idx}_comparison.png")
            cv2.imwrite(save_path, comparison)
            print(f"Saved: {save_path}")
            
        print(f"Finished! Total crops processed: {len(xywhr)}")
    else:
        print("No bounding boxes detected by YOLO on this page.")

if __name__ == "__main__":
    # Resolve default paths
    default_yolo = "runs/obb/runs/obb/trained_on_1000_pdfs-2/weights/best.pt"
    if not os.path.exists(default_yolo):
        default_yolo = "runs/obb/yolo_obb_project/symbol_obb_train-2/weights/best.pt"
    if not os.path.exists(default_yolo):
        default_yolo = "runs/obb/yolo_obb_project/symbol_obb_train/weights/best.pt"
        
    pdfs_dir = "pdfs"
    default_pdf = None
    if os.path.exists(pdfs_dir):
        pdf_files = [f for f in os.listdir(pdfs_dir) if f.lower().endswith(".pdf")]
        if pdf_files:
            default_pdf = os.path.join(pdfs_dir, pdf_files[0])
            
    parser = argparse.ArgumentParser(description="Test CAD Line Cleaning on YOLO Crops")
    parser.add_argument("--pdf", default=default_pdf, help="Path to input PDF file")
    parser.add_argument("--yolo", default=default_yolo, help="Path to YOLOv8-OBB model (.pt)")
    parser.add_argument("--output", default="debug_crops_cleaned_pipeline", help="Output directory for cleaned crops")
    args = parser.parse_args()
    
    if not args.pdf or not os.path.exists(args.pdf):
        print(f"Error: PDF file '{args.pdf}' not found. Place a PDF in the 'pdfs/' directory.")
    elif not os.path.exists(args.yolo):
        print(f"Error: YOLO model weights '{args.yolo}' not found.")
    else:
        run_test_pipeline(args.pdf, args.yolo, args.output)
