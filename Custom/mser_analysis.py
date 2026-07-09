import os
import cv2
import numpy as np

def analyze_mser():
    # Resolve image.png relative to the script directory
    script_dir = os.path.dirname(os.path.abspath(__file__))
    image_path = os.path.join(script_dir, "image.png")
    
    print(f"Loading image from: {image_path}")
    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(f"Could not load image at {image_path}. Please check if the file exists.")
        
    h_img, w_img = img.shape[:2]
    print(f"Image dimensions: {w_img}x{h_img}")

    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # ----------------------------------------------------
    # 1. Detect individual character regions using MSER
    # ----------------------------------------------------
    # We can tune MSER parameters for better character detection:
    # _delta: comparing pixel differences (default 5)
    # _min_area: min area of region (default 60)
    # _max_area: max area of region (default 14400)
    mser = cv2.MSER_create(5, 30, 15000)
    regions, bboxes = mser.detectRegions(gray)
    
    print(f"MSER detected {len(bboxes)} raw regions.")
    
    # Filter bboxes to remove noise, borders, or large blocks using oriented bounding boxes
    # Normal characters usually have width and height between 5 and 80 pixels.
    filtered_indices = []
    filtered_char_boxes = [] # elements will be: (rect, w_oriented, h_oriented)
    
    for idx, r in enumerate(regions):
        rect = cv2.minAreaRect(r)
        box_pts = cv2.boxPoints(rect)
        # Compute oriented width and height relative to horizontal and vertical axis
        v1 = box_pts[1] - box_pts[0]
        v2 = box_pts[2] - box_pts[1]
        L1 = np.linalg.norm(v1)
        L2 = np.linalg.norm(v2)
        
        # Classify the horizontal-ish side as width and vertical-ish side as height
        if abs(v1[0]) > abs(v1[1]):
            w_o, h_o = L1, L2
        else:
            w_o, h_o = L2, L1
            
        if 5 <= w_o <= 80 and 5 <= h_o <= 80:
            filtered_indices.append(idx)
            filtered_char_boxes.append((rect, w_o, h_o))
        
    print(f"Filtered to {len(filtered_char_boxes)} character candidates.")
    
    # Calculate aspect ratios for individual characters
    char_ratios_w_h = []
    char_ratios_h_w = []
    
    img_chars = img.copy()
    for rect, w_o, h_o in filtered_char_boxes:
        ar_w_h = w_o / h_o if h_o != 0 else 0
        ar_h_w = h_o / w_o if w_o != 0 else 0
        char_ratios_w_h.append(ar_w_h)
        char_ratios_h_w.append(ar_h_w)
        
        # Draw oriented bounding box (Green)
        box_pts = cv2.boxPoints(rect)
        box_pts = np.intp(box_pts)
        cv2.drawContours(img_chars, [box_pts], 0, (0, 255, 0), 1)
        
    # Save characters visualization
    chars_output_path = os.path.join(script_dir, "mser_characters.png")
    cv2.imwrite(chars_output_path, img_chars)
    print(f"Saved character detection visualization to: {chars_output_path}")

    # ----------------------------------------------------
    # 2. Merge regions into "islands of characters" / lines
    # ----------------------------------------------------
    # Create a binary mask of detected character contours (more precise than bounding boxes)
    mask = np.zeros((h_img, w_img), dtype=np.uint8)
    for idx in filtered_indices:
        cv2.drawContours(mask, regions, idx, 255, -1)
        
    # Apply morphological closing/dilation to connect characters into words or lines
    # Since text runs horizontally, we use 1D horizontal kernels to prevent vertical merging.
    
    # Word level clustering:
    kernel_word = cv2.getStructuringElement(cv2.MORPH_RECT, (11, 1))
    mask_words = cv2.dilate(mask, kernel_word, iterations=1)
    mask_words = cv2.morphologyEx(mask_words, cv2.MORPH_CLOSE, kernel_word)
    
    # Line level clustering:
    kernel_line = cv2.getStructuringElement(cv2.MORPH_RECT, (35, 1))
    mask_lines = cv2.dilate(mask, kernel_line, iterations=1)
    mask_lines = cv2.morphologyEx(mask_lines, cv2.MORPH_CLOSE, kernel_line)
    
    # Helper to find contours and calculate aspect ratio
    def get_island_metrics(binary_mask, visual_img, box_color, label_prefix):
        contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        island_boxes = []
        ratios_w_h = []
        ratios_h_w = []
        
        for ctr in contours:
            rect = cv2.minAreaRect(ctr)
            box_pts = cv2.boxPoints(rect)
            v1 = box_pts[1] - box_pts[0]
            v2 = box_pts[2] - box_pts[1]
            L1 = np.linalg.norm(v1)
            L2 = np.linalg.norm(v2)
            
            if abs(v1[0]) > abs(v1[1]):
                w_o, h_o = L1, L2
            else:
                w_o, h_o = L2, L1
                
            if w_o < 5 or h_o < 5:
                continue
                
            island_boxes.append((rect, w_o, h_o))
            ar_w_h = w_o / h_o if h_o != 0 else 0
            ar_h_w = h_o / w_o if w_o != 0 else 0
            ratios_w_h.append(ar_w_h)
            ratios_h_w.append(ar_h_w)
            
            # Draw oriented bounding box
            box_pts = np.intp(box_pts)
            cv2.drawContours(visual_img, [box_pts], 0, box_color, 2)
            
            # Label with aspect ratio (w/h)
            label = f"{ar_w_h:.2f}"
            cx, cy = int(rect[0][0]), int(rect[0][1])
            cv2.putText(visual_img, label, (cx, cy), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, box_color, 1, cv2.LINE_AA)
                        
        return ratios_w_h, ratios_h_w, island_boxes

    img_words = img.copy()
    word_ratios_w_h, word_ratios_h_w, word_boxes = get_island_metrics(
        mask_words, img_words, (255, 0, 0), "word"
    )
    words_output_path = os.path.join(script_dir, "mser_word_islands.png")
    cv2.imwrite(words_output_path, img_words)
    print(f"Saved word islands visualization to: {words_output_path}")
    
    img_lines = img.copy()
    line_ratios_w_h, line_ratios_h_w, line_boxes = get_island_metrics(
        mask_lines, img_lines, (0, 0, 255), "line"
    )
    lines_output_path = os.path.join(script_dir, "mser_line_islands.png")
    cv2.imwrite(lines_output_path, img_lines)
    print(f"Saved line islands visualization to: {lines_output_path}")

    # ----------------------------------------------------
    # 3. Print and Save Statistics
    # ----------------------------------------------------
    def print_stats(name, w_h, h_w):
        if len(w_h) == 0:
            print(f"\n--- {name} --- \nNo elements detected.")
            return
        w_h = np.array(w_h)
        h_w = np.array(h_w)
        print(f"\n--- {name} (Count: {len(w_h)}) ---")
        print(f"Aspect Ratio (Width / Height):")
        print(f"  Min:    {w_h.min():.4f}")
        print(f"  Max:    {w_h.max():.4f}")
        print(f"  Mean:   {w_h.mean():.4f}")
        print(f"  Median: {np.median(w_h):.4f}")
        print(f"Aspect Ratio (Height / Width):")
        print(f"  Min:    {h_w.min():.4f}")
        print(f"  Max:    {h_w.max():.4f}")
        print(f"  Mean:   {h_w.mean():.4f}")
        print(f"  Median: {np.median(h_w):.4f}")

    print_stats("Individual Characters (MSER)", char_ratios_w_h, char_ratios_h_w)
    print_stats("Word-level Islands (Merged)", word_ratios_w_h, word_ratios_h_w)
    print_stats("Line-level Islands (Merged)", line_ratios_w_h, line_ratios_h_w)
    
    # Save raw results text file
    results_txt_path = os.path.join(script_dir, "mser_aspect_ratio_results.txt")
    with open(results_txt_path, "w") as f:
        f.write("=== MSER Aspect Ratio Analysis ===\n\n")
        for name, w_h, h_w in [
            ("Individual Characters (MSER)", char_ratios_w_h, char_ratios_h_w),
            ("Word-level Islands (Merged)", word_ratios_w_h, word_ratios_h_w),
            ("Line-level Islands (Merged)", line_ratios_w_h, line_ratios_h_w)
        ]:
            if len(w_h) == 0:
                f.write(f"--- {name} ---\nNo elements detected.\n\n")
                continue
            w_h = np.array(w_h)
            h_w = np.array(h_w)
            f.write(f"--- {name} (Count: {len(w_h)}) ---\n")
            f.write("Aspect Ratio (Width / Height):\n")
            f.write(f"  Min:    {w_h.min():.4f}\n")
            f.write(f"  Max:    {w_h.max():.4f}\n")
            f.write(f"  Mean:   {w_h.mean():.4f}\n")
            f.write(f"  Median: {np.median(w_h):.4f}\n")
            f.write("Aspect Ratio (Height / Width):\n")
            f.write(f"  Min:    {h_w.min():.4f}\n")
            f.write(f"  Max:    {h_w.max():.4f}\n")
            f.write(f"  Mean:   {h_w.mean():.4f}\n")
            f.write(f"  Median: {np.median(h_w):.4f}\n\n")
            
    print(f"\nStats report saved to: {results_txt_path}")

if __name__ == "__main__":
    analyze_mser()
