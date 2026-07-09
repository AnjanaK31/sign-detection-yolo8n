import os
import sys
import cv2
import numpy as np
from PIL import Image

sys.path.insert(0, "d:/Internship/OCR_PDF/internt-ocrmodel")

from preprocessor import to_grayscale, apply_threshold

def generate_islands_visualization():
    print("Loading page image...")
    eval_page_path = "eval_output/eval_page.png"
    
    if not os.path.exists(eval_page_path):
        print("ERROR: Evaluation page does not exist. Please run run_page_eval.py first.")
        return
        
    page_img = Image.open(eval_page_path)
    
    # 1. Convert to grayscale and apply adaptive threshold
    print("Preprocessing to binary...")
    gray = to_grayscale(page_img)
    thresh = apply_threshold(gray)
    
    # Invert so foreground (ink/lines/characters) is 255 (white) and background is 0 (black)
    binary_foreground = cv2.bitwise_not(thresh)
    
    # 2. Run connected components analysis
    print("Running connected components (islands detection)...")
    num_labels, labels_im, stats, centroids = cv2.connectedComponentsWithStats(binary_foreground)
    print(f"Found {num_labels - 1} islands of pixels.")
    
    # 3. Create a color lookup table
    # Seed random for consistent colors
    np.random.seed(42)
    
    # Table of colors (RGB)
    colors = np.zeros((num_labels, 3), dtype=np.uint8)
    # Background label (0) remains white
    colors[0] = [255, 255, 255]
    
    # Average font size is ~28, character height is normally under ~35-40 pixels
    font_threshold = 35
    
    red_count = 0
    small_count = 0
    noise_count = 0
    
    for i in range(1, num_labels):
        w = stats[i, cv2.CC_STAT_WIDTH]
        h = stats[i, cv2.CC_STAT_HEIGHT]
        area = stats[i, cv2.CC_STAT_AREA]
        
        # If the component's height or width is larger than the threshold (average font size)
        if h > font_threshold or w > font_threshold:
            # Color it RED
            colors[i] = [255, 0, 0]
            red_count += 1
        elif w < 3 and h < 3:
            # Color noise in light gray
            colors[i] = [220, 220, 220]
            noise_count += 1
        else:
            # Color normal symbol/text characters with a distinct random color (excluding red)
            r = np.random.randint(0, 100)
            g = np.random.randint(100, 255)
            b = np.random.randint(100, 255)
            colors[i] = [r, g, b]
            small_count += 1
            
    print(f"Large islands (lines/borders/CAD) colored RED: {red_count}")
    print(f"Character-sized islands colored randomly: {small_count}")
    print(f"Noise components colored gray: {noise_count}")
    
    # 4. Generate colored image by mapping labels image through color lookup table
    print("Coloring islands...")
    colored_islands_np = colors[labels_im]
    
    # 5. Save output images
    output_dir = "eval_output"
    os.makedirs(output_dir, exist_ok=True)
    
    output_path = os.path.join(output_dir, "eval_page_islands.png")
    Image.fromarray(colored_islands_np).save(output_path)
    print(f"Saved islands visualization to: {output_path}")
    
    # Copy to target folders for user convenience
    destinations = [
        "d:/Internship/OCR_PDF/eval_page_islands.png",
        "d:/Internship/OCR_PDF/internt-ocrmodel/eval_page_islands.png",
        "d:/Internship/OCR_PDF/internt-ocrmodel/scratch/eval_page_islands.png"
    ]
    
    for dest in destinations:
        try:
            Image.fromarray(colored_islands_np).save(dest)
            print(f"Copied to: {dest}")
        except Exception as e:
            print(f"Failed to copy to {dest}: {e}")
            
    print("Islands visualization complete!")

if __name__ == "__main__":
    generate_islands_visualization()
