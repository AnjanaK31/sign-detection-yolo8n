import os
import sys
import cv2
import numpy as np
from PIL import Image

sys.path.insert(0, "d:/Internship/OCR_PDF/internt-ocrmodel")

from preprocessor import to_grayscale, apply_threshold

def run_islands_on_test_folder():
    test_dir = "d:/Internship/OCR_PDF/sign-detection-yolo8n/toTest"
    output_dir = os.path.join(test_dir, "predictions")
    os.makedirs(output_dir, exist_ok=True)
    
    # Supported image extensions
    valid_exts = (".png", ".jpg", ".jpeg", ".bmp", ".tiff")
    images = [f for f in os.listdir(test_dir) if f.lower().endswith(valid_exts)]
    
    if not images:
        print(f"No images found in {test_dir}")
        return
        
    print(f"Found {len(images)} images to process in {test_dir}.")
    
    # Font threshold for large lines
    font_threshold = 20
    
    for img_name in images:
        img_path = os.path.join(test_dir, img_name)
        print(f"Processing image: {img_name}...")
        
        try:
            page_img = Image.open(img_path)
            
            # 1. Convert to grayscale and apply adaptive threshold
            gray = to_grayscale(page_img)
            thresh = apply_threshold(gray)
            
            # Invert to make foreground ink 255 and background 0
            binary_foreground = cv2.bitwise_not(thresh)
            
            # 2. Run connected components
            num_labels, labels_im, stats, _ = cv2.connectedComponentsWithStats(binary_foreground)
            print(f"  -> Found {num_labels - 1} islands in {img_name}")
            
            # 3. Create color map
            np.random.seed(42)
            colors = np.zeros((num_labels, 3), dtype=np.uint8)
            colors[0] = [255, 255, 255] # Background is white
            
            red_count = 0
            small_count = 0
            
            for i in range(1, num_labels):
                w = stats[i, cv2.CC_STAT_WIDTH]
                h = stats[i, cv2.CC_STAT_HEIGHT]
                
                # If width or height is larger than the threshold, color RED
                if h > font_threshold or w > font_threshold:
                    colors[i] = [255, 0, 0]
                    red_count += 1
                else:
                    # Random color (low red channel, high green/blue channels)
                    r = np.random.randint(0, 100)
                    g = np.random.randint(100, 255)
                    b = np.random.randint(100, 255)
                    colors[i] = [r, g, b]
                    small_count += 1
                    
            print(f"  -> Large components (lines/borders) colored RED: {red_count}")
            print(f"  -> Smaller components colored randomly: {small_count}")
            
            # 4. Color the image
            colored_islands_np = colors[labels_im]
            
            # 5. Save output
            base_name, _ = os.path.splitext(img_name)
            out_img_name = f"{base_name}_islands.png"
            out_path = os.path.join(output_dir, out_img_name)
            Image.fromarray(colored_islands_np).save(out_path)
            print(f"  -> Saved output to: {out_path}")
            
        except Exception as e:
            print(f"Error processing {img_name}: {e}")
            
    print("\nTest folder processing complete!")

if __name__ == "__main__":
    run_islands_on_test_folder()
