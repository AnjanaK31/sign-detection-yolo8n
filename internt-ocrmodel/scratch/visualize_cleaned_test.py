import os
import sys
import cv2
import numpy as np
from PIL import Image

sys.path.insert(0, "d:/Internship/OCR_PDF/internt-ocrmodel")

from preprocessor import to_grayscale, apply_threshold
from pipeline import load_yolo_model


def run_cleaned_visualization():
    test_dir = "d:/Internship/OCR_PDF/sign-detection-yolo8n/toTest"
    output_dir = os.path.join(test_dir, "predictions")
    os.makedirs(output_dir, exist_ok=True)

    # Locate YOLO model
    yolo_path = "../YOLO_expression_best.pt"
    if not os.path.exists(yolo_path):
        yolo_path = "D:/Internship/OCR_PDF/YOLO_expression_best.pt"

    if not os.path.exists(yolo_path):
        print(f"ERROR: YOLO model not found at {yolo_path}")
        return

    print(f"Loading YOLO model from: {yolo_path}...")
    yolo_model = load_yolo_model(yolo_path)

    valid_exts = (".png", ".jpg", ".jpeg", ".bmp", ".tiff")
    images = [f for f in os.listdir(test_dir) if f.lower().endswith(valid_exts)]

    if not images:
        print(f"No images found in {test_dir}")
        return

    print(f"Found {len(images)} images to process.")
    font_threshold = 20

    for img_name in images:
        img_path = os.path.join(test_dir, img_name)
        print(f"\nProcessing {img_name}...")

        try:
            page_img = Image.open(img_path)
            gray = to_grayscale(page_img)
            thresh = apply_threshold(gray)
            binary_foreground = cv2.bitwise_not(thresh)

            # 1. Run YOLO to get OBB boxes
            img_bgr = cv2.cvtColor(np.array(page_img.convert("RGB")), cv2.COLOR_RGB2BGR)
            results = yolo_model(img_bgr, verbose=False, conf=0.25, imgsz=1280)
            result = results[0]

            # Create YOLO mask
            yolo_mask = np.zeros(binary_foreground.shape, dtype=np.uint8)
            if result.obb is not None and len(result.obb) > 0:
                xyxyxyxy = result.obb.xyxyxyxy.cpu().numpy()
                for corners in xyxyxyxy:
                    pts = np.array(corners, dtype=np.int32)
                    cv2.fillPoly(yolo_mask, [pts], 255)

            # 2. Run connected components
            num_labels, labels_im, stats, _ = cv2.connectedComponentsWithStats(binary_foreground)

            # 3. Compute big lines mask (CAD lines to remove)
            big_lines_mask = np.zeros(binary_foreground.shape, dtype=np.uint8)
            for i in range(1, num_labels):
                w = stats[i, cv2.CC_STAT_WIDTH]
                h = stats[i, cv2.CC_STAT_HEIGHT]
                if h > font_threshold or w > font_threshold:
                    big_lines_mask[labels_im == i] = 255

            # Red mask = big line pixels OUTSIDE YOLO boxes  (these get erased)
            red_mask = cv2.bitwise_and(big_lines_mask, cv2.bitwise_not(yolo_mask))

            # 4. Start from original image and erase red pixels → white
            out_canvas = np.array(page_img.convert("RGB"))
            out_canvas[red_mask > 0] = [255, 255, 255]   # Erase CAD lines

            # 5. Save output
            base_name, _ = os.path.splitext(img_name)
            out_img_name = f"{base_name}_cleaned.png"
            out_path = os.path.join(output_dir, out_img_name)
            Image.fromarray(out_canvas).save(out_path)
            print(f"  -> Saved output to: {out_path}")

            # Copy to scratch folder
            dest = f"d:/Internship/OCR_PDF/internt-ocrmodel/scratch/{out_img_name}"
            Image.fromarray(out_canvas).save(dest)
            print(f"  -> Copied to: {dest}")

        except Exception as e:
            import traceback
            print(f"Error processing {img_name}: {e}")
            traceback.print_exc()

    print("\nCleaned visualization complete!")


if __name__ == "__main__":
    run_cleaned_visualization()
