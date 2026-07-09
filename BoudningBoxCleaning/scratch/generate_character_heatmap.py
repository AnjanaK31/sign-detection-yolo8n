import os
import cv2
import numpy as np

def main():
    # Define paths
    gt_path = r"d:\Internship\OCR_PDF\internt-ocrmodel\eval_output\eval_page_gt_colored.png"
    orig_path = r"d:\Internship\OCR_PDF\internt-ocrmodel\eval_output\eval_page.png"
    out_mask_path = r"d:\Internship\OCR_PDF\BoudningBoxCleaning\scratch\character_mask.png"
    out_heatmap_path = r"d:\Internship\OCR_PDF\BoudningBoxCleaning\scratch\character_heatmap.png"
    out_overlay_path = r"d:\Internship\OCR_PDF\BoudningBoxCleaning\scratch\character_heatmap_overlay.png"

    if not os.path.exists(gt_path):
        print(f"Error: Ground truth file not found at {gt_path}")
        return

    # Load images
    gt_img = cv2.imread(gt_path)  # BGR
    orig_img = cv2.imread(orig_path) if os.path.exists(orig_path) else None

    print("Loaded images successfully.")

    # 1. Identify characters from the Ground Truth colored page
    # In BGR: Red has high R (index 2), low G (index 1), low B (index 0)
    # R > 200, G < 100, B < 100
    b = gt_img[:, :, 0]
    g = gt_img[:, :, 1]
    r = gt_img[:, :, 2]

    # Create binary mask of character pixels
    char_mask = ((r > 200) & (g < 100) & (b < 100)).astype(np.uint8) * 255

    # Save the raw character mask
    cv2.imwrite(out_mask_path, char_mask)
    print(f"Saved binary character mask to: {out_mask_path}")

    # 2. Generate a heatmap of character pixels
    # We can do this using a Gaussian blur to create a soft probability-like density map
    # A large kernel size (e.g. 51x51) makes a smooth, continuous heatmap
    heatmap_float = cv2.GaussianBlur(char_mask.astype(np.float32), (51, 51), 0)
    
    # Normalize the density map to [0, 255]
    max_val = np.max(heatmap_float)
    if max_val > 0:
        heatmap_normalized = (heatmap_float / max_val * 255.0).astype(np.uint8)
    else:
        heatmap_normalized = heatmap_float.astype(np.uint8)

    # Apply a color map (e.g. COLORMAP_JET)
    heatmap_color = cv2.applyColorMap(heatmap_normalized, cv2.COLORMAP_JET)
    cv2.imwrite(out_heatmap_path, heatmap_color)
    print(f"Saved clean heatmap image to: {out_heatmap_path}")

    # 3. Create a semi-transparent overlay on the original page
    if orig_img is not None:
        # Resize heatmap if shapes differ (should be identical)
        if heatmap_color.shape != orig_img.shape:
            heatmap_color = cv2.resize(heatmap_color, (orig_img.shape[1], orig_img.shape[0]))
        
        # Blend the original image and the heatmap color (0.6 original, 0.4 heatmap)
        overlay = cv2.addWeighted(orig_img, 0.6, heatmap_color, 0.4, 0)
        cv2.imwrite(out_overlay_path, overlay)
        print(f"Saved heatmap overlay to: {out_overlay_path}")
    else:
        print("Original page image not found, skipping overlay generation.")

if __name__ == "__main__":
    main()
