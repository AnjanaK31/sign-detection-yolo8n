import os
import sys
import cv2
import numpy as np
from PIL import Image
from data_gen import SyntheticDataGenerator
from line_cleaner import clean_patch_lines, evaluate_cleaning

# Reconfigure stdout to support printing unicode characters to console on Windows
sys.stdout.reconfigure(encoding='utf-8')
sys.stderr.reconfigure(encoding='utf-8')

def run_evaluation_tests(output_dir="debug_evaluation"):
    os.makedirs(output_dir, exist_ok=True)
    generator = SyntheticDataGenerator()
    
    # Define test cases: (label/text, line_type, angle_deg)
    test_cases = [
        ("Ra 3.2", "straight", 0),
        ("R 10.0", "curved", 30),
        ("⌀ 40 ± 0.05", "straight", -15),
        ("Ra 6.3 μm", "curved", 45),
        ("perpendicular 0.02 A", "straight", 90),
        ("parallel 0.05 A B", "curved", -45),
        ("0", "straight", 15),
        ("1", "curved", 0)
    ]
    
    scores = []
    
    print("\n==================================================")
    print("RUNNING LINE DELETION ALGORITHM SCORING TESTS")
    print("==================================================\n")
    
    for idx, (text, line_type, angle) in enumerate(test_cases):
        print(f"Test case {idx + 1}: Text='{text}', Line Type='{line_type}', Angle={angle}°")
        
        # 1. Generate test crops with exact ground truth masks
        crop_std, crop_big, gt_text_mask, gt_line_mask = generator.generate_test_crop(
            text=text, line_type=line_type, angle=angle
        )
        
        # 2. Run the line cleaner
        cleaned_crop, review_img, details = clean_patch_lines(crop_std, crop_big)
        
        # 3. Formulate prediction masks for scoring
        # Predicted Text Mask = Pixels present in the cleaned crop threshold
        cleaned_np = np.array(cleaned_crop.convert("L"))
        _, pred_text_mask = cv2.threshold(cleaned_np, 127, 255, cv2.THRESH_BINARY_INV)
        
        # Predicted Line Mask = Pixels erased
        orig_np = np.array(crop_std.convert("L"))
        _, orig_thresh = cv2.threshold(orig_np, 127, 255, cv2.THRESH_BINARY_INV)
        
        # Intersect ground truth masks with orig_thresh to only evaluate on pixels present in the crop
        gt_text_mask_clean = cv2.bitwise_and(gt_text_mask, orig_thresh)
        gt_line_mask_clean = cv2.bitwise_and(gt_line_mask, orig_thresh)
        
        # Erased pixels = orig_thresh - pred_text_mask
        pred_line_mask = cv2.subtract(orig_thresh, pred_text_mask)
        
        # 4. Evaluate and score
        metrics = evaluate_cleaning(pred_text_mask, pred_line_mask, gt_text_mask_clean, gt_line_mask_clean)

        metrics["text"] = text
        metrics["line_type"] = line_type
        scores.append(metrics)
        
        # Print metrics
        print(f"   -> Text Preservation Rate: {metrics['text_preservation_rate'] * 100:.2f}%")
        print(f"   -> Line Deletion Rate:     {metrics['line_deletion_rate'] * 100:.2f}%")
        print(f"   -> Over-deletion Error:    {metrics['text_false_deletion_rate'] * 100:.2f}%")
        print(f"   -> Line Leakage Error:     {metrics['line_leakage_rate'] * 100:.2f}%")
        print(f"   -> Harmonic F1 Score:      {metrics['f1_score'] * 100:.2f}%\n")
        
        # 5. Save visual comparison grid
        # We save standard original, ground truth line/text, cleaned, and review image
        h, w = orig_np.shape
        
        orig_bgr = cv2.cvtColor(orig_np, cv2.COLOR_GRAY2BGR)
        cleaned_bgr = cv2.cvtColor(cleaned_np, cv2.COLOR_GRAY2BGR)
        
        # Review image comes from clean_patch_lines: text=red, lines=blue
        # Convert RGB to BGR for cv2 saving
        review_bgr = cv2.cvtColor(review_img, cv2.COLOR_RGB2BGR)
        
        # Add labels to visualizer
        def label_img(img, txt):
            res = img.copy()
            cv2.putText(res, txt, (5, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 180, 0), 1)
            return res
            
        grid = np.hstack((
            label_img(orig_bgr, "Original"),
            label_img(review_bgr, "Review (Red:Txt, Blue:Line)"),
            label_img(cleaned_bgr, "Cleaned")
        ))
        
        filename = f"test_{idx}_{line_type}.png"
        filepath = os.path.join(output_dir, filename)
        cv2.imwrite(filepath, grid)
        
    # Print overall summary table
    print("==================================================")
    print("SUMMARY RESULTS TABLE")
    print("==================================================")
    print(f"{'Text':<20} | {'Line Type':<10} | {'Text Pres.%':<11} | {'Line Del.%':<11} | {'F1 Score%':<9}")
    print("-" * 72)
    
    avg_tpr = []
    avg_ldr = []
    avg_f1 = []
    
    for s in scores:
        print(f"{s['text']:<20} | {s['line_type']:<10} | {s['text_preservation_rate']*100:>10.2f}% | {s['line_deletion_rate']*100:>10.2f}% | {s['f1_score']*100:>8.2f}%")
        avg_tpr.append(s['text_preservation_rate'])
        avg_ldr.append(s['line_deletion_rate'])
        avg_f1.append(s['f1_score'])
        
    print("-" * 72)
    print(f"{'AVERAGES':<20} | {'-':<10} | {np.mean(avg_tpr)*100:>10.2f}% | {np.mean(avg_ldr)*100:>10.2f}% | {np.mean(avg_f1)*100:>8.2f}%")
    print("==================================================\n")
    print(f"Review comparison grids saved to: {output_dir}")

if __name__ == "__main__":
    run_evaluation_tests()
