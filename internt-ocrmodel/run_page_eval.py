import os
import sys
import torch
import numpy as np
from PIL import Image

# Reconfigure stdout/stderr to support printing unicode characters on Windows
sys.stdout.reconfigure(encoding='utf-8')
sys.stderr.reconfigure(encoding='utf-8')

# Ensure we can import modules from the current directory
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from data_gen import SyntheticDataGenerator
from pipeline import load_yolo_model, process_page
from classifier import SymbolClassifier

def run_page_evaluation():
    print("==================================================")
    # 1. Initialize Synthetic Data Generator
    print("Initializing Synthetic Data Generator...")
    generator = SyntheticDataGenerator()
    
    # 2. Synthesize page and colored ground truth page
    print("Generating synthetic background...")
    bg_dir = "temp_eval_bg"
    generator.generate_backgrounds(bg_dir, count=1)
    bg_path = os.path.join(bg_dir, "bg_0.png")
    
    print("Generating full page with 15 random symbol expressions...")
    page_img, page_colored, labels, gt_details = generator.generate_full_page(bg_path, num_annotations=15)
    
    # Save the generated page and its ground truth colored page
    output_dir = "eval_output"
    os.makedirs(output_dir, exist_ok=True)
    
    eval_page_path = os.path.join(output_dir, "eval_page.png")
    eval_page_gt_path = os.path.join(output_dir, "eval_page_gt_colored.png")
    
    page_img.save(eval_page_path)
    page_colored.save(eval_page_gt_path)
    
    print(f"Saved evaluation page image: {eval_page_path}")
    print(f"Saved colored ground truth page image: {eval_page_gt_path}")
    
    # 3. Locate trained model files
    print("\nLocating trained models...")
    yolo_paths = [
        "../YOLO_expression_best.pt",
        "../sign-detection-yolo8n/YOLO_expression_best.pt",
        "D:/Internship/OCR_PDF/YOLO_expression_best.pt"
    ]
    yolo_path = None
    for path in yolo_paths:
        if os.path.exists(path):
            yolo_path = path
            break
            
    if yolo_path is None:
        print("ERROR: Could not find YOLO_expression_best.pt in standard paths.")
        sys.exit(1)
    print(f"Using YOLO model: {yolo_path}")
        
    classifier_paths = [
        "../sign-detection-yolo8n/classifier_best.pt",
        "D:/Internship/OCR_PDF/sign-detection-yolo8n/classifier_best.pt"
    ]
    classifier_path = None
    for path in classifier_paths:
        if os.path.exists(path):
            classifier_path = path
            break
            
    if classifier_path is None:
        print("ERROR: Could not find classifier_best.pt in standard paths.")
        sys.exit(1)
    print(f"Using Classifier model: {classifier_path}")
    
    # 4. Load Models
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Loading models on device: {device}...")
    yolo_model = load_yolo_model(yolo_path)
    classifier = SymbolClassifier(model_path=classifier_path, device=device)
    
    # 5. Run page-level pipeline with colored GT cross-checking
    print("\nRunning pipeline inference and evaluation on the synthetic page...")
    annotated_img, detections = process_page(
        page_img,
        yolo_model,
        classifier,
        conf_threshold=0.25,
        output_dir=output_dir,
        gt_colored_pil=page_colored
    )
    
    # Save annotated result
    annotated_save_path = os.path.join(output_dir, "eval_page_annotated.png")
    annotated_img.save(annotated_save_path)
    print(f"Saved annotated page image: {annotated_save_path}")
    
    # Extract evaluation metrics from detections
    evals = [d["evaluation"] for d in detections if "evaluation" in d]
    
    print("\n" + "="*50)
    print("DETECTION & EVALUATION SUMMARY REPORT")
    print("="*50)
    print(f"{'Crop Index':<10} | {'YOLO Conf':<10} | {'Text':<15} | {'TPR %':<8} | {'LDR %':<8} | {'F1 %':<8}")
    print("-" * 72)
    
    tprs, ldrs, f1s = [], [], []
    for d in detections:
        idx = d["idx"]
        yolo_conf = d["yolo_conf"]
        text = d["text"]
        
        tpr_str, ldr_str, f1_str = "-", "-", "-"
        if "evaluation" in d:
            e = d["evaluation"]
            tpr_str = f"{e['text_preservation_rate']*100:.1f}%"
            ldr_str = f"{e['line_deletion_rate']*100:.1f}%"
            f1_str = f"{e['f1_score']*100:.1f}%"
            
            tprs.append(e['text_preservation_rate'])
            ldrs.append(e['line_deletion_rate'])
            f1s.append(e['f1_score'])
            
        print(f"{idx:<10} | {yolo_conf:<10.3f} | {text:<15} | {tpr_str:<8} | {ldr_str:<8} | {f1_str:<8}")
        
    print("-" * 72)
    if tprs:
        print(f"{'AVERAGES':<10} | {'-':<10} | {'-':<15} | {np.mean(tprs)*100:>6.2f}% | {np.mean(ldrs)*100:>6.2f}% | {np.mean(f1s)*100:>6.2f}%")
    else:
        print("No evaluations were completed.")
    print("="*50 + "\n")
    
    # Cleanup temp directory
    print("Cleaning up temporary background directory...")
    try:
        import shutil
        shutil.rmtree(bg_dir)
    except Exception as e:
        print(f"Warning: Failed to delete {bg_dir}: {e}")

if __name__ == "__main__":
    run_page_evaluation()
