import os
import argparse
import math
import json
import cv2
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from PIL import Image, ImageDraw, ImageFont
from shapely.geometry import Polygon

from ultralytics import YOLO
from rectifier import rectify_crop
from classifier import SymbolClassifier, CLASSES, IDX_TO_CLASS

# ── Display charset for Unicode drawing ─────────────────────────────────────────
CLASS_TO_CHAR = {
    '0': '0', '1': '1', '2': '2', '3': '3', '4': '4',
    '5': '5', '6': '6', '7': '7', '8': '8', '9': '9',
    'plus_minus': '±', 'diameter': '⌀', 'radius': 'R',
    'Rz': 'Rz', 'Ra': 'Ra', 'perpendicular': '⊥',
    'parallel': '∥', 'circularity': '○',
    'true_position': '⊕', 'arrow': '→', 'comma': ','
}

def load_font(size=18):
    for p in ["C:\\Windows\\Fonts\\seguisym.ttf",
              "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
              "/usr/share/fonts/truetype/freefont/FreeSans.ttf",
              "DejaVuSans.ttf",
              "C:\\Windows\\Fonts\\arial.ttf",
              "C:\\Windows\\Fonts\\calibri.ttf",
              "C:\\Windows\\Fonts\\segoeui.ttf"]:
        if os.path.exists(p):
            try:
                return ImageFont.truetype(p, size)
            except Exception:
                pass
    return ImageFont.load_default()

def calculate_iou(box1, box2):
    try:
        poly1 = Polygon(box1)
        poly2 = Polygon(box2)
        if not poly1.is_valid:
            poly1 = poly1.buffer(0)
        if not poly2.is_valid:
            poly2 = poly2.buffer(0)
        inter = poly1.intersection(poly2).area
        union = poly1.area + poly2.area - inter
        return inter / union if union > 0 else 0
    except Exception:
        return 0

def get_padded_crop(img_np: np.ndarray,
                    cx: float, cy: float,
                    w: float, h: float,
                    pad_factor: float = 2.0) -> np.ndarray:
    """Returns an axis-aligned square crop centered at (cx, cy) padded with white."""
    size = int(math.ceil(max(w, h) * pad_factor))
    half = size // 2
    ih, iw = img_np.shape[:2]
    x0 = max(0, int(cx) - half)
    y0 = max(0, int(cy) - half)
    x1 = min(iw, x0 + size)
    y1 = min(ih, y0 + size)
    crop = img_np[y0:y1, x0:x1]
    
    # Ensure square with white padding
    sq = np.ones((size, size, 3) if img_np.ndim == 3 else (size, size), dtype=np.uint8) * 255
    ch, cw = crop.shape[:2]
    sq[:ch, :cw] = crop
    return sq

def get_angle_from_corners(corners):
    """Calculates box rotation angle in degrees from its 4 corners (TL, TR, BR, BL order)."""
    p1 = corners[0]
    p2 = corners[1]
    dx = p2[0] - p1[0]
    dy = p2[1] - p1[1]
    angle = math.degrees(math.atan2(dy, dx))
    return angle

def run_single_image(image_path, yolo, clf, args, gt_db, page_pil=None):
    if page_pil is None:
        if not os.path.exists(image_path):
            print(f"ERROR: File not found: {image_path}")
            return
        try:
            page_pil = Image.open(image_path).convert("RGB")
        except Exception as e:
            print(f"ERROR: Failed to open image file {image_path}: {e}")
            return

    base_name = os.path.splitext(os.path.basename(image_path))[0]
    out_dir = os.path.join(args.out, f"output_{base_name}")
    os.makedirs(out_dir, exist_ok=True)
    print(f"\nProcessing: {image_path}")
    print(f"Created output folder: {out_dir}")

    # Pipeline Preprocessing (Stages 1-3)
    from preprocessor import preprocess_stages
    raw_np, gray_np, thresh_np, cleaned_np = preprocess_stages(page_pil)
    clean_page_pil = Image.fromarray(cv2.cvtColor(cleaned_np, cv2.COLOR_GRAY2RGB))

    # YOLO OBB Detection
    img_bgr = cv2.cvtColor(cleaned_np, cv2.COLOR_GRAY2BGR)
    print("  Running YOLO-OBB symbol detection...")
    results = yolo(img_bgr, verbose=False, conf=args.conf, imgsz=1280)
    result = results[0]

    # Find if filename or its base matches a key in ground_truth.json
    filename_key = os.path.basename(image_path)
    gt_list = []
    matched_key = None
    if filename_key in gt_db:
        gt_list = gt_db[filename_key]
        matched_key = filename_key
    else:
        # Try exact basename match (ignoring extensions)
        for key in gt_db.keys():
            if base_name == os.path.splitext(key)[0]:
                gt_list = gt_db[key]
                matched_key = key
                print(f"  Matched input image to ground truth database key: '{key}'")
                break

    # Resolve coordinate scaling if matched
    scale_x, scale_y = 1.0, 1.0
    if matched_key:
        # Search for the original image in the workspace to get its dimensions
        orig_img_path = None
        for root, dirs, files in os.walk("dataset_yolo"):
            if matched_key in files:
                orig_img_path = os.path.join(root, matched_key)
                break
        
        if orig_img_path:
            try:
                with Image.open(orig_img_path) as orig_img:
                    orig_w, orig_h = orig_img.width, orig_img.height
                if orig_w != page_pil.width or orig_h != page_pil.height:
                    scale_x = page_pil.width / orig_w
                    scale_y = page_pil.height / orig_h
                    print(f"  Scaling ground truth coordinates from {orig_w}x{orig_h} to {page_pil.width}x{page_pil.height} (Scale X: {scale_x:.3f}, Y: {scale_y:.3f})")
            except Exception as e:
                print(f"  Warning: Could not check original image dimensions for scaling: {e}")

    # Process Detections and match with Ground Truth
    detections = []
    matched_gt_indices = set()

    if result.obb is not None and len(result.obb) > 0:
        xywhr = result.obb.xywhr.cpu().numpy()
        xyxyxyxy = result.obb.xyxyxyxy.cpu().numpy()
        yolo_confs = result.obb.conf.cpu().numpy()
        print(f"  Detected {len(xywhr)} symbols. Rectifying and classifying...")

        for idx in range(len(xywhr)):
            cx, cy, w, h, r = xywhr[idx]
            corners = xyxyxyxy[idx]
            yolo_conf = float(yolo_confs[idx])
            angle_deg = math.degrees(r)

            # Preprocessing crops for visual grid
            raw_crop = get_padded_crop(raw_np, cx, cy, w, h, pad_factor=2.0)
            gray_crop = get_padded_crop(cv2.cvtColor(gray_np, cv2.COLOR_GRAY2RGB), cx, cy, w, h, pad_factor=2.0)
            thresh_crop = get_padded_crop(cv2.cvtColor(thresh_np, cv2.COLOR_GRAY2RGB), cx, cy, w, h, pad_factor=2.0)
            cleaned_crop = get_padded_crop(cv2.cvtColor(cleaned_np, cv2.COLOR_GRAY2RGB), cx, cy, w, h, pad_factor=2.0)

            # Rectification
            rect_crop_pil = rectify_crop(
                clean_page_pil,
                bbox_metrics={"cx": cx, "cy": cy, "w": w, "h": h, "angle": angle_deg},
                buffer_percent=0.08
            )

            # Classify
            class_name, class_conf = clf.predict(rect_crop_pil)
            char_display = CLASS_TO_CHAR.get(class_name, class_name)

            # Match with Ground Truth
            gt_match = None
            best_iou = 0.0
            best_gt_idx = -1

            for g_idx, g in enumerate(gt_list):
                # Scale corners for IoU calculation
                g_corners_scaled = [[pt[0] * scale_x, pt[1] * scale_y] for pt in g["corners"]]
                iou = calculate_iou(corners.tolist(), g_corners_scaled)
                if iou > best_iou:
                    best_iou = iou
                    best_gt_idx = g_idx

            is_class_correct = None
            is_orient_correct = None
            gt_expected_char = None
            gt_angle = None

            if best_iou > 0.4:
                gt_match = gt_list[best_gt_idx]
                matched_gt_indices.add(best_gt_idx)
                
                # Check classification correctness
                gt_class_idx = gt_match["class"]
                gt_class_name = CLASSES[gt_class_idx]
                gt_expected_char = CLASS_TO_CHAR.get(gt_class_name, gt_class_name)
                is_class_correct = (class_name == gt_class_name)

                # Check orientation correctness
                scaled_gt_corners = [[pt[0] * scale_x, pt[1] * scale_y] for pt in gt_match["corners"]]
                gt_angle = get_angle_from_corners(scaled_gt_corners)
                ang_diff = (angle_deg - gt_angle) % 180
                if ang_diff > 90:
                    ang_diff -= 180
                is_orient_correct = (abs(ang_diff) < 20.0)

            detections.append({
                "id": idx + 1,
                "cx": cx, "cy": cy, "w": w, "h": h,
                "angle": angle_deg,
                "corners": corners.tolist(),
                "yolo_conf": yolo_conf,
                "pred_class": class_name,
                "pred_char": char_display,
                "class_confidence": class_conf,
                "raw_crop": raw_crop,
                "gray_crop": gray_crop,
                "thresh_crop": thresh_crop,
                "cleaned_crop": cleaned_crop,
                "rect_crop": np.array(rect_crop_pil.convert("RGB")),
                "best_iou": best_iou,
                "is_class_correct": is_class_correct,
                "is_orient_correct": is_orient_correct,
                "gt_expected_char": gt_expected_char,
                "gt_angle": gt_angle
            })
    else:
        print("  No symbols detected on the page.")

    # Image A: Annotated Page Image with OBB boxes & Prediction text overlayed
    annotated_page = clean_page_pil.copy()
    draw = ImageDraw.Draw(annotated_page, "RGBA")
    font = load_font(size=18)
    
    for det in detections:
        corners = det["corners"]
        poly = [(float(pt[0]), float(pt[1])) for pt in corners]
        
        # Color based on correctness if GT available
        if det["is_class_correct"] is True:
            stroke_color = (0, 200, 100, 255) # Green
            fill_color = (0, 200, 100, 20)
        elif det["is_class_correct"] is False:
            stroke_color = (255, 60, 60, 255) # Red
            fill_color = (255, 60, 60, 20)
        else:
            stroke_color = (0, 180, 220, 255) # Cyan (Default)
            fill_color = (0, 180, 220, 20)

        draw.polygon(poly, outline=stroke_color, fill=fill_color, width=3)
        
        # Draw label
        lbl = f"#{det['id']} '{det['pred_char']}' ({det['class_confidence']:.1%})"
        lx, ly = poly[0][0], poly[0][1] - 22
        draw.text((lx, ly), lbl, font=font, fill=(20, 20, 20) if stroke_color[1] > 150 else (220, 20, 20))

    page_save_path = os.path.join(out_dir, "page_with_bboxes.png")
    annotated_page.save(page_save_path)
    print(f"  Saved page overlay: {page_save_path}")

    # Image B: High-Quality Preprocessing & Recognition Grid Table
    n_dets = len(detections)
    if n_dets > 0:
        fig_w = 14.0
        fig_h = 1.25 * n_dets + 1.8
        fig = plt.figure(figsize=(fig_w, fig_h), facecolor="#0f111a")
        
        # Calculate summary statistics
        has_gt = any(d["best_iou"] > 0.0 for d in detections)
        matched_dets = [d for d in detections if d["best_iou"] > 0.4]
        
        correct_class = sum(1 for d in matched_dets if d["is_class_correct"] is True)
        correct_orient = sum(1 for d in matched_dets if d["is_orient_correct"] is True)
        
        class_acc = (correct_class / len(matched_dets)) if matched_dets else 0.0
        orient_acc = (correct_orient / len(matched_dets)) if matched_dets else 0.0
        
        title_str = f"Pipeline Performance Summary Grid  │  Detections: {n_dets}"
        if gt_list:
            title_str += f"  │  Ground Truth: {len(gt_list)}  │  Classification Accuracy: {class_acc:.1%} ({correct_class}/{len(matched_dets)})  │  Orientation Score: {orient_acc:.1%} ({correct_orient}/{len(matched_dets)})"
        
        fig.suptitle(title_str, color="white", fontsize=13, fontweight="bold", y=0.99)
        
        outer_gs = GridSpec(n_dets, 1, figure=fig, hspace=0.15, top=0.96, bottom=0.01)
        
        for i, det in enumerate(detections):
            row_gs = outer_gs[i, 0].subgridspec(1, 9, wspace=0.08, width_ratios=[0.5, 1.1, 1.1, 1.1, 1.1, 1.1, 1.6, 1.2, 1.2])
            
            # 1. Index column
            ax_idx = fig.add_subplot(row_gs[0, 0])
            ax_idx.set_facecolor("#161822")
            ax_idx.set_xticks([])
            ax_idx.set_yticks([])
            for spine in ax_idx.spines.values():
                spine.set_color("#232738")
            ax_idx.text(0.5, 0.5, f"#{det['id']}", color="#8b949e", fontsize=11, fontweight="bold", ha="center", va="center")
            
            # Preprocessing Steps (Raw -> Gray -> Thresh -> Clean -> Rect)
            crops = [
                ("Raw", det["raw_crop"]),
                ("Gray", det["gray_crop"]),
                ("Threshold", det["thresh_crop"]),
                ("Cleaned", det["cleaned_crop"]),
                ("Rectified", det["rect_crop"])
            ]
            
            for c_idx, (name, crop_img) in enumerate(crops):
                ax_img = fig.add_subplot(row_gs[0, c_idx + 1])
                ax_img.set_xticks([])
                ax_img.set_yticks([])
                for spine in ax_img.spines.values():
                    spine.set_color("#232738")
                    spine.set_linewidth(0.8)
                ax_img.imshow(crop_img)
                if i == 0:
                    ax_img.set_title(name, color="#58a6ff", fontsize=9, fontweight="bold", pad=4)
            
            # 7. Prediction details
            ax_pred = fig.add_subplot(row_gs[0, 6])
            ax_pred.set_facecolor("#161822")
            ax_pred.set_xticks([])
            ax_pred.set_yticks([])
            for spine in ax_pred.spines.values():
                spine.set_color("#232738")
            
            pred_text = f"Pred: '{det['pred_char']}'\nConf: {det['class_confidence']:.1%}\nAngle: {det['angle']:+.0f}°"
            ax_pred.text(0.1, 0.5, pred_text, color="#e6edf3", fontsize=9, va="center", ha="left", fontfamily="monospace")
            if i == 0:
                ax_pred.set_title("Classification", color="#58a6ff", fontsize=9, fontweight="bold", pad=4)

            # 8. Classification Correctness (✓ / ✗)
            ax_corr = fig.add_subplot(row_gs[0, 7])
            ax_corr.set_facecolor("#161822")
            ax_corr.set_xticks([])
            ax_corr.set_yticks([])
            for spine in ax_corr.spines.values():
                spine.set_color("#232738")
            
            if det["is_class_correct"] is True:
                ax_corr.text(0.5, 0.5, "✓", color="#3fb950", fontsize=28, fontweight="bold", ha="center", va="center")
            elif det["is_class_correct"] is False:
                ax_corr.text(0.5, 0.6, "✗", color="#f85149", fontsize=28, fontweight="bold", ha="center", va="center")
                ax_corr.text(0.5, 0.2, f"Exp: '{det['gt_expected_char']}'", color="#f85149", fontsize=7.5, ha="center", va="center")
            else:
                ax_corr.text(0.5, 0.5, "-", color="#8b949e", fontsize=24, ha="center", va="center")
            
            if i == 0:
                ax_corr.set_title("Correct Class?", color="#58a6ff", fontsize=9, fontweight="bold", pad=4)

            # 9. Orientation Correctness (✓ / ✗)
            ax_orient = fig.add_subplot(row_gs[0, 8])
            ax_orient.set_facecolor("#161822")
            ax_orient.set_xticks([])
            ax_orient.set_yticks([])
            for spine in ax_orient.spines.values():
                spine.set_color("#232738")
            
            if det["is_orient_correct"] is True:
                ax_orient.text(0.5, 0.5, "✓", color="#3fb950", fontsize=28, fontweight="bold", ha="center", va="center")
            elif det["is_orient_correct"] is False:
                ax_orient.text(0.5, 0.6, "✗", color="#f85149", fontsize=28, fontweight="bold", ha="center", va="center")
                ax_orient.text(0.5, 0.2, f"GT: {det['gt_angle']:+.0f}°", color="#f85149", fontsize=7.5, ha="center", va="center")
            else:
                ax_orient.text(0.5, 0.5, "-", color="#8b949e", fontsize=24, ha="center", va="center")
            
            if i == 0:
                ax_orient.set_title("Correct Orient?", color="#58a6ff", fontsize=9, fontweight="bold", pad=4)

        grid_save_path = os.path.join(out_dir, "pipeline_summary_grid.png")
        fig.savefig(grid_save_path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
        plt.close(fig)
        print(f"  Saved summary grid image: {grid_save_path}")

        # Write text metrics and save a JSON report
        report = {
            "image": filename_key,
            "detections_count": n_dets,
            "ground_truth_count": len(gt_list),
            "matched_detections": len(matched_dets),
            "correct_classifications": correct_class,
            "correct_orientations": correct_orient,
            "classification_accuracy": class_acc,
            "orientation_accuracy": orient_acc,
            "details": [
                {
                    "id": d["id"],
                    "prediction": d["pred_char"],
                    "yolo_conf": round(d["yolo_conf"], 4),
                    "class_conf": round(d["class_confidence"], 4),
                    "pred_angle": round(d["angle"], 2),
                    "gt_expected": d["gt_expected_char"],
                    "gt_angle": round(d["gt_angle"], 2) if d["gt_angle"] is not None else None,
                    "is_class_correct": d["is_class_correct"],
                    "is_orient_correct": d["is_orient_correct"],
                    "iou": round(d["best_iou"], 4),
                    "cx": float(d["cx"]),
                    "cy": float(d["cy"]),
                    "w": float(d["w"]),
                    "h": float(d["h"]),
                    "corners": d["corners"]
                } for d in detections
            ]
        }
        
        report_save_path = os.path.join(out_dir, "pipeline_report.json")
        with open(report_save_path, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=4)
        print(f"  Saved text report: {report_save_path}")
        
        # Output summary metrics on the terminal
        print("\n" + "="*50)
        print(f" PIPELINE PERFORMANCE REPORT - {filename_key}")
        print("="*50)
        print(f"Total Detected Symbols : {n_dets}")
        if gt_list:
            print(f"Total Ground Truths    : {len(gt_list)}")
            print(f"Matched Detections     : {len(matched_dets)}")
            print(f"Correct Classifications: {correct_class} ({class_acc:.1%})")
            print(f"Correct Orientations   : {correct_orient} ({orient_acc:.1%})")
        print("="*50)
    else:
        print("  No detections to render in the summary grid.")

def main():
    parser = argparse.ArgumentParser(description="High-Quality Pipeline Inference & Visualizer")
    parser.add_argument("--input", required=True, help="Path to test image, PDF, or folder containing images")
    parser.add_argument("--yolo", default="weights/best.pt", help="Path to YOLOv8-OBB model weights")
    parser.add_argument("--classifier", default="classifier_best_updated.pt", help="Path to MobileNetV3 classifier weights")
    parser.add_argument("--out", default="output_pipeline", help="Root directory for outputs")
    parser.add_argument("--conf", type=float, default=0.25, help="YOLO OBB confidence threshold")
    parser.add_argument("--gt", default="dataset_yolo/ground_truth.json", help="Path to ground truth database JSON")
    args = parser.parse_args()

    # 1. Resolve paths and models
    yolo_path = args.yolo
    if not os.path.exists(yolo_path):
        for fallback in ["runs/obb/runs/obb/trained_on_1000_pdfs-2/weights/best.pt",
                         "runs/obb/yolo_obb_project/symbol_obb_train-2/weights/best.pt",
                         "yolov8n-obb.pt"]:
            if os.path.exists(fallback):
                yolo_path = fallback
                break
    
    clf_path = args.classifier
    if not os.path.exists(clf_path):
        for fallback in ["classifier_best_bigdatset.pt", "classifier_best.pt"]:
            if os.path.exists(fallback):
                clf_path = fallback
                break

    print(f"Loading YOLOv8-OBB from: {yolo_path}")
    yolo = YOLO(yolo_path)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Loading classifier from: {clf_path} (Device: {device})")
    clf = SymbolClassifier(model_path=clf_path, device=device)

    # Load Ground Truth database if available
    gt_db = {}
    if os.path.exists(args.gt):
        try:
            with open(args.gt, "r") as f:
                gt_db = json.load(f)
            print(f"Loaded ground truth database: {args.gt}")
        except Exception as e:
            print(f"Warning: Could not load ground truth file. {e}")

    # Check input type
    if os.path.isdir(args.input):
        print(f"Input is a directory: {args.input}. Scanning for images and PDFs...")
        exts = {".png", ".jpg", ".jpeg", ".bmp", ".tiff", ".tif"}
        files = sorted(os.listdir(args.input))
        for f in files:
            f_path = os.path.join(args.input, f)
            ext = os.path.splitext(f)[1].lower()
            if ext in exts:
                run_single_image(f_path, yolo, clf, args, gt_db)
            elif ext == ".pdf":
                print(f"Processing PDF in folder: {f_path}")
                from pdf2image import convert_from_path
                try:
                    pages = convert_from_path(f_path, dpi=200)
                    for i, page in enumerate(pages):
                        run_single_image(f_path + f"_page_{i+1}.png", yolo, clf, args, gt_db, page_pil=page.convert("RGB"))
                except Exception as e:
                    print(f"ERROR: Failed to process PDF {f_path}. {e}")
    else:
        is_pdf = args.input.lower().endswith(".pdf")
        if is_pdf:
            print(f"Converting PDF: {args.input} at 200 DPI...")
            from pdf2image import convert_from_path
            pages = convert_from_path(args.input, dpi=200)
            if not pages:
                print("ERROR: Failed to extract pages from PDF.")
                return
            for i, page in enumerate(pages):
                run_single_image(args.input + f"_page_{i+1}.png", yolo, clf, args, gt_db, page_pil=page.convert("RGB"))
        else:
            run_single_image(args.input, yolo, clf, args, gt_db)

if __name__ == "__main__":
    main()
