# -*- coding: utf-8 -*-
import sys
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")
"""
eval_island_removal.py
-----------------------
Evaluate the island-based line removal against synthetic GT data.

GT colored image format (from data_gen.py):
  - Red pixels  (R>200, G<100, B<100)  = text/annotation pixels  → should be KEPT
  - Blue pixels (R<100, G<100, B>200)  = CAD line pixels           → should be DELETED
  - White                               = background               → ignored

The island-based removal:
  1. Threshold the input image to binary (foreground = ink pixels)
  2. Run connected components
  3. Mark islands with W or H > font_threshold as "lines" → red_mask
  4. Erase red_mask pixels (set to white)

Metrics computed against GT:
  - TPR  (Text Preservation Rate)  = text pixels kept / total text pixels
  - LDR  (Line Deletion Rate)      = line pixels removed / total line pixels
  - F1   = harmonic mean of TPR and LDR
"""

import os
import sys
import cv2
import numpy as np
from PIL import Image

sys.path.insert(0, "d:/Internship/OCR_PDF/internt-ocrmodel")

from preprocessor import to_grayscale, apply_threshold

FONT_THRESHOLD = 20   # pixels – same value used in production


# ──────────────────────────────────────────────────────────────────────────────
# Core helpers
# ──────────────────────────────────────────────────────────────────────────────

def compute_red_mask(page_pil, font_threshold=FONT_THRESHOLD):
    """Return a binary mask of all large-island pixels (candidates for deletion)."""
    gray = to_grayscale(page_pil)
    thresh = apply_threshold(gray)
    binary_fg = cv2.bitwise_not(thresh)          # ink pixels = 255

    num_labels, labels_im, stats, _ = cv2.connectedComponentsWithStats(binary_fg)

    big_mask = np.zeros(binary_fg.shape, dtype=np.uint8)
    for i in range(1, num_labels):
        w = stats[i, cv2.CC_STAT_WIDTH]
        h = stats[i, cv2.CC_STAT_HEIGHT]
        if h > font_threshold or w > font_threshold:
            big_mask[labels_im == i] = 255

    return big_mask   # pixels that will be deleted


def parse_gt(gt_pil):
    """
    Extract text and line masks from the GT colored PIL image.
    Returns (text_mask, line_mask) both uint8 arrays 0/255.
    """
    gt = np.array(gt_pil.convert("RGB"))
    # Red = text
    text_mask = ((gt[:, :, 0] > 200) & (gt[:, :, 1] < 100) & (gt[:, :, 2] < 100)).astype(np.uint8) * 255
    # Blue = lines
    line_mask = ((gt[:, :, 0] < 100) & (gt[:, :, 1] < 100) & (gt[:, :, 2] > 200)).astype(np.uint8) * 255
    return text_mask, line_mask


def evaluate(page_pil, gt_pil, font_threshold=FONT_THRESHOLD, label=""):
    """
    Run island removal and compute metrics.
    Returns dict with TPR, LDR, F1 and masks for visualisation.
    """
    red_mask = compute_red_mask(page_pil, font_threshold)

    # Align GT to page size if they differ (e.g. JPEG downsample)
    gt_resized = gt_pil.resize(page_pil.size, Image.NEAREST)
    text_mask, line_mask = parse_gt(gt_resized)

    # Pixels the method deleted (red_mask set to white)
    # For evaluation we need to know: of those pixels, which were text vs line
    text_deleted   = np.sum((red_mask > 0) & (text_mask > 0))
    text_total     = np.sum(text_mask > 0)
    line_deleted   = np.sum((red_mask > 0) & (line_mask > 0))
    line_total     = np.sum(line_mask > 0)

    tpr = (text_total - text_deleted) / text_total if text_total > 0 else 1.0
    ldr = line_deleted / line_total if line_total > 0 else 0.0
    f1  = 2 * tpr * ldr / (tpr + ldr) if (tpr + ldr) > 0 else 0.0

    print(f"\n{'='*60}")
    print(f"  {label}")
    print(f"{'='*60}")
    print(f"  font_threshold  : {font_threshold} px")
    print(f"  Text pixels     : {text_total:>8}")
    print(f"  Text DELETED    : {text_deleted:>8}  (bad – over-deletion)")
    print(f"  Line pixels     : {line_total:>8}")
    print(f"  Line DELETED    : {line_deleted:>8}  (good)")
    print(f"")
    print(f"  TPR  (text kept)    : {tpr*100:6.2f}%")
    print(f"  LDR  (lines removed): {ldr*100:6.2f}%")
    print(f"  F1                  : {f1*100:6.2f}%")
    print(f"{'='*60}")

    return {
        "tpr": tpr, "ldr": ldr, "f1": f1,
        "text_total": int(text_total), "text_deleted": int(text_deleted),
        "line_total": int(line_total), "line_deleted": int(line_deleted),
        "red_mask": red_mask, "text_mask": text_mask, "line_mask": line_mask,
    }


def save_overlay(page_pil, metrics, out_path):
    """
    Save a visualisation with three highlights on the original page:
      - DARK RED     = text pixels that were wrongly deleted (over-deletion)
      - BRIGHT RED   = line pixels correctly deleted
      - GREEN        = line pixels the method MISSED (under-deletion)
      - BLUE         = text pixels correctly preserved
    """
    canvas = np.array(page_pil.convert("RGB"))

    red_mask   = metrics["red_mask"]
    text_mask  = metrics["text_mask"]
    line_mask  = metrics["line_mask"]

    # Correct deletions  (line + deleted)
    correct_del = (red_mask > 0) & (line_mask > 0)
    # Missed lines       (line + NOT deleted)
    missed_line = (red_mask == 0) & (line_mask > 0)
    # Over-deleted text  (text + deleted)
    over_del    = (red_mask > 0) & (text_mask > 0)
    # Preserved text     (text + NOT deleted) – just keep original color, mark lightly
    # (not drawn to keep the image readable)

    canvas[correct_del]  = [255,  60,  60]   # bright red – line deleted ✓
    canvas[missed_line]  = [ 60, 200,  60]   # green      – line missed  ✗
    canvas[over_del]     = [255, 140,   0]   # orange     – text deleted ✗ (over-deletion)

    Image.fromarray(canvas).save(out_path)
    print(f"  Overlay saved -> {out_path}")


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────

def main():
    out_dir = "d:/Internship/OCR_PDF/internt-ocrmodel/scratch/eval_island"
    os.makedirs(out_dir, exist_ok=True)

    results = []

    # ── 1. Use the pre-generated eval page ───────────────────────────────────
    eval_page_path  = "d:/Internship/OCR_PDF/internt-ocrmodel/eval_output/eval_page.png"
    eval_gt_path    = "d:/Internship/OCR_PDF/internt-ocrmodel/eval_output/eval_page_gt_colored.png"

    if os.path.exists(eval_page_path) and os.path.exists(eval_gt_path):
        page = Image.open(eval_page_path)
        gt   = Image.open(eval_gt_path)
        m = evaluate(page, gt, label="Pre-generated eval_page.png")
        save_overlay(page, m, os.path.join(out_dir, "eval_page_overlay.png"))
        results.append(("eval_page", m))
    else:
        print("WARNING: eval_page.png not found – skipping.")

    # ── 2. Run on all train split synthetic pages ─────────────────────────────
    train_dir = "d:/Internship/OCR_PDF/sign-detection-yolo8n/dataset_yolo/images/train"
    page_files = sorted([f for f in os.listdir(train_dir)
                         if f.endswith(".png") and "_gt_colored" not in f])

    print(f"\nFound {len(page_files)} synthetic training pages.")

    all_tpr, all_ldr, all_f1 = [], [], []

    for pg_name in page_files:
        pg_path = os.path.join(train_dir, pg_name)
        gt_path = os.path.join(train_dir, pg_name.replace(".png", "_gt_colored.png"))

        if not os.path.exists(gt_path):
            print(f"  [SKIP] No GT for {pg_name}")
            continue

        page = Image.open(pg_path)
        gt   = Image.open(gt_path)
        label = f"synthetic/{pg_name}"
        m = evaluate(page, gt, label=label)

        # Save overlay
        overlay_name = pg_name.replace(".png", "_overlay.png")
        save_overlay(page, m, os.path.join(out_dir, overlay_name))

        all_tpr.append(m["tpr"])
        all_ldr.append(m["ldr"])
        all_f1.append(m["f1"])
        results.append((pg_name, m))

    # ── 3. Also generate a fresh synthetic page for a clean test ─────────────
    print("\n--- Generating a fresh synthetic page for evaluation ---")
    try:
        sys.path.insert(0, "d:/Internship/OCR_PDF/internt-ocrmodel")
        from data_gen import SyntheticDataGenerator
        gen = SyntheticDataGenerator()

        # Generate background
        import tempfile, shutil
        tmp_bg = os.path.join(out_dir, "tmp_bg")
        gen.generate_backgrounds(tmp_bg, count=1)
        bg_path = os.path.join(tmp_bg, "bg_0.png")

        fresh_page, fresh_gt, _, _ = gen.generate_full_page(bg_path, num_annotations=12)
        fresh_page_path = os.path.join(out_dir, "fresh_page.png")
        fresh_gt_path   = os.path.join(out_dir, "fresh_page_gt.png")
        fresh_page.save(fresh_page_path)
        fresh_gt.save(fresh_gt_path)

        m = evaluate(fresh_page, fresh_gt, label="fresh synthetic page (12 annotations)")
        save_overlay(fresh_page, m, os.path.join(out_dir, "fresh_page_overlay.png"))
        results.append(("fresh_page", m))
        shutil.rmtree(tmp_bg, ignore_errors=True)
    except Exception as e:
        print(f"  WARNING: Could not generate fresh page: {e}")

    # ── 4. Summary ────────────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print("  OVERALL SUMMARY (synthetic train pages)")
    print(f"{'='*60}")
    if all_tpr:
        import numpy as np
        print(f"  Pages evaluated : {len(all_tpr)}")
        print(f"  Mean TPR        : {np.mean(all_tpr)*100:.2f}%  ± {np.std(all_tpr)*100:.2f}%")
        print(f"  Mean LDR        : {np.mean(all_ldr)*100:.2f}%  ± {np.std(all_ldr)*100:.2f}%")
        print(f"  Mean F1         : {np.mean(all_f1)*100:.2f}%  ± {np.std(all_f1)*100:.2f}%")
    else:
        print("  No GT-paired pages found in train set.")
    print(f"{'='*60}")
    print(f"\nAll overlay images saved to: {out_dir}")


if __name__ == "__main__":
    main()
