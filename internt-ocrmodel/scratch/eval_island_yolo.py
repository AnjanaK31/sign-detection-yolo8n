# -*- coding: utf-8 -*-
"""
eval_island_yolo.py
--------------------
Evaluate island-based line removal WITH YOLO bounding box protection.

Pipeline (mirrors the production code):
  1. Binary threshold the page
  2. Run YOLO OBB → build yolo_mask (fill all detected text boxes)
  3. Connected components → flag islands with W or H > font_threshold
  4. red_mask = big_islands AND NOT yolo_mask  (only erase outside YOLO boxes)
  5. Erase red_mask pixels from original → cleaned image

Metrics against GT:
  - TPR  = text pixels kept / text pixels total
  - LDR  = line pixels removed / line pixels total
  - F1   = harmonic mean of TPR and LDR

Outputs per page  (saved in eval_island_yolo/):
  <prefix>_1_gt.png        - GT: red=text, blue=CAD lines
  <prefix>_2_cleaned.png   - After YOLO+island removal
  <prefix>_3_diff.png      - Difference map
  <prefix>_composite.png   - All 3 side-by-side
"""
import sys, os
if hasattr(sys.stdout, "reconfigure"): sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"): sys.stderr.reconfigure(encoding="utf-8")

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

sys.path.insert(0, "d:/Internship/OCR_PDF/internt-ocrmodel")
from preprocessor import to_grayscale, apply_threshold
from pipeline import load_yolo_model

FONT_THRESHOLD = 20
YOLO_CONF      = 0.25
YOLO_IMGSZ     = 1280
YOLO_PATH      = "D:/Internship/OCR_PDF/YOLO_expression_best.pt"
OUT_DIR        = "d:/Internship/OCR_PDF/internt-ocrmodel/scratch/eval_island_yolo"
os.makedirs(OUT_DIR, exist_ok=True)


# ── helpers ────────────────────────────────────────────────────────────────────

def build_yolo_mask(page_pil, yolo_model):
    """Run YOLO on page and return a filled binary mask of all detected boxes."""
    img_bgr = cv2.cvtColor(np.array(page_pil.convert("RGB")), cv2.COLOR_RGB2BGR)
    results  = yolo_model(img_bgr, verbose=False, conf=YOLO_CONF, imgsz=YOLO_IMGSZ)
    result   = results[0]
    H, W     = img_bgr.shape[:2]
    mask     = np.zeros((H, W), dtype=np.uint8)
    if result.obb is not None and len(result.obb) > 0:
        for corners in result.obb.xyxyxyxy.cpu().numpy():
            pts = np.array(corners, dtype=np.int32)
            cv2.fillPoly(mask, [pts], 255)
    return mask


def compute_red_mask_yolo(page_pil, yolo_mask, font_threshold=FONT_THRESHOLD):
    """Return pixels that will be erased: big islands OUTSIDE YOLO boxes."""
    gray      = to_grayscale(page_pil)
    thresh    = apply_threshold(gray)
    binary_fg = cv2.bitwise_not(thresh)

    num_labels, labels_im, stats, _ = cv2.connectedComponentsWithStats(binary_fg)
    big_mask = np.zeros(binary_fg.shape, dtype=np.uint8)
    for i in range(1, num_labels):
        w = stats[i, cv2.CC_STAT_WIDTH]
        h = stats[i, cv2.CC_STAT_HEIGHT]
        if h > font_threshold or w > font_threshold:
            big_mask[labels_im == i] = 255

    # Only erase what is OUTSIDE YOLO-protected regions
    red_mask = cv2.bitwise_and(big_mask, cv2.bitwise_not(yolo_mask))
    return red_mask, big_mask


def parse_gt(gt_pil, page_size):
    gt = np.array(gt_pil.resize(page_size, Image.NEAREST).convert("RGB"))
    text_mask = ((gt[:,:,0]>200)&(gt[:,:,1]<100)&(gt[:,:,2]<100)).astype(np.uint8)*255
    line_mask = ((gt[:,:,0]<100)&(gt[:,:,1]<100)&(gt[:,:,2]>200)).astype(np.uint8)*255
    return text_mask, line_mask


def label_banner(img_np, text, font_size=26, bg=(20,20,20), fg=(255,255,255)):
    pil  = Image.fromarray(img_np)
    draw = ImageDraw.Draw(pil)
    try:    font = ImageFont.truetype("C:/Windows/Fonts/consola.ttf", font_size)
    except: font = ImageFont.load_default()
    draw.rectangle([0, 0, pil.width, font_size+14], fill=bg)
    draw.text((8, 7), text, fill=fg, font=font)
    return np.array(pil)


def evaluate(red_mask, text_mask, line_mask, label=""):
    text_del  = int(np.sum((red_mask > 0) & (text_mask > 0)))
    text_tot  = int(np.sum(text_mask > 0))
    line_del  = int(np.sum((red_mask > 0) & (line_mask > 0)))
    line_tot  = int(np.sum(line_mask > 0))

    tpr = (text_tot - text_del) / text_tot if text_tot > 0 else 1.0
    ldr = line_del / line_tot              if line_tot > 0 else 0.0
    f1  = 2*tpr*ldr / (tpr+ldr)           if (tpr+ldr) > 0 else 0.0

    print(f"\n{'='*60}")
    print(f"  {label}")
    print(f"{'='*60}")
    print(f"  Text pixels total   : {text_tot:>8}")
    print(f"  Text pixels DELETED : {text_del:>8}  (bad – over-deletion)")
    print(f"  Line pixels total   : {line_tot:>8}")
    print(f"  Line pixels DELETED : {line_del:>8}  (good)")
    print(f"")
    print(f"  TPR  (text kept)    : {tpr*100:6.2f}%")
    print(f"  LDR  (lines removed): {ldr*100:6.2f}%")
    print(f"  F1                  : {f1*100:6.2f}%")
    print(f"{'='*60}")
    return {"tpr": tpr, "ldr": ldr, "f1": f1,
            "text_total": text_tot, "text_deleted": text_del,
            "line_total": line_tot, "line_deleted": line_del}


def make_trio(page_pil, gt_pil, yolo_mask, red_mask, prefix, title, metrics):
    W, H = page_pil.size
    orig = np.array(page_pil.convert("RGB"))
    text_mask, line_mask = parse_gt(gt_pil, (W, H))

    # ── 1. GT ─────────────────────────────────────────────────────────────────
    gt_canvas = np.ones((H, W, 3), dtype=np.uint8) * 255
    gt_canvas[text_mask > 0] = [220,  30,  30]   # red  = text
    gt_canvas[line_mask > 0] = [ 30,  80, 220]   # blue = CAD lines
    # show YOLO boxes as thin green outline
    yolo_border = cv2.dilate(yolo_mask, np.ones((5,5),np.uint8)) - yolo_mask
    gt_canvas[yolo_border > 0] = [0, 180, 0]
    gt_canvas = label_banner(gt_canvas,
        "GROUND TRUTH  (red=text | blue=CAD-lines | green-border=YOLO boxes)")

    # ── 2. Cleaned ────────────────────────────────────────────────────────────
    cleaned = orig.copy()
    cleaned[red_mask > 0] = [255, 255, 255]
    # Draw YOLO boxes as faint green outline so user can see what was protected
    yolo_border2 = cv2.dilate(yolo_mask, np.ones((5,5),np.uint8)) - yolo_mask
    cleaned[yolo_border2 > 0] = [0, 160, 0]
    cleaned = label_banner(cleaned,
        "AFTER YOLO+ISLAND REMOVAL  (green border = YOLO-protected text regions)")

    # ── 3. Diff ───────────────────────────────────────────────────────────────
    diff = orig.copy()
    correct_del = (red_mask > 0) & (line_mask > 0)
    missed_line = (red_mask == 0) & (line_mask > 0)
    over_del    = (red_mask > 0) & (text_mask > 0)
    kept_text   = (red_mask == 0) & (text_mask > 0)

    diff[correct_del] = [255,  50,  50]   # bright red   – correct deletion
    diff[missed_line] = [ 30, 200,  60]   # green        – line missed
    diff[over_del]    = [255, 140,   0]   # orange       – text wrongly deleted
    diff[kept_text]   = [ 30, 100, 255]   # blue         – text correctly kept

    tpr_s = f"{metrics['tpr']*100:.1f}%"
    ldr_s = f"{metrics['ldr']*100:.1f}%"
    f1_s  = f"{metrics['f1']*100:.1f}%"
    diff = label_banner(diff,
        f"DIFF | TPR={tpr_s} LDR={ldr_s} F1={f1_s} "
        f"[red=line-OK | orange=text-WRONG | green=line-MISSED | blue=text-KEPT]",
        font_size=22)

    # ── save individual ────────────────────────────────────────────────────────
    paths = {}
    for name, canvas in [("1_gt", gt_canvas), ("2_cleaned", cleaned), ("3_diff", diff)]:
        p = os.path.join(OUT_DIR, f"{prefix}_{name}.png")
        Image.fromarray(canvas).save(p)
        print(f"  Saved {p}")
        paths[name] = p

    # ── composite ─────────────────────────────────────────────────────────────
    target_h = 900
    def rh(arr, h):
        pil = Image.fromarray(arr)
        r   = h / pil.height
        return np.array(pil.resize((int(pil.width*r), h), Image.LANCZOS))

    gap  = np.ones((target_h, 14, 3), dtype=np.uint8) * 60
    comp = np.concatenate([rh(gt_canvas,target_h), gap,
                           rh(cleaned,  target_h), gap,
                           rh(diff,     target_h)], axis=1)

    title_bar = np.zeros((50, comp.shape[1], 3), dtype=np.uint8)
    title_bar[:] = [15, 15, 15]
    comp_full = np.concatenate([title_bar, comp], axis=0)
    comp_pil  = Image.fromarray(comp_full)
    draw      = ImageDraw.Draw(comp_pil)
    try:    tfont = ImageFont.truetype("C:/Windows/Fonts/consola.ttf", 28)
    except: tfont = ImageFont.load_default()
    draw.text((10, 11), title, fill=(255, 220, 60), font=tfont)

    cp = os.path.join(OUT_DIR, f"{prefix}_composite.png")
    comp_pil.save(cp)
    print(f"  Saved composite -> {cp}")
    return cp


# ── main ──────────────────────────────────────────────────────────────────────

def process_one(page_pil, gt_pil, yolo_model, prefix, title):
    W, H = page_pil.size
    yolo_mask = build_yolo_mask(page_pil, yolo_model)
    n_boxes   = int(np.sum(yolo_mask > 0) > 0)   # rough count
    print(f"  YOLO detected boxes covering {np.sum(yolo_mask>0):,} pixels")

    red_mask, _ = compute_red_mask_yolo(page_pil, yolo_mask)
    text_mask, line_mask = parse_gt(gt_pil, (W, H))

    m = evaluate(red_mask, text_mask, line_mask, label=title)
    make_trio(page_pil, gt_pil, yolo_mask, red_mask, prefix, title, m)
    return m


def main():
    print("Loading YOLO model...")
    yolo_model = load_yolo_model(YOLO_PATH)

    all_results = []

    # ── 1. Pre-generated eval page ─────────────────────────────────────────────
    ep = "d:/Internship/OCR_PDF/internt-ocrmodel/eval_output/eval_page.png"
    eg = "d:/Internship/OCR_PDF/internt-ocrmodel/eval_output/eval_page_gt_colored.png"
    if os.path.exists(ep) and os.path.exists(eg):
        print("\n--- eval_page.png ---")
        m = process_one(Image.open(ep), Image.open(eg), yolo_model,
                        prefix="evalpage",
                        title="eval_page.png  |  YOLO+island removal")
        all_results.append(("eval_page", m))
    else:
        print("WARNING: eval_page.png not found, skipping.")

    # ── 2. Fresh synthetic page ────────────────────────────────────────────────
    print("\n--- Generating fresh synthetic page ---")
    try:
        from data_gen import SyntheticDataGenerator
        import shutil
        gen = SyntheticDataGenerator()
        tmp = os.path.join(OUT_DIR, "_tmp_bg")
        gen.generate_backgrounds(tmp, count=1)
        fresh_page, fresh_gt, _, _ = gen.generate_full_page(
            os.path.join(tmp, "bg_0.png"), num_annotations=12)
        shutil.rmtree(tmp, ignore_errors=True)

        fresh_page.save(os.path.join(OUT_DIR, "fresh_page.png"))
        fresh_gt.save(os.path.join(OUT_DIR, "fresh_page_gt.png"))

        m = process_one(fresh_page, fresh_gt, yolo_model,
                        prefix="freshpage",
                        title="fresh synthetic page (12 annotations)  |  YOLO+island removal")
        all_results.append(("fresh_page", m))
    except Exception as e:
        import traceback
        print(f"WARNING: Could not generate fresh page: {e}")
        traceback.print_exc()

    # ── Summary ────────────────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print("  FINAL SUMMARY")
    print(f"{'='*60}")
    print(f"  {'Page':<20} {'TPR':>8} {'LDR':>8} {'F1':>8}")
    print(f"  {'-'*44}")
    for name, m in all_results:
        print(f"  {name:<20} {m['tpr']*100:>7.1f}% {m['ldr']*100:>7.1f}% {m['f1']*100:>7.1f}%")
    print(f"{'='*60}")
    print(f"\nAll outputs saved to: {OUT_DIR}")


if __name__ == "__main__":
    main()
