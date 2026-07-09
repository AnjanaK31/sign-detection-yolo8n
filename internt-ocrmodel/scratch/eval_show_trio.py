# -*- coding: utf-8 -*-
import sys
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

import os
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

sys.path.insert(0, "d:/Internship/OCR_PDF/internt-ocrmodel")
from preprocessor import to_grayscale, apply_threshold

FONT_THRESHOLD = 20
OUT_DIR = "d:/Internship/OCR_PDF/internt-ocrmodel/scratch/eval_island"
os.makedirs(OUT_DIR, exist_ok=True)


def compute_red_mask(page_pil, font_threshold=FONT_THRESHOLD):
    gray  = to_grayscale(page_pil)
    thresh = apply_threshold(gray)
    binary_fg = cv2.bitwise_not(thresh)
    num_labels, labels_im, stats, _ = cv2.connectedComponentsWithStats(binary_fg)
    big_mask = np.zeros(binary_fg.shape, dtype=np.uint8)
    for i in range(1, num_labels):
        w = stats[i, cv2.CC_STAT_WIDTH]
        h = stats[i, cv2.CC_STAT_HEIGHT]
        if h > font_threshold or w > font_threshold:
            big_mask[labels_im == i] = 255
    return big_mask


def parse_gt(gt_pil, page_size):
    gt = np.array(gt_pil.resize(page_size, Image.NEAREST).convert("RGB"))
    text_mask = ((gt[:,:,0]>200) & (gt[:,:,1]<100) & (gt[:,:,2]<100)).astype(np.uint8)*255
    line_mask = ((gt[:,:,0]<100) & (gt[:,:,1]<100) & (gt[:,:,2]>200)).astype(np.uint8)*255
    return text_mask, line_mask


def add_label(img_np, text, font_size=28, bg=(30,30,30), fg=(255,255,255)):
    """Burn a label banner onto the top of an RGB numpy image."""
    pil = Image.fromarray(img_np)
    draw = ImageDraw.Draw(pil)
    try:
        font = ImageFont.truetype("C:/Windows/Fonts/consola.ttf", font_size)
    except Exception:
        font = ImageFont.load_default()
    draw.rectangle([0, 0, pil.width, font_size + 12], fill=bg)
    draw.text((8, 6), text, fill=fg, font=font)
    return np.array(pil)


def make_trio(page_pil, gt_pil, prefix, title):
    """
    Generate and save three images:
      1. GT visualisation      – text=red, lines=blue, background=white
      2. After removal         – cleaned page (red-mask pixels set to white)
      3. Difference map        – bright-red=correctly deleted line, orange=wrongly deleted text,
                                 green=missed lines, dark-blue=correctly kept text
    Also saves a horizontal composite of all three.
    """
    W, H = page_pil.size
    red_mask = compute_red_mask(page_pil)
    text_mask, line_mask = parse_gt(gt_pil, (W, H))

    orig = np.array(page_pil.convert("RGB"))

    # ── 1. GT image ──────────────────────────────────────────────────────────
    gt_canvas = np.ones((H, W, 3), dtype=np.uint8) * 255   # white bg
    gt_canvas[text_mask > 0] = [220, 30,  30]   # red  = text
    gt_canvas[line_mask > 0] = [ 30, 80, 220]   # blue = lines
    gt_canvas = add_label(gt_canvas, "GROUND TRUTH  (red=text  blue=CAD-lines)")

    # ── 2. Cleaned (removed) image ────────────────────────────────────────────
    cleaned = orig.copy()
    cleaned[red_mask > 0] = [255, 255, 255]      # erase detected lines
    cleaned = add_label(cleaned, "AFTER ISLAND REMOVAL  (erased pixels = white)")

    # ── 3. Difference / error map ──────────────────────────────────────────────
    diff = orig.copy()
    correct_del  = (red_mask > 0) & (line_mask > 0)   # line correctly deleted
    missed_line  = (red_mask == 0) & (line_mask > 0)  # line NOT deleted (missed)
    over_del     = (red_mask > 0) & (text_mask > 0)   # text wrongly deleted
    kept_text    = (red_mask == 0) & (text_mask > 0)  # text correctly kept

    diff[correct_del] = [255,  50,  50]   # bright red   – correct deletion
    diff[missed_line] = [ 30, 200,  60]   # green        – missed line
    diff[over_del]    = [255, 140,   0]   # orange       – over-deletion (bad)
    diff[kept_text]   = [ 30, 100, 255]   # blue         – text preserved

    legend = ("bright-red=line deleted OK | green=line MISSED | "
              "orange=text WRONGLY deleted | blue=text kept OK")
    diff = add_label(diff, legend, font_size=22)

    # ── Save individual images ────────────────────────────────────────────────
    gt_path      = os.path.join(OUT_DIR, f"{prefix}_1_gt.png")
    cleaned_path = os.path.join(OUT_DIR, f"{prefix}_2_cleaned.png")
    diff_path    = os.path.join(OUT_DIR, f"{prefix}_3_diff.png")

    Image.fromarray(gt_canvas).save(gt_path)
    Image.fromarray(cleaned).save(cleaned_path)
    Image.fromarray(diff).save(diff_path)

    # ── Composite: scale each to same height then paste side-by-side ──────────
    target_h = 900
    def resize_h(img_np, h):
        pil = Image.fromarray(img_np)
        ratio = h / pil.height
        new_w = int(pil.width * ratio)
        return np.array(pil.resize((new_w, h), Image.LANCZOS))

    gt_r   = resize_h(gt_canvas, target_h)
    cl_r   = resize_h(cleaned,   target_h)
    di_r   = resize_h(diff,      target_h)

    gap = np.ones((target_h, 12, 3), dtype=np.uint8) * 80  # dark grey separator

    composite = np.concatenate([gt_r, gap, cl_r, gap, di_r], axis=1)

    # Add top title bar
    title_bar = np.zeros((46, composite.shape[1], 3), dtype=np.uint8)
    title_bar[:] = [20, 20, 20]
    comp_pil = Image.fromarray(np.concatenate([title_bar, composite], axis=0))
    draw = ImageDraw.Draw(comp_pil)
    try:
        tfont = ImageFont.truetype("C:/Windows/Fonts/consola.ttf", 30)
    except Exception:
        tfont = ImageFont.load_default()
    draw.text((10, 8), title, fill=(255, 220, 60), font=tfont)
    composite_np = np.array(comp_pil)

    comp_path = os.path.join(OUT_DIR, f"{prefix}_composite.png")
    Image.fromarray(composite_np).save(comp_path)

    print(f"Saved: {gt_path}")
    print(f"Saved: {cleaned_path}")
    print(f"Saved: {diff_path}")
    print(f"Saved composite: {comp_path}")
    return comp_path


# ─── Run on pre-generated eval page ──────────────────────────────────────────
eval_page = Image.open("d:/Internship/OCR_PDF/internt-ocrmodel/eval_output/eval_page.png")
eval_gt   = Image.open("d:/Internship/OCR_PDF/internt-ocrmodel/eval_output/eval_page_gt_colored.png")
make_trio(eval_page, eval_gt,
          prefix="evalpage",
          title="eval_page.png  |  font_threshold=20px  (no YOLO)")

# ─── Generate and run on a fresh synthetic page ───────────────────────────────
print("\nGenerating fresh synthetic page...")
from data_gen import SyntheticDataGenerator
import shutil

gen = SyntheticDataGenerator()
tmp = os.path.join(OUT_DIR, "_tmp_bg")
gen.generate_backgrounds(tmp, count=1)
fresh_page, fresh_gt, _, _ = gen.generate_full_page(
    os.path.join(tmp, "bg_0.png"), num_annotations=12)
shutil.rmtree(tmp, ignore_errors=True)

make_trio(fresh_page, fresh_gt,
          prefix="freshpage",
          title="fresh synthetic page  |  font_threshold=20px  (no YOLO)")

print("\nDone. All images in:", OUT_DIR)
