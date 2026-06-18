"""
visualize_pipeline.py
---------------------
Runs the full pipeline on a test image and produces three rich visualization panels:

  Panel A — Background Cleanup (ZOOMED IN on real symbol regions)
      For each detection: a 3-column strip showing
        • The raw pixel crop (no processing)
        • The grayscale crop
        • The adaptive-threshold crop (what YOLO actually sees)

  Panel B — Orientation Alignment (per detection)
      For each detection: raw tilted crop (with orange OBB box) | rectified upright crop
      + OBB angle and classifier prediction label

  Panel C — Full Annotated Page (resized to a sane resolution)
      The cleaned + annotated output page with all OBB polygons and predicted labels

All outputs are saved to output_pipeline/visualizations/

Usage:
    python visualize_pipeline.py --input test_images/blueprint_0_page0.png
    python visualize_pipeline.py --input test_images/blueprint_0_page0.png --conf 0.20
    python visualize_pipeline.py --input test_images/            # all images in folder
"""

import os
import argparse
import math
import cv2
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from PIL import Image, ImageDraw, ImageFont

from ultralytics import YOLO
from rectifier import rectify_crop
from classifier import SymbolClassifier, IDX_TO_CLASS
from preprocessor import preprocess_stages, full_preprocess

# ── Default paths ─────────────────────────────────────────────────────────────
DEFAULT_YOLO       = "weights_yolo_char/weights/best.pt"
DEFAULT_CLASSIFIER = "classifier_best.pt"
DEFAULT_OUT        = "output_pipeline/visualizations"

# ── Display charset ───────────────────────────────────────────────────────────
CLASS_TO_CHAR = {
    '0': '0', '1': '1', '2': '2', '3': '3', '4': '4',
    '5': '5', '6': '6', '7': '7', '8': '8', '9': '9',
    'plus_minus': '±', 'diameter': '⌀', 'radius': 'R',
    'Rz': 'Rz', 'Ra': 'Ra', 'perpendicular': '⊥',
    'parallel': '∥', 'circularity': '○',
    'true_position': '⊕', 'arrow': '→', 'comma': ','
}

# ── How large the page is allowed to be for YOLO (saves RAM / time) ───────────
MAX_PAGE_DIM = 4000   # pixels on longest side; originals are 6667×8334

# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

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


def scale_image(img_pil: Image.Image, max_dim: int):
    """Downscales an image so its longest side ≤ max_dim, preserving aspect ratio."""
    w, h = img_pil.size
    scale = min(max_dim / max(w, h), 1.0)
    if scale == 1.0:
        return img_pil, scale
    new_w, new_h = int(w * scale), int(h * scale)
    return img_pil.resize((new_w, new_h), Image.LANCZOS), scale



def get_padded_crop(img_np: np.ndarray,
                    cx: float, cy: float,
                    w: float, h: float,
                    pad_factor: float = 1.8) -> np.ndarray:
    """
    Returns an axis-aligned square crop centred at (cx, cy).
    pad_factor controls how much context to include around the bounding box.
    """
    size   = int(math.ceil(max(w, h) * pad_factor))
    half   = size // 2
    ih, iw = img_np.shape[:2]
    x0 = max(0, int(cx) - half)
    y0 = max(0, int(cy) - half)
    x1 = min(iw, x0 + size)
    y1 = min(ih, y0 + size)
    crop = img_np[y0:y1, x0:x1]
    # Ensure square with white padding
    sq = np.ones((size, size, 3) if img_np.ndim == 3 else (size, size),
                 dtype=np.uint8) * 255
    ch, cw = crop.shape[:2]
    sq[:ch, :cw] = crop
    return sq


def draw_obb_on_crop(crop_np: np.ndarray,
                     cx: float, cy: float,
                     w: float, h: float,
                     angle_deg: float,
                     pad_factor: float = 1.8) -> np.ndarray:
    """Overlays the rotated OBB (orange) onto a padded crop."""
    size = int(math.ceil(max(w, h) * pad_factor))
    half = size // 2
    # offset of detection centre within the crop
    ox = half
    oy = half

    rad = math.radians(angle_deg)
    ca, sa = math.cos(rad), math.sin(rad)
    corners_local = [(-w/2, -h/2), (w/2, -h/2), (w/2, h/2), (-w/2, h/2)]
    pts = []
    for lx, ly in corners_local:
        rx = int(ox + lx * ca - ly * sa)
        ry = int(oy + lx * sa + ly * ca)
        pts.append((rx, ry))

    out = crop_np.copy()
    if out.ndim == 2:
        out = cv2.cvtColor(out, cv2.COLOR_GRAY2RGB)
    for i in range(4):
        cv2.line(out, pts[i], pts[(i + 1) % 4], (255, 130, 0), 3)
    # Mark centre
    cv2.circle(out, (ox, oy), 5, (255, 0, 100), -1)
    return out


# ─────────────────────────────────────────────────────────────────────────────
# Panel A  — Background Cleanup   (4 columns: Raw | Gray | Threshold | Cleaned)
# ─────────────────────────────────────────────────────────────────────────────

def save_panel_a(raw_np: np.ndarray,
                 gray_np: np.ndarray,
                 thresh_np: np.ndarray,
                 cleaned_np: np.ndarray,
                 detections_xywhr: list,
                 out_path: str,
                 image_name: str,
                 max_show: int = 12):
    """
    For each detection: 4-column micro-panel showing each preprocessing stage
    zoomed in on the actual symbol region:
        [Raw crop] | [Grayscale] | [Adaptive Threshold] | [After Line Removal]
    """
    data = detections_xywhr[:max_show]
    n = len(data)
    if n == 0:
        print("  [A] No detections — skipping Panel A.")
        return

    STAGES = [
        ("① Raw",               raw_np,    "#e0e0e0"),
        ("② Grayscale",         gray_np,   "#74b9ff"),
        ("③ Threshold",         thresh_np, "#fdcb6e"),
        ("④ Lines Removed",     cleaned_np,"#00d4aa"),
    ]

    cols_per_row = min(n, 3)          # up to 3 detection groups per row
    n_rows       = math.ceil(n / cols_per_row)
    fig_w        = cols_per_row * 9.0   # 4 sub-images × ~2.25 in each
    fig_h        = n_rows * 3.4 + 1.6

    fig = plt.figure(figsize=(fig_w, fig_h), facecolor="#0d1117")
    fig.suptitle(
        f"Panel A — Preprocessing Stages (zoomed per detection)  |  {image_name}",
        color="white", fontsize=14, fontweight="bold", y=1.02
    )

    outer_gs = GridSpec(n_rows, cols_per_row, figure=fig, hspace=0.65, wspace=0.18)

    for det_idx, (cx, cy, w, h, r) in enumerate(data):
        row = det_idx // cols_per_row
        col = det_idx % cols_per_row
        inner_gs = outer_gs[row, col].subgridspec(1, 4, wspace=0.05)

        for ci, (stage_title, stage_np, color) in enumerate(STAGES):
            ax = fig.add_subplot(inner_gs[0, ci])
            ax.set_facecolor("#0d1117")
            ax.set_xticks([])
            ax.set_yticks([])
            for spine in ax.spines.values():
                spine.set_color("#2a2a3a")
                spine.set_linewidth(0.7)

            # Get a padded crop of this stage at the detection location
            crop = get_padded_crop(stage_np if stage_np.ndim == 3
                                   else cv2.cvtColor(stage_np, cv2.COLOR_GRAY2RGB),
                                   cx, cy, w, h, pad_factor=2.0)
            ax.imshow(crop, cmap=None)

            # Show stage label only on the first row of each column
            if row == 0 and det_idx < cols_per_row:
                ax.set_title(stage_title, color=color,
                             fontsize=7.5, pad=3, fontweight="bold")

            # Detection index badge on leftmost column
            if ci == 0:
                ax.text(2, 2, f"#{det_idx+1}", color="#ff9f43",
                        fontsize=7, va="top", ha="left",
                        bbox=dict(facecolor="#0d1117", alpha=0.7,
                                  pad=1, edgecolor="none"))

    fig.savefig(out_path, dpi=150, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"  [A] Saved: {out_path}")


# ─────────────────────────────────────────────────────────────────────────────
# Panel B  — Orientation Alignment  (tilted crop vs rectified crop)
# ─────────────────────────────────────────────────────────────────────────────

def save_panel_b(panel_b_data: list, out_path: str, image_name: str,
                 max_cols: int = 5):
    if not panel_b_data:
        print("  [B] No detections — skipping Panel B.")
        return

    n            = len(panel_b_data)
    pairs_per_row = max_cols
    n_rows        = math.ceil(n / pairs_per_row)

    fig_w = pairs_per_row * 4.6
    fig_h = n_rows * 3.2 + 1.4

    fig = plt.figure(figsize=(fig_w, fig_h), facecolor="#0f0f23")
    fig.suptitle(
        f"Panel B — Orientation Alignment  |  {image_name}",
        color="white", fontsize=14, fontweight="bold", y=1.02
    )

    outer_gs = GridSpec(n_rows, pairs_per_row, figure=fig, hspace=0.65, wspace=0.30)

    for det_idx, det in enumerate(panel_b_data):
        row = det_idx // pairs_per_row
        col = det_idx % pairs_per_row
        inner_gs = outer_gs[row, col].subgridspec(1, 2, wspace=0.06)

        ax_raw  = fig.add_subplot(inner_gs[0, 0])
        ax_rect = fig.add_subplot(inner_gs[0, 1])

        for ax in (ax_raw, ax_rect):
            ax.set_facecolor("#0f0f23")
            ax.set_xticks([])
            ax.set_yticks([])
            for spine in ax.spines.values():
                spine.set_color("#2a2a4a")
                spine.set_linewidth(0.8)

        # ── Left: raw tilted crop with OBB outline ────────────────────────────
        raw = det["raw_obb_crop"]
        if isinstance(raw, Image.Image):
            raw = np.array(raw)
        ax_raw.imshow(raw)
        ang = det["angle_deg"]
        ax_raw.set_title(f"Tilted  {ang:+.1f}°", color="#ff9f43",
                         fontsize=8.5, pad=3, fontweight="bold")

        # ── Right: rectified (upright) crop ───────────────────────────────────
        rect = det["rect_crop"]
        if isinstance(rect, Image.Image):
            rect = np.array(rect)
        ax_rect.imshow(rect)
        char  = det["pred_char"]
        conf  = det["class_conf"]
        col_c = "#00d4aa" if conf >= 0.7 else "#ff6b6b"
        ax_rect.set_title(f"Rectified\n'{char}'  {conf:.2f}",
                          color=col_c, fontsize=8.5, pad=3, fontweight="bold")

        # Detection index badge
        ax_raw.text(2, 2, f"#{det_idx+1}", color="#74b9ff",
                    fontsize=7, va="top", ha="left",
                    bbox=dict(facecolor="#0f0f23", alpha=0.7, pad=1, edgecolor="none"))

    # Hide unused slots
    total_slots = n_rows * pairs_per_row
    for slot in range(n, total_slots):
        r = slot // pairs_per_row
        c = slot % pairs_per_row
        ax = fig.add_subplot(outer_gs[r, c])
        ax.set_visible(False)

    fig.savefig(out_path, dpi=150, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"  [B] Saved: {out_path}  ({n} detections)")


# ─────────────────────────────────────────────────────────────────────────────
# Annotation helper  — draws only small numbered circle tags on the image
# ─────────────────────────────────────────────────────────────────────────────

def annotate_page(pil: Image.Image, detections: list) -> Image.Image:
    """
    Draws on the image:
      - Green OBB polygon outline for each detection
      - A small filled circle with the detection index number (#1, #2…)
        placed at the topmost corner of the OBB
    Labels are intentionally minimal so the symbols remain visible.
    The sidebar in save_panel_c() provides the full legend.
    """
    img  = pil.copy().convert("RGB")
    W, H = img.size

    # Ultra-thin, uncluttered configuration
    outline_w = 2  # Thin borders
    r        = 10  # Tiny circle radius
    font     = load_font(10)  # Small, clean font size

    draw = ImageDraw.Draw(img, "RGBA")

    for i, d in enumerate(detections):
        corners = d["corners"]
        poly    = [(float(c[0]), float(c[1])) for c in corners]

        # ── OBB outline (green) ────────────────────────────────────
        draw.polygon(poly, outline=(0, 220, 110, 200), width=outline_w)

        # ── Small numbered badge at topmost corner ────────────────────
        top = min(poly, key=lambda p: p[1])
        bx, by = int(top[0]), int(top[1]) - r - 2
        by = max(r, by)     # clamp inside image
        bx = max(r, min(W - r, bx))

        # Filled green circle (semi-transparent for less clutter)
        draw.ellipse([bx - r, by - r, bx + r, by + r],
                     fill=(0, 180, 100, 180), outline=(255, 255, 255, 180),
                     width=1)

        # Number text centred in circle
        num_str = str(i + 1)
        try:
            tw = font.getlength(num_str)
        except AttributeError:
            tw = len(num_str) * (10 // 2)
        try:
            _, top_b, _, bot_b = font.getbbox(num_str)
            th = bot_b - top_b
        except AttributeError:
            th = 10
        draw.text((bx - tw / 2, by - th / 2 - 1), num_str,
                  font=font, fill=(255, 255, 255, 255))

    return img.convert("RGB")


# ─────────────────────────────────────────────────────────────────────────────
# Panel C  — Annotated image + legend sidebar
# ─────────────────────────────────────────────────────────────────────────────

def save_panel_c(annotated_pil: Image.Image, detections: list,
                 out_path: str, image_name: str):
    """
    Left column : annotated page (numbered circle tags on symbols).
    Right column: legend table mapping each number to its full detection info.
    """
    disp_pil, _ = scale_image(annotated_pil, 2800)

    fig = plt.figure(figsize=(22, 20), facecolor="#0d1117")
    gs  = GridSpec(1, 2, figure=fig,
                   width_ratios=[4, 1.1], wspace=0.03)

    # ── Left: annotated image ─────────────────────────────────────────
    ax_img = fig.add_subplot(gs[0, 0])
    ax_img.imshow(np.array(disp_pil))
    ax_img.axis("off")
    ax_img.set_facecolor("#0d1117")

    # ── Right: legend table ──────────────────────────────────────────
    ax_leg = fig.add_subplot(gs[0, 1])
    ax_leg.set_facecolor("#161b22")
    ax_leg.axis("off")

    # Header
    ax_leg.text(0.5, 0.99, "Detection Legend",
                transform=ax_leg.transAxes,
                color="white", fontsize=11, fontweight="bold",
                ha="center", va="top")
    ax_leg.plot([0.05, 0.95], [0.975, 0.975], color="#30363d",
                linewidth=1, transform=ax_leg.transAxes, clip_on=False)

    col_headers = ["  #", "Char", "Class", "Conf", "Angle"]
    col_x       = [0.04, 0.18, 0.35, 0.68, 0.84]
    ax_leg.text(col_x[0], 0.965, "  #",
                transform=ax_leg.transAxes,
                color="#8b949e", fontsize=8, va="top", fontfamily="monospace")
    for cx_frac, hdr in zip(col_x[1:], col_headers[1:]):
        ax_leg.text(cx_frac, 0.965, hdr,
                    transform=ax_leg.transAxes,
                    color="#8b949e", fontsize=8, va="top", fontfamily="monospace")

    row_h   = 0.038      # vertical step per row
    y       = 0.945
    max_rows = int((0.945 - 0.02) / row_h)

    for i, d in enumerate(detections[:max_rows]):
        char = d.get("pred_char",  "?")
        cls  = d.get("pred_class", "?")
        ccnf = d.get("class_confidence", 0)
        ang  = d.get("rotation_degrees", 0)

        clr  = "#3fb950" if ccnf >= 0.9 else ("#d29922" if ccnf >= 0.7 else "#f85149")

        # Alternating row background
        row_bg = "#1c2128" if i % 2 == 0 else "#161b22"
        ax_leg.add_patch(
            plt.Rectangle((0, y - row_h + 0.002), 1, row_h - 0.004,
                           transform=ax_leg.transAxes,
                           color=row_bg, zorder=0)
        )

        # Number badge
        ax_leg.text(col_x[0], y, f"  {i+1}",
                    transform=ax_leg.transAxes,
                    color="#58a6ff", fontsize=8.5, va="top",
                    fontfamily="monospace", fontweight="bold")
        # Char
        ax_leg.text(col_x[1], y, f"{char}",
                    transform=ax_leg.transAxes,
                    color=clr, fontsize=9, va="top", fontweight="bold")
        # Class name (truncate if long)
        cls_short = cls[:10] if len(cls) > 10 else cls
        ax_leg.text(col_x[2], y, cls_short,
                    transform=ax_leg.transAxes,
                    color="#e6edf3", fontsize=7.5, va="top",
                    fontfamily="monospace")
        # Confidence
        ax_leg.text(col_x[3], y, f"{ccnf:.2f}",
                    transform=ax_leg.transAxes,
                    color=clr, fontsize=8, va="top",
                    fontfamily="monospace")
        # Angle
        ax_leg.text(col_x[4], y, f"{ang:+.0f}°",
                    transform=ax_leg.transAxes,
                    color="#8b949e", fontsize=8, va="top",
                    fontfamily="monospace")

        y -= row_h

    if len(detections) > max_rows:
        ax_leg.text(0.5, y, f"... +{len(detections) - max_rows} more",
                    transform=ax_leg.transAxes,
                    color="#8b949e", fontsize=7.5, ha="center", va="top")

    fig.suptitle(
        f"Panel C — {image_name}  │  {len(detections)} detection(s)  │  ≥ 0.70 confidence",
        color="white", fontsize=13, fontweight="bold", y=1.005
    )
    plt.tight_layout(pad=0.8)
    fig.savefig(out_path, dpi=150, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"  [C] Saved: {out_path}")


# ─────────────────────────────────────────────────────────────────────────────
# Main orchestration
# ─────────────────────────────────────────────────────────────────────────────

def run_visualization(image_path: str, yolo_path: str, classifier_path: str,
                      out_dir: str, conf_threshold: float = 0.25):
    os.makedirs(out_dir, exist_ok=True)
    base = os.path.splitext(os.path.basename(image_path))[0]

    print(f"\n{'='*65}")
    print(f"  Input : {image_path}")
    print(f"  YOLO  : {yolo_path}")
    print(f"  CLF   : {classifier_path}")
    print(f"{'='*65}")

    # ── Load models ──────────────────────────────────────────────────────────
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"  Device: {device}")
    yolo = YOLO(yolo_path)
    clf  = SymbolClassifier(model_path=classifier_path, device=device)

    # ── Load & optionally downscale for YOLO ─────────────────────────────────
    raw_pil_full = Image.open(image_path).convert("RGB")
    W_orig, H_orig = raw_pil_full.size
    print(f"  Original size : {W_orig}×{H_orig} px")

    raw_pil, scale = scale_image(raw_pil_full, MAX_PAGE_DIM)
    W, H = raw_pil.size
    if scale < 1.0:
        print(f"  Downscaled to : {W}×{H} px (scale={scale:.3f})")

    # ── Stage 1-3: All preprocessing stages ──────────────────────────────────
    raw_np, gray_np, thresh_np, cleaned_np = preprocess_stages(raw_pil)
    clean_pil = Image.fromarray(cv2.cvtColor(cleaned_np, cv2.COLOR_GRAY2RGB))

    # ── Stage 2: YOLO-OBB detection ───────────────────────────────────────────
    img_bgr = cv2.cvtColor(np.array(clean_pil), cv2.COLOR_RGB2BGR)
    print("  Running YOLO-OBB detection...")
    results = yolo(img_bgr, verbose=False, conf=conf_threshold, imgsz=1280)
    result  = results[0]

    # ── Collect per-detection data ────────────────────────────────────────────
    xywhr_list   = []   # for Panel A
    panel_b_data = []   # for Panel B
    detections   = []   # for Panel C

    if result.obb is not None and len(result.obb) > 0:
        xywhr      = result.obb.xywhr.cpu().numpy()
        xyxyxyxy   = result.obb.xyxyxyxy.cpu().numpy()
        yolo_confs = result.obb.conf.cpu().numpy()
        print(f"  Detections found: {len(xywhr)}")

        for idx in range(len(xywhr)):
            cx, cy, w, h, r = xywhr[idx]
            corners   = xyxyxyxy[idx]
            yolo_conf = float(yolo_confs[idx])
            angle_deg = math.degrees(r)

            xywhr_list.append((cx, cy, w, h, r))

            # ── Rectification ────────────────────────────────────────────────
            rect_crop_pil = rectify_crop(
                clean_pil,
                bbox_metrics={"cx": cx, "cy": cy, "w": w, "h": h,
                               "angle": angle_deg},
                buffer_percent=0.08
            )

            # Raw tilted crop with OBB drawn (on the raw image)
            raw_obb_crop = draw_obb_on_crop(
                get_padded_crop(np.array(raw_pil), cx, cy, w, h, pad_factor=2.0),
                cx, cy, w, h, angle_deg, pad_factor=2.0
            )

            # ── Classifier ───────────────────────────────────────────────────
            class_name, class_conf = clf.predict(rect_crop_pil)
            char_disp = CLASS_TO_CHAR.get(class_name, class_name)

            panel_b_data.append({
                "raw_obb_crop": raw_obb_crop,
                "rect_crop":    rect_crop_pil,
                "angle_deg":    angle_deg,
                "pred_class":   class_name,
                "pred_char":    char_disp,
                "class_conf":   class_conf,
                "yolo_conf":    yolo_conf,
            })

            if class_conf >= 0.7:
                detections.append({
                    "yolo_conf":        yolo_conf,
                    "pred_class":       class_name,
                    "pred_char":        char_disp,
                    "class_confidence": float(class_conf),
                    "rotation_degrees": float(angle_deg),
                    "center":           [float(cx), float(cy)],
                    "size":             [float(w), float(h)],
                    "corners":          corners.tolist(),
                })
    else:
        print("  No detections found.")

    # ── Panel A ───────────────────────────────────────────────────────────────
    panel_a_path = os.path.join(out_dir, f"{base}_panelA_cleanup.png")
    save_panel_a(
        raw_np            = raw_np,
        gray_np           = gray_np,
        thresh_np         = thresh_np,
        cleaned_np        = cleaned_np,
        detections_xywhr  = xywhr_list,
        out_path          = panel_a_path,
        image_name        = base,
    )

    # ── Panel B ───────────────────────────────────────────────────────────────
    panel_b_path = os.path.join(out_dir, f"{base}_panelB_orientation.png")
    save_panel_b(panel_b_data, panel_b_path, base)

    # ── Panel C ───────────────────────────────────────────────────────────────
    annotated_pil = annotate_page(raw_pil, detections)
    panel_c_path  = os.path.join(out_dir, f"{base}_panelC_annotated.png")
    save_panel_c(annotated_pil, detections, panel_c_path, base)

    # Save the thin-bordered full page directly
    thin_page_path = os.path.join(out_dir, f"{base}_thin_annotated.png")
    annotated_pil.save(thin_page_path)
    print(f"  [T] Saved thin page: {thin_page_path}")

    # ── JSON detections report ─────────────────────────────────────────────
    import json
    json_report = {
        "image":      base,
        "total":      len(detections),
        "detections": [
            {
                "id":               i + 1,
                "character":        d["pred_char"],
                "class":            d["pred_class"],
                "class_confidence": round(d["class_confidence"], 4),
                "yolo_confidence":  round(d["yolo_conf"], 4),
                "rotation_deg":     round(d["rotation_degrees"], 2),
                "center_xy":        [round(v, 1) for v in d["center"]],
                "size_wh":          [round(v, 1) for v in d["size"]],
            }
            for i, d in enumerate(detections)
        ]
    }
    json_path = os.path.join(out_dir, f"{base}_detections.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(json_report, f, indent=2, ensure_ascii=False)
    print(f"  [J] Saved: {json_path}")

    print(f"  ✓ {len(detections)} detections (≥0.70 confidence)")
    print(f"  ✓ Panels saved to: {out_dir}/\n")


def main():
    parser = argparse.ArgumentParser(description="Pipeline Visualization Tool")
    parser.add_argument("--input",      required=True)
    parser.add_argument("--yolo",       default=DEFAULT_YOLO)
    parser.add_argument("--classifier", default=DEFAULT_CLASSIFIER)
    parser.add_argument("--out",        default=DEFAULT_OUT)
    parser.add_argument("--conf",       type=float, default=0.25)
    args = parser.parse_args()

    exts = {".png", ".jpg", ".jpeg", ".bmp", ".tiff", ".tif"}
    if os.path.isdir(args.input):
        images = sorted([
            os.path.join(args.input, f)
            for f in os.listdir(args.input)
            if os.path.splitext(f)[1].lower() in exts
        ])
        if not images:
            print(f"No images found in: {args.input}")
            return
        for img_path in images:
            run_visualization(img_path, args.yolo, args.classifier,
                               args.out, args.conf)
    elif os.path.isfile(args.input):
        run_visualization(args.input, args.yolo, args.classifier,
                           args.out, args.conf)
    else:
        print(f"ERROR: not found: {args.input}")


if __name__ == "__main__":
    main()
