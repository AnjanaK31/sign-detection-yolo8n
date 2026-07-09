"""
gen_paddle_rec_dataset.py
--------------------------
Generates a PaddleOCR text-recognition fine-tuning dataset for engineering
drawing symbols. Produces:

  paddle_rec_dataset/
    images/
      train/   ← 270 crops per symbol
      val/     ← 30 crops per symbol
    train.txt  ← tab-separated: images/train/xxx.png\t<label>
    val.txt

Each image is a tight text-line crop (~32px tall) rendered to look like
it came from a scanned engineering blueprint: white background, black text,
random font size, slight rotation jitter, scanning noise, and optional
background grid/line clutter.

Usage
-----
  # Generate with defaults (300 per symbol, train/val 90/10 split):
  python gen_paddle_rec_dataset.py

  # Custom count or output dir:
  python gen_paddle_rec_dataset.py --count 500 --out paddle_rec_dataset
"""

import os
import sys
import random
import math
import argparse
import numpy as np
import cv2
from PIL import Image, ImageDraw, ImageFont

# Force UTF-8 stdout on Windows so Unicode symbols (μ, ±, Ø, ×, °) print cleanly
if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")


# ---------------------------------------------------------------------------
# TARGET SYMBOLS  (from the engineering drawing character table)
# ---------------------------------------------------------------------------
# Each entry: (label_string, display_name)
# The label_string is EXACTLY what will appear in train.txt / val.txt
# and what the model will be trained to predict.

SYMBOLS = [
    ("±",   "plus_minus"),
    ("Ø",   "diameter"),
    ("R",   "radius"),
    ("°",   "degree"),
    (">",   "greater_than"),
    ("<",   "less_than"),
    (">=",  "gte"),
    ('"',   "inch"),
    ("μm",  "micrometre"),
    ("M",   "metric_thread_M"),
    ("x",   "thread_pitch_sep"),
    ("-",   "thread_tol_sep"),
    ("+",   "plus_unilateral"),
    ("×",   "dims_sep"),
]

# ---------------------------------------------------------------------------
# CONTEXT TEMPLATES
# The model learns better when the symbol appears in context (short strings)
# rather than in isolation.  Each template function returns a string whose
# label is still the target symbol — we render the full string but we only
# store the symbol as the label.
# Actually for a recognition model we store the FULL rendered string as the
# label so the model learns from context.  This mirrors how PaddleOCR is
# fine-tuned on real data.
# ---------------------------------------------------------------------------

def make_context_templates():
    """Returns a dict: label_char -> list of callables that produce a sample string."""
    digits = lambda: str(random.randint(1, 999))
    dec    = lambda: f"{random.randint(1,99)}.{random.randint(0,9)}"

    return {
        "±": [
            lambda: f"±{dec()}",
            lambda: f"{dec()} ± {dec()}",
            lambda: f"Ø{dec()} ± 0.{random.randint(1,9)}",
            lambda: f"±{random.choice(['0.01','0.02','0.05','0.10','0.20','0.50'])}",
            lambda: f"{digits()} ± {random.choice(['0.5','1','2'])}",
        ],
        "Ø": [
            lambda: f"Ø{dec()}",
            lambda: f"Ø{digits()}",
            lambda: f"Ø{dec()} mm",
            lambda: f"Ø{dec()} ± 0.{random.randint(1,5)}",
            lambda: f"Ø{random.randint(5,300)}.{random.randint(0,9)}mm",
        ],
        "R": [
            lambda: f"R{dec()}",
            lambda: f"R{digits()}",
            lambda: f"R{dec()} mm",
            lambda: f"R{random.randint(2,100)}",
            lambda: f"R{dec()} ± {random.choice(['0.1','0.2','0.5'])}",
        ],
        "°": [
            lambda: f"{random.randint(1,360)}°",
            lambda: f"{random.choice([15,30,45,60,90,120,180])}°",
            lambda: f"{dec()}°",
            lambda: f"{random.randint(1,90)}° ± 0.{random.randint(1,9)}°",
            lambda: f"{random.choice([18,36,45,90])}°",
        ],
        ">": [
            lambda: f">{dec()}",
            lambda: f"> {random.choice(['40%','>98%','1.33','0.05'])}",
            lambda: f">{random.randint(1,100)}",
            lambda: f"> {dec()} mm",
            lambda: f">{random.choice(['25','50','75','100'])} PPM",
        ],
        "<": [
            lambda: f"<{dec()}",
            lambda: f"< {random.choice(['25 PPM','30dB','500 μm'])}",
            lambda: f"<{random.randint(1,100)}",
            lambda: f"< {dec()} mm",
            lambda: f"<{random.choice(['0.1','0.5','1.0','5.0'])}",
        ],
        ">=": [
            lambda: f">={dec()}",
            lambda: f">= {dec()}",
            lambda: f">= {random.choice(['1.67','2.0','1.0','0.5'])}",
            lambda: f">={random.randint(1,100)}",
            lambda: f">= {random.randint(10,99)}%",
        ],
        '"': [
            lambda: f'{random.randint(1,24)}"',
            lambda: f'{dec()}"',
            lambda: f'{random.randint(1,12)}" OD',
            lambda: f'{dec()}" ID',
            lambda: f'{random.randint(1,10)}.{random.randint(0,9)}"',
        ],
        "μm": [
            lambda: f"{random.randint(1,999)} μm",
            lambda: f"{dec()} μm",
            lambda: f"Ra {random.choice(['0.8','1.6','3.2','6.3'])} μm",
            lambda: f"Rz {random.choice(['1.6','3.2','6.3','12.5'])} μm",
            lambda: f"< {random.randint(100,999)} μm",
        ],
        "M": [
            lambda: f"M{random.choice([3,4,5,6,8,10,12,16,20,24])}",
            lambda: f"M{random.choice([6,8,10,12])}x{random.choice(['1.0','1.25','1.5','2.0'])}",
            lambda: f"M{random.choice([8,10,12,16])}x{random.choice(['1.25','1.5','2.0'])}-6H",
            lambda: f"M{random.choice([10,12,16])}x{random.choice(['1.5','2.0'])}-6g",
            lambda: f"M{random.choice([4,5,6,8])} JASO",
        ],
        "x": [
            lambda: f"M{random.choice([6,8,10,12])}x{random.choice(['1.0','1.25','1.5'])}",
            lambda: f"{dec()} x {dec()}",
            lambda: f"{random.randint(2,20)} x {random.randint(2,50)}",
            lambda: f"M{random.choice([10,12])}x{random.choice(['1.5','2.0'])}-6H",
            lambda: f"{dec()}" + ' x ' + f"{dec()}" + ' mm',
        ],
        "-": [
            lambda: f"M{random.choice([8,10,12])}x{random.choice(['1.25','1.5'])}-{random.choice(['6H','6g'])}",
            lambda: f"{random.randint(1,99)}-{random.randint(1,99)}",
            lambda: f"DIN {random.randint(100,999)}-{random.randint(1,9)}",
            lambda: f"ISO {random.randint(1,99)}-{random.randint(1,9)}",
            lambda: f"M{random.choice([10,12,16])}x{random.choice(['1.5','2.0'])}-{random.choice(['6H','6g','4H'])}",
        ],
        "+": [
            lambda: f"+{random.choice(['0.01','0.02','0.05','0.10','0.20'])}",
            lambda: f"Ø{dec()}+{random.choice(['0.05','0.1'])} mm",
            lambda: f"+{dec()}",
            lambda: f"+{random.randint(1,9)}.{random.randint(0,9)}",
            lambda: f"Ø{random.randint(5,50)}.{random.randint(0,9)}+0.{random.randint(1,9)} mm",
        ],
        "×": [
            lambda: f"{random.randint(10,500)} × {random.randint(10,500)} mm",
            lambda: f"{dec()} × {dec()}",
            lambda: f"{random.randint(1,20)} × {random.randint(1,20)}",
            lambda: f"{random.randint(10,200)} × {random.randint(10,200)} × {random.randint(1,50)} mm",
            lambda: f"80 × 60 mm",
        ],
    }


CONTEXT_TEMPLATES = make_context_templates()


# ---------------------------------------------------------------------------
# FONT DISCOVERY
# ---------------------------------------------------------------------------

# Fonts that support Unicode symbols (μ, ±, Ø, ×, °, etc.)
FONT_CANDIDATES = [
    "C:\\Windows\\Fonts\\seguisym.ttf",     # Segoe UI Symbol — best Unicode support
    "C:\\Windows\\Fonts\\calibri.ttf",
    "C:\\Windows\\Fonts\\arial.ttf",
    "C:\\Windows\\Fonts\\segoeui.ttf",
    "C:\\Windows\\Fonts\\tahoma.ttf",
    "C:\\Windows\\Fonts\\cour.ttf",         # Courier New (monospace, good for engineering)
    "C:\\Windows\\Fonts\\times.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",
    "/usr/share/fonts/TTF/DejaVuSans.ttf",
]

def find_fonts():
    """Returns all available font paths that exist on this system."""
    found = [p for p in FONT_CANDIDATES if os.path.exists(p)]
    if not found:
        # Fallback: download DejaVuSans
        local = "DejaVuSans.ttf"
        if os.path.exists(local):
            found = [local]
        else:
            print("[WARN] No system font found. Downloading DejaVuSans.ttf for Unicode symbol support...")
            try:
                import urllib.request
                url = "https://raw.githubusercontent.com/dejavu-fonts/dejavu-fonts/master/resources/DejaVuSans.ttf"
                req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
                with urllib.request.urlopen(req) as r, open(local, "wb") as f:
                    f.write(r.read())
                found = [local]
                print("[OK] Downloaded DejaVuSans.ttf")
            except Exception as e:
                print(f"[FAIL] Font download failed: {e}. Symbol glyphs may not render correctly.")
    return found


# ---------------------------------------------------------------------------
# IMAGE AUGMENTATION HELPERS
# ---------------------------------------------------------------------------

def add_background_clutter(img_pil, strength=0.4):
    """Randomly adds faint grid lines or a single diagonal line to simulate
    engineering drawing background clutter behind the text."""
    if random.random() > strength:
        return img_pil

    draw = ImageDraw.Draw(img_pil)
    w, h = img_pil.size
    gray = random.randint(180, 230)
    line_color = (gray, gray, gray)

    choice = random.random()
    if choice < 0.4:
        # Single horizontal or vertical line crossing the crop
        if random.random() < 0.5:
            y = random.randint(h // 4, 3 * h // 4)
            draw.line([(0, y), (w, y)], fill=line_color, width=1)
        else:
            x = random.randint(w // 4, 3 * w // 4)
            draw.line([(x, 0), (x, h)], fill=line_color, width=1)
    elif choice < 0.7:
        # Diagonal crossing line
        x1, y1 = random.randint(0, w // 3), random.randint(0, h)
        x2, y2 = random.randint(2 * w // 3, w), random.randint(0, h)
        draw.line([(x1, y1), (x2, y2)], fill=line_color, width=1)
    else:
        # Faint grid
        spacing = random.randint(8, 16)
        for gx in range(0, w, spacing):
            draw.line([(gx, 0), (gx, h)], fill=(220, 220, 220), width=1)
        for gy in range(0, h, spacing):
            draw.line([(0, gy), (w, gy)], fill=(220, 220, 220), width=1)

    return img_pil


def apply_scan_noise(img_pil, noise_std=8):
    """Adds subtle Gaussian pixel noise (simulates scanner grain)."""
    arr = np.array(img_pil).astype(np.float32)
    noise = np.random.normal(0, noise_std, arr.shape).astype(np.float32)
    arr = np.clip(arr + noise, 0, 255).astype(np.uint8)
    return Image.fromarray(arr)


def apply_morphology(img_pil):
    """Randomly thickens or thins strokes via erode/dilate — mimics scan variation."""
    arr = np.array(img_pil.convert("L"))
    kernel = np.ones((2, 2), np.uint8)
    r = random.random()
    if r < 0.25:
        arr = cv2.erode(arr, kernel, iterations=1)   # thickens text
    elif r < 0.50:
        arr = cv2.dilate(arr, kernel, iterations=1)  # thins text
    return Image.fromarray(cv2.cvtColor(arr, cv2.COLOR_GRAY2RGB))


def apply_slight_blur(img_pil):
    """Applies very mild Gaussian blur to simulate soft scan focus."""
    arr = np.array(img_pil)
    sigma = random.uniform(0.3, 0.8)
    blurred = cv2.GaussianBlur(arr, (3, 3), sigma)
    return Image.fromarray(blurred)


def apply_low_quality_degradation(img_pil):
    """
    Heavy degradation stack that mimics the worst real scanned engineering
    drawing crops (like 1_crop_0.png):
      - Grey / uneven background (not clean white)
      - Faded, thin ink (low contrast text)
      - JPEG compression artifacts
      - Heavy Gaussian scan noise
      - Salt-and-pepper speckle
      - Ink smear via motion blur kernel
      - Optional downscale-upscale to add blockiness
    """
    arr = np.array(img_pil).astype(np.float32)

    # 1. Uneven grey background — replace white with a dirty grey gradient
    bg_level = random.randint(200, 240)   # base grey (not pure white)
    bg_noise  = np.random.normal(0, random.uniform(4, 14), arr.shape).astype(np.float32)
    # Create a slightly uneven background by darkening white areas
    white_mask = (arr > 200).astype(np.float32)   # pixels that were background
    arr = arr * (1 - white_mask) + (bg_level + bg_noise) * white_mask

    # 2. Fade the ink (reduce contrast) — text becomes lighter/greyer
    fade_factor = random.uniform(0.35, 0.70)  # 1.0 = pure black, 0.0 = invisible
    text_mask   = (arr < 150).astype(np.float32)
    ink_level   = random.randint(50, 160)     # target grey for text strokes
    arr = arr * (1 - text_mask) + (ink_level + np.random.normal(0, 8, arr.shape)) * text_mask * fade_factor \
          + arr * text_mask * (1 - fade_factor)

    arr = np.clip(arr, 0, 255).astype(np.uint8)
    img_pil = Image.fromarray(arr)

    # 3. JPEG compression artifacts (encode at low quality → decode back)
    import io
    quality = random.randint(15, 45)   # 15 = very blocky, 45 = mildly compressed
    buf = io.BytesIO()
    img_pil.save(buf, format="JPEG", quality=quality)
    buf.seek(0)
    img_pil = Image.open(buf).copy()
    arr = np.array(img_pil).astype(np.float32)

    # 4. Heavy Gaussian scan noise
    noise_std = random.uniform(10, 28)
    arr = np.clip(arr + np.random.normal(0, noise_std, arr.shape), 0, 255)

    # 5. Salt-and-pepper speckle
    speck_prob = random.uniform(0.01, 0.06)
    mask_salt   = np.random.random(arr.shape[:2]) < speck_prob / 2
    mask_pepper = np.random.random(arr.shape[:2]) < speck_prob / 2
    arr[mask_salt]   = 240
    arr[mask_pepper] = 20

    # 6. Ink smear via random motion-blur kernel
    arr = arr.astype(np.uint8)
    direction = random.choice(["h", "v", "d"])
    k = random.randint(2, 5)
    if direction == "h":
        kernel = np.zeros((1, k), np.float32)
        kernel[0, :] = 1.0 / k
    elif direction == "v":
        kernel = np.zeros((k, 1), np.float32)
        kernel[:, 0] = 1.0 / k
    else:
        kernel = np.eye(k, dtype=np.float32) / k
    arr = cv2.filter2D(arr, -1, kernel)

    # 7. Optional downscale-then-upscale blockiness
    if random.random() < 0.4:
        h, w = arr.shape[:2]
        scale_down = random.uniform(0.4, 0.7)
        small = cv2.resize(arr, (max(1, int(w * scale_down)), max(1, int(h * scale_down))),
                           interpolation=cv2.INTER_LINEAR)
        arr = cv2.resize(small, (w, h), interpolation=cv2.INTER_LINEAR)

    return Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8))


# ---------------------------------------------------------------------------
# CORE CROP GENERATOR
# ---------------------------------------------------------------------------

class RecCropGenerator:
    def __init__(self, font_paths):
        self.font_paths = font_paths
        print(f"Using {len(font_paths)} font(s): {', '.join(os.path.basename(p) for p in font_paths)}")

    def get_font(self, size, font_path=None):
        fp = font_path or random.choice(self.font_paths)
        try:
            return ImageFont.truetype(fp, size), fp
        except Exception:
            return ImageFont.load_default(), fp

    def render_crop(self, text, target_height=48, low_quality=False):
        """
        Renders `text` as a tight crop image ready for PaddleOCR recognition.

        Parameters
        ----------
        low_quality : bool
            If True, applies heavy degradation (grey bg, faded ink, JPEG
            artifacts, smear) to mimic poor real-world scanned crops.
        """
        # --- Font & size ---
        font_size = random.randint(int(target_height * 0.7), int(target_height * 1.4))
        font, font_path = self.get_font(font_size)

        # --- Measure text ---
        try:
            left, top, right, bottom = font.getbbox(text)
            tw = right - left
            th = bottom - top
        except AttributeError:
            tw = len(text) * (font_size // 2)
            th = font_size

        if tw <= 0 or th <= 0:
            tw = max(tw, font_size)
            th = max(th, font_size)

        # --- Canvas: add horizontal padding ---
        pad_x = random.randint(6, 20)
        pad_y = random.randint(4, 12)
        canvas_w = tw + pad_x * 2
        canvas_h = th + pad_y * 2

        img = Image.new("RGB", (canvas_w, canvas_h), (255, 255, 255))

        # --- Background clutter (before text) ---
        img = add_background_clutter(img, strength=0.45)

        draw = ImageDraw.Draw(img)

        # --- Text position jitter ---
        tx = pad_x - left + random.randint(-2, 2)
        ty = pad_y - top  + random.randint(-2, 2)
        draw.text((tx, ty), text, font=font, fill=(0, 0, 0))

        # --- Standard augmentations ---
        angle = random.uniform(-5.0, 5.0)
        if abs(angle) > 0.5:
            img = img.rotate(angle, expand=False, resample=Image.BICUBIC,
                             fillcolor=(255, 255, 255))

        if not low_quality:
            # Standard quality path
            if random.random() < 0.6:
                img = apply_scan_noise(img, noise_std=random.uniform(3, 12))
            if random.random() < 0.4:
                img = apply_morphology(img)
            if random.random() < 0.3:
                img = apply_slight_blur(img)
        else:
            # Low quality path — apply heavy degradation stack
            img = apply_low_quality_degradation(img)

        # --- Resize to fixed height preserving aspect ratio ---
        orig_w, orig_h = img.size
        scale = target_height / orig_h
        new_w = max(1, int(orig_w * scale))
        img = img.resize((new_w, target_height), Image.LANCZOS)

        return img


# ---------------------------------------------------------------------------
# DATASET BUILDER
# ---------------------------------------------------------------------------

def generate_dataset(out_dir="paddle_rec_dataset",
                     count_per_symbol=300,
                     val_fraction=0.10,
                     target_height=48):
    """
    Generates the full fine-tuning dataset.

    Parameters
    ----------
    out_dir           : root output folder
    count_per_symbol  : total images per symbol (train + val combined)
    val_fraction      : fraction held out for val (default 10%)
    target_height     : height in pixels of each output crop image
    """
    font_paths = find_fonts()
    if not font_paths:
        raise RuntimeError("No usable fonts found — cannot render Unicode symbols.")

    gen = RecCropGenerator(font_paths)

    train_img_dir = os.path.join(out_dir, "images", "train")
    val_img_dir   = os.path.join(out_dir, "images", "val")
    os.makedirs(train_img_dir, exist_ok=True)
    os.makedirs(val_img_dir,   exist_ok=True)

    train_lines = []
    val_lines   = []

    val_count   = max(1, int(round(count_per_symbol * val_fraction)))
    train_count = count_per_symbol - val_count

    print(f"\nGenerating {count_per_symbol} crops x {len(SYMBOLS)} symbols "
          f"-> {count_per_symbol * len(SYMBOLS)} total images")
    print(f"  Train: {train_count}/symbol   Val: {val_count}/symbol\n")

    templates = CONTEXT_TEMPLATES

    for label_char, sym_name in SYMBOLS:
        sym_templates = templates.get(label_char, [lambda: label_char])
        print(f"  [{sym_name:20s}]  label='{label_char}'  …", end="", flush=True)

        for split, n_imgs, img_dir, lines in [
            ("train", train_count, train_img_dir, train_lines),
            ("val",   val_count,   val_img_dir,   val_lines),
        ]:
            for i in range(n_imgs):
                # Pick a context template and generate the rendered string
                tmpl_fn = random.choice(sym_templates)
                text = tmpl_fn()

                # 30% of train images use heavy degradation to match real scan quality
                # Val images always use standard quality for consistent evaluation
                use_low_quality = (split == "train") and (random.random() < 0.30)

                try:
                    crop = gen.render_crop(text, target_height=target_height,
                                           low_quality=use_low_quality)
                except Exception as e:
                    print(f"    [WARN] Render failed for '{text}': {e}. Skipping.")
                    continue

                # Filename: <sym_name>_<split>_<i>.png
                fname = f"{sym_name}_{i:04d}.png"
                fpath = os.path.join(img_dir, fname)
                crop.save(fpath)

                rel_path = f"images/{split}/{fname}"
                lines.append(f"{rel_path}\t{text}")

        print(f" OK {train_count + val_count} images")

    # Write label files
    train_txt = os.path.join(out_dir, "train.txt")
    val_txt   = os.path.join(out_dir, "val.txt")

    with open(train_txt, "w", encoding="utf-8") as f:
        f.write("\n".join(train_lines) + "\n")

    with open(val_txt, "w", encoding="utf-8") as f:
        f.write("\n".join(val_lines) + "\n")

    # Write character dictionary
    char_set = set()
    for label_char, _ in SYMBOLS:
        for ch in label_char:
            char_set.add(ch)
    # Add all characters that appear in any rendered string
    for line in train_lines + val_lines:
        text = line.split("\t", 1)[1] if "\t" in line else ""
        for ch in text:
            char_set.add(ch)

    # PaddleOCR dict: space first, then sorted unique chars
    char_list = [" "] + sorted(char_set - {" "})
    dict_path = os.path.join(out_dir, "my_chars.txt")
    with open(dict_path, "w", encoding="utf-8") as f:
        f.write("\n".join(char_list) + "\n")

    print(f"\n{'='*60}")
    print(f"[OK]  Dataset saved to: {os.path.abspath(out_dir)}")
    print(f"    Train images : {len(train_lines)}")
    print(f"    Val images   : {len(val_lines)}")
    print(f"    train.txt    : {train_txt}")
    print(f"    val.txt      : {val_txt}")
    print(f"    my_chars.txt : {dict_path}  ({len(char_list)} unique characters)")
    print(f"{'='*60}\n")
    print("Next steps:")
    print("  1. Update your rec_finetune.yaml:")
    print(f"       data_dir: {os.path.abspath(out_dir)}")
    print(f"       label_file_list: [\"{os.path.abspath(train_txt)}\"]")
    print(f"       character_dict_path: {os.path.abspath(dict_path)}")
    print("  2. Run training:")
    print("       ..\\PaddleOCR\\venv\\Scripts\\python.exe tools/train.py -c rec_finetune.yaml")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate PaddleOCR recognition fine-tuning dataset for engineering symbols"
    )
    parser.add_argument("--out",    "-o", default="paddle_rec_dataset",
                        help="Output directory (default: paddle_rec_dataset)")
    parser.add_argument("--count",  "-c", type=int, default=300,
                        help="Number of images per symbol (default: 300)")
    parser.add_argument("--height", "-H", type=int, default=48,
                        help="Crop image height in pixels (default: 48)")
    parser.add_argument("--val-frac", type=float, default=0.10,
                        help="Fraction of images used for validation (default: 0.10)")
    args = parser.parse_args()

    generate_dataset(
        out_dir=args.out,
        count_per_symbol=args.count,
        val_fraction=args.val_frac,
        target_height=args.height,
    )
