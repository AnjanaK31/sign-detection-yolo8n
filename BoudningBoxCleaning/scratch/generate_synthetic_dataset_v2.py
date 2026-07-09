import cv2
import math
import numpy as np
import os
import random
from PIL import Image, ImageDraw, ImageFont, ImageOps

# Reconfigure stdout for Windows unicode support
import sys
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

# List of template expressions mimicking real mechanical drawings
expressions_templates = [
    "{num}",
    "{num} ± {tol}",
    "{num} +{tol_plus}\n-{tol_minus}",
    "Ø {num}",
    "Ø {num} HOLE",
    "Ø {num} {num_holes} HOLES",
    "R{num_r}",
    "R{num_r} ± {tol}",
    "M{num_m} x {pitch}",
    "M{num_m} x {pitch} - {class_fit}",
    "Ra {ra} \u03bcm",
    "Ra {ra} MAX",
    "{num} MIN",
    "{num} MAX",
    "({num} TYP)",
    "SECTION {char}-{char}",
    "DETAIL {char}"
]

def generate_random_expression():
    num = f"{random.uniform(1.0, 500.0):.2f}" if random.random() > 0.3 else f"{random.randint(1, 500)}"
    tol = f"{random.uniform(0.01, 2.0):.2f}"
    tol_plus = f"{random.uniform(0.01, 1.0):.2f}"
    tol_minus = f"{random.uniform(0.01, 1.0):.2f}"
    num_holes = str(random.randint(2, 12))
    num_r = f"{random.uniform(1.0, 100.0):.1f}" if random.random() > 0.5 else f"{random.randint(1, 50)}"
    num_m = str(random.choice([3, 4, 5, 6, 8, 10, 12, 16, 20, 24]))
    pitch = str(random.choice([0.5, 0.7, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0, 3.0]))
    class_fit = random.choice(["6H", "6g", "7H", "8g"])
    ra = random.choice(["0.8", "1.6", "3.2", "6.3", "12.5", "25"])
    char = random.choice("ABCDEFGHJKLMNOPRSTUYZ")
    
    template = random.choice(expressions_templates)
    return template.format(
        num=num, tol=tol, tol_plus=tol_plus, tol_minus=tol_minus,
        num_holes=num_holes, num_r=num_r, num_m=num_m, pitch=pitch,
        class_fit=class_fit, ra=ra, char=char
    )

def get_font(font_size):
    font_paths = [
        "C:/Windows/Fonts/arial.ttf",
        "C:/Windows/Fonts/cour.ttf",
        "C:/Windows/Fonts/consola.ttf"
    ]
    for fp in font_paths:
        if os.path.exists(fp):
            try:
                return ImageFont.truetype(fp, font_size)
            except:
                pass
    return ImageFont.load_default()

def draw_arrowhead(draw, p1, p2, size=6, angle=20):
    """Draws an arrowhead at p2 pointing from p1."""
    x1, y1 = p1
    x2, y2 = p2
    theta = math.atan2(y2 - y1, x2 - x1)
    
    alpha1 = theta + math.pi - math.radians(angle)
    alpha2 = theta + math.pi + math.radians(angle)
    
    ap1 = (x2 + size * math.cos(alpha1), y2 + size * math.sin(alpha1))
    ap2 = (x2 + size * math.cos(alpha2), y2 + size * math.sin(alpha2))
    
    draw.polygon([p2, ap1, ap2], fill=0)

def draw_dashed_line(draw, p1, p2, dash_length=4, gap_length=4, width=1):
    x1, y1 = p1
    x2, y2 = p2
    dist = math.hypot(x2 - x1, y2 - y1)
    if dist == 0:
        return
        
    dx = (x2 - x1) / dist
    dy = (y2 - y1) / dist
    
    curr = 0
    while curr < dist:
        end = min(curr + dash_length, dist)
        draw.line([
            (x1 + curr * dx, y1 + curr * dy),
            (x1 + end * dx, y1 + end * dy)
        ], fill=0, width=width)
        curr += dash_length + gap_length

def generate_pair_v2(index, output_dir):
    # 1. Create a large clean canvas
    canvas_size = 400
    clean_canvas = Image.new("L", (canvas_size, canvas_size), 255)
    draw_clean = ImageDraw.Draw(clean_canvas)
    
    # Random text expression
    text = generate_random_expression()
    font_size = random.randint(18, 28)
    font = get_font(font_size)
    
    # Calculate text bounding box to center it
    bbox = draw_clean.textbbox((0, 0), text, font=font)
    text_w = bbox[2] - bbox[0]
    text_h = bbox[3] - bbox[1]
    
    tx = (canvas_size - text_w) // 2
    ty = (canvas_size - text_h) // 2
    
    # Draw clean text in the middle
    draw_clean.text((tx, ty), text, font=font, fill=0)
    
    # Get actual text bounding box in canvas coordinates
    text_left = tx + bbox[0]
    text_top = ty + bbox[1]
    text_right = tx + bbox[2]
    text_bottom = ty + bbox[3]
    
    # Crop with random buffer mimicking standard OBB crop (e.g. 15% to 25%)
    buffer_percent = random.uniform(0.15, 0.25)
    buf_w = int(text_w * buffer_percent)
    buf_h = int(text_h * buffer_percent)
    
    crop_left = max(0, text_left - buf_w)
    crop_top = max(0, text_top - buf_h)
    crop_right = min(canvas_size, text_right + buf_w)
    crop_bottom = min(canvas_size, text_bottom + buf_h)
    
    crop_w = crop_right - crop_left
    crop_h = crop_bottom - crop_top
    
    # Target crop (perfectly clean text)
    target_crop = clean_canvas.crop((crop_left, crop_top, crop_right, crop_bottom))
    
    # 2. Create the input crop (with lines overlaid)
    input_crop = target_crop.copy()
    draw_input = ImageDraw.Draw(input_crop)
    
    # Determine text box coordinates in cropped space
    # (these are horizontal axes coordinates)
    t_left = text_left - crop_left
    t_top = text_top - crop_top
    t_right = text_right - crop_left
    t_bottom = text_bottom - crop_top
    
    line_width = random.choice([1, 2])
    
    # ── Draw Realistic CAD lines (aligned with text orientation) ──
    # Select CAD layout type
    layout_type = random.choice(["parallel_below", "parallel_above", "crossing_horizontal", "parallel_double", "perpendicular_only"])
    
    if layout_type == "parallel_below":
        # Draw a horizontal line parallel to text, located just below it
        y_line = t_bottom + random.randint(2, 5)
        if y_line < crop_h:
            draw_input.line([(0, y_line), (crop_w - 1, y_line)], fill=0, width=line_width)
            # Add an arrowhead on the leader end (e.g. on the left or right side)
            if random.random() > 0.4:
                arrow_side = random.choice(["left", "right"])
                if arrow_side == "left":
                    draw_arrowhead(draw_input, (crop_w // 2, y_line), (0, y_line))
                else:
                    draw_arrowhead(draw_input, (crop_w // 2, y_line), (crop_w - 1, y_line))
                    
    elif layout_type == "parallel_above":
        # Draw a horizontal line parallel to text, located just above it
        y_line = t_top - random.randint(2, 5)
        if y_line >= 0:
            draw_input.line([(0, y_line), (crop_w - 1, y_line)], fill=0, width=line_width)
            if random.random() > 0.4:
                arrow_side = random.choice(["left", "right"])
                if arrow_side == "left":
                    draw_arrowhead(draw_input, (crop_w // 2, y_line), (0, y_line))
                else:
                    draw_arrowhead(draw_input, (crop_w // 2, y_line), (crop_w - 1, y_line))
                    
    elif layout_type == "crossing_horizontal":
        # Draw a horizontal line that crosses right through the middle of the text
        y_line = (t_top + t_bottom) // 2
        draw_input.line([(0, y_line), (crop_w - 1, y_line)], fill=0, width=line_width)
        
    elif layout_type == "parallel_double":
        # Draw two parallel lines, one above and one below (like hatching or bounding lines)
        y_above = t_top - random.randint(2, 6)
        y_below = t_bottom + random.randint(2, 6)
        if y_above >= 0:
            draw_input.line([(0, y_above), (crop_w - 1, y_above)], fill=0, width=1)
        if y_below < crop_h:
            draw_input.line([(0, y_below), (crop_w - 1, y_below)], fill=0, width=1)
            
    # Add a crossing perpendicular/extension line in 70% of cases (very common in 2.jpeg)
    if random.random() > 0.3:
        # Perpendicular line (extension line crossing vertically)
        x_line = random.choice([
            random.randint(0, max(5, t_left - 3)),           # Left of text
            random.randint(min(crop_w - 5, t_right + 3), crop_w - 1)  # Right of text
        ])
        # Sometimes draw it slanted (e.g. 75 to 90 degrees)
        angle_rad = math.radians(random.uniform(75, 105))
        h_half = crop_h // 2
        dx_val = int(h_half / math.tan(angle_rad))
        p1 = (x_line - dx_val, 0)
        p2 = (x_line + dx_val, crop_h - 1)
        draw_input.line([p1, p2], fill=0, width=1)

    # Rotate both images by the SAME small angle to simulate tilted OBB crops (rectification error)
    # Typically YOLO OBB alignments are off by -4 to +4 degrees.
    angle = random.uniform(-4, 4)
    pad_val = 15
    padded_input = ImageOps.expand(input_crop, border=pad_val, fill=255)
    padded_target = ImageOps.expand(target_crop, border=pad_val, fill=255)
    
    rot_input = padded_input.rotate(angle, resample=Image.BICUBIC, fillcolor=255)
    rot_target = padded_target.rotate(angle, resample=Image.BICUBIC, fillcolor=255)
    
    final_input = rot_input.crop((pad_val, pad_val, pad_val + crop_w, pad_val + crop_h))
    final_target = rot_target.crop((pad_val, pad_val, pad_val + crop_w, pad_val + crop_h))
    
    # Convert to numpy for thresholding and noise
    np_in = np.array(final_input)
    np_tg = np.array(final_target)
    
    # Add scanner noise to input
    if random.random() > 0.6:
        np_in = cv2.GaussianBlur(np_in, (3, 3), 0)
    if random.random() > 0.8:
        noise = np.random.randint(0, 255, np_in.shape, dtype=np_in.dtype)
        np_in = np.where(noise < 3, 0, np_in)
        np_in = np.where(noise > 252, 255, np_in)
        
    _, np_in = cv2.threshold(np_in, 180, 255, cv2.THRESH_BINARY)
    _, np_tg = cv2.threshold(np_tg, 180, 255, cv2.THRESH_BINARY)
    
    # Save outputs
    input_save_path = os.path.join(output_dir, "inputs", f"pair_{index:06d}.png")
    target_save_path = os.path.join(output_dir, "targets", f"pair_{index:06d}.png")
    
    cv2.imwrite(input_save_path, np_in)
    cv2.imwrite(target_save_path, np_tg)

def generate_dataset_v2(num_pairs=50, output_dir="d:/Internship/OCR_PDF/BoudningBoxCleaning/synthetic_dataset_v2"):
    print(f"Creating directories in: {output_dir}")
    os.makedirs(os.path.join(output_dir, "inputs"), exist_ok=True)
    os.makedirs(os.path.join(output_dir, "targets"), exist_ok=True)
    
    print(f"Generating {num_pairs} training pairs matching 2.jpeg styles...")
    for idx in range(num_pairs):
        generate_pair_v2(idx, output_dir)
        if (idx + 1) % 10 == 0 or idx == num_pairs - 1:
            print(f"  Generated {idx + 1}/{num_pairs} pairs.")
    print("Dataset generation complete!")

if __name__ == "__main__":
    test_dir = "d:/Internship/OCR_PDF/BoudningBoxCleaning/synthetic_dataset_v2_test"
    generate_dataset_v2(num_pairs=20, output_dir=test_dir)
