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

# Comprehensive vocabulary lists to ensure ALL character sets are covered
uppercase_chars = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
lowercase_chars = "abcdefghijklmnopqrstuvwxyz"
special_chars = ["\u00d8", "\u00b1", "\u00b0", "(", ")", "/", "x", "X", ":", "#", "<", ">", "\u03bc", "\"", "'", "%", "*", "=", ",", "."]

# Sample words/expressions common in CAD drawings (covering a-z, A-Z)
eng_words_upper = [
    "HOLE", "HOLES", "DIA", "RAD", "MAX", "MIN", "TYP", "PITCH", "CLASS", 
    "THD", "DEPTH", "CHAMFER", "CLEARANCE", "TOLERANCE", "REF", "QTY", 
    "SECTION", "DETAIL", "VEHICLE", "HARNESS", "BOARD", "SIDE", "UPWARD"
]

eng_words_lower = [
    "hole", "holes", "dia", "rad", "max", "min", "typ", "pitch", "class", 
    "thd", "depth", "chamfer", "clearance", "tolerance", "ref", "qty", 
    "section", "detail", "vehicle", "harness", "board", "side", "upward"
]

# Set of diverse template expressions utilizing all special characters
templates = [
    # Decimals, Tolerances, Diameter, Parentheses, Plus/Minus
    "\u00d8 {num_small} \u00b1 {tol}",
    "({num_large} \u00b1 {num_decimal})",
    "\u00d8 {num_small} +{tol_plus}/-{tol_minus}",
    # Multipliers, Degrees, Chamfers
    "{num_small}x \u00d8 {num_decimal} DEPTH {num_small}",
    "CHAMFER {num_decimal} x {num_small}\u00b0",
    # Feet, Inches, Minutes/Seconds, Comparisons
    "{num_large}' - {num_small}\" < {limit_large}",
    "ANGLE = {num_small}\u00b0 {num_small}' {num_small}\"",
    # Reference keys, Colons, Asterisks, Percents, Commas
    "QTY: {num_small}* (REF #{num_id})",
    "RATIO = {num_decimal}%",
    "COORD: {num_large}, {num_large}",
    # Slash, Class fit, Microns, Letter tags
    "M{num_small} x {pitch} - {class_fit}",
    "Ra {ra_val} \u03bcm MAX",
    "SECTION {char_upper}-{char_upper} (SCALE 1/{num_small})",
    # Mix of lowercase and uppercase text notes
    "{word_upper} qty: {num_small} [min {num_decimal}]",
    "depth > {num_decimal} mm ({word_lower})"
]

def generate_random_expression_v4():
    # 1. Alphanumeric / Value generation
    # Large numbers (e.g. 1000 - 9999)
    num_large = f"{random.randint(100, 3000)}"
    # Small numbers (e.g. 1 - 99)
    num_small = f"{random.randint(1, 99)}"
    # Decimals with varying decimal places (1 to 3 decimal places)
    dec_places = random.choice([1, 2, 3])
    num_decimal = f"{random.uniform(0.005, 50.0):.{dec_places}f}"
    
    # Tolerances
    tol = f"{random.uniform(0.01, 1.5):.{random.choice([1, 2])}f}"
    tol_plus = f"{random.uniform(0.01, 0.5):.2f}"
    tol_minus = f"{random.uniform(0.01, 0.5):.2f}"
    
    # Other values
    pitch = random.choice(["0.5", "0.75", "1.0", "1.25", "1.5", "2.0"])
    class_fit = random.choice(["6H", "6g", "7H", "8g"])
    ra_val = random.choice(["0.8", "1.6", "3.2", "6.3", "12.5"])
    limit_large = str(random.randint(2000, 5000))
    num_id = str(random.randint(1, 99))
    
    char_upper = random.choice(uppercase_chars)
    word_upper = random.choice(eng_words_upper)
    word_lower = random.choice(eng_words_lower)
    
    # Format a random template
    template = random.choice(templates)
    expr = template.format(
        num_large=num_large, num_small=num_small, num_decimal=num_decimal,
        tol=tol, tol_plus=tol_plus, tol_minus=tol_minus, pitch=pitch,
        class_fit=class_fit, ra_val=ra_val, limit_large=limit_large,
        num_id=num_id, char_upper=char_upper, word_upper=word_upper,
        word_lower=word_lower
    )
    
    # Sometimes force inject random characters/symbols to guarantee complete coverage
    if random.random() > 0.8:
        extra_sym = random.choice(special_chars)
        extra_let_l = random.choice(lowercase_chars)
        extra_let_u = random.choice(uppercase_chars)
        expr += f" {extra_sym}{extra_let_l}{extra_let_u}"
        
    return expr

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
    x1, y1 = p1
    x2, y2 = p2
    theta = math.atan2(y2 - y1, x2 - x1)
    
    alpha1 = theta + math.pi - math.radians(angle)
    alpha2 = theta + math.pi + math.radians(angle)
    
    ap1 = (x2 + size * math.cos(alpha1), y2 + size * math.sin(alpha1))
    ap2 = (x2 + size * math.cos(alpha2), y2 + size * math.sin(alpha2))
    
    draw.polygon([p2, ap1, ap2], fill=0)

def generate_pair_v4(index, output_dir):
    canvas_size = 450
    clean_canvas = Image.new("L", (canvas_size, canvas_size), 255)
    draw_clean = ImageDraw.Draw(clean_canvas)
    
    text = generate_random_expression_v4()
    font_size = random.randint(18, 26)
    font = get_font(font_size)
    
    bbox = draw_clean.textbbox((0, 0), text, font=font)
    text_w = bbox[2] - bbox[0]
    text_h = bbox[3] - bbox[1]
    
    tx = (canvas_size - text_w) // 2
    ty = (canvas_size - text_h) // 2
    
    draw_clean.text((tx, ty), text, font=font, fill=0)
    
    text_left = tx + bbox[0]
    text_top = ty + bbox[1]
    text_right = tx + bbox[2]
    text_bottom = ty + bbox[3]
    # Vary the buffer percent over a wide range:
    # 20% probability of very tight crop (5% to 12%)
    # 50% probability of standard crop (15% to 25%)
    # 30% probability of expanded crop (30% to 55%)
    crop_type_rand = random.random()
    if crop_type_rand < 0.20:
        buffer_percent = random.uniform(0.05, 0.12)  # Tight crop
    elif crop_type_rand < 0.70:
        buffer_percent = random.uniform(0.15, 0.25)  # Standard crop
    else:
        buffer_percent = random.uniform(0.30, 0.55)  # Expanded crop
        
    buf_w = int(text_w * buffer_percent)
    buf_h = int(text_h * buffer_percent)
    
    crop_left = max(0, text_left - buf_w)
    crop_top = max(0, text_top - buf_h)
    crop_right = min(canvas_size, text_right + buf_w)
    crop_bottom = min(canvas_size, text_bottom + buf_h)
    
    crop_w = crop_right - crop_left
    crop_h = crop_bottom - crop_top
    
    target_crop = clean_canvas.crop((crop_left, crop_top, crop_right, crop_bottom))
    input_crop = target_crop.copy()
    draw_input = ImageDraw.Draw(input_crop)
    
    t_left = text_left - crop_left
    t_top = text_top - crop_top
    t_right = text_right - crop_left
    t_bottom = text_bottom - crop_top
    
    line_width = random.choice([1, 2])
    
    layout_type = random.choices(
        ["parallel_below", "curved_below", "parallel_double", "crossing_horizontal", "parallel_above"],
        weights=[0.40, 0.30, 0.15, 0.10, 0.05],
        k=1
    )[0]
    
    if layout_type == "parallel_below":
        y_line = t_bottom + random.randint(2, 5)
        if y_line < crop_h:
            draw_input.line([(0, y_line), (crop_w - 1, y_line)], fill=0, width=line_width)
            if random.random() > 0.4:
                arrow_side = random.choice(["left", "right", "both"])
                if arrow_side in ["left", "both"]:
                    draw_arrowhead(draw_input, (crop_w // 2, y_line), (0, y_line))
                if arrow_side in ["right", "both"]:
                    draw_arrowhead(draw_input, (crop_w // 2, y_line), (crop_w - 1, y_line))
                    
    elif layout_type == "curved_below":
        y_p = t_bottom + random.randint(2, 4)
        x_c = crop_w // 2
        R = random.randint(150, 450)
        
        concave_up = random.random() > 0.5
        if concave_up:
            y_center = y_p - R
        else:
            y_center = y_p + R
            
        pts = []
        for x in range(crop_w):
            term = R**2 - (x - x_c)**2
            if term >= 0:
                if concave_up:
                    y = y_center + math.sqrt(term)
                else:
                    y = y_center - math.sqrt(term)
                pts.append((x, int(round(y))))
                
        for j in range(len(pts) - 1):
            if 0 <= pts[j][1] < crop_h and 0 <= pts[j+1][1] < crop_h:
                draw_input.line([pts[j], pts[j+1]], fill=0, width=line_width)
                
        if len(pts) > 10 and random.random() > 0.4:
            arrow_side = random.choice(["left", "right", "both"])
            if arrow_side in ["left", "both"]:
                draw_arrowhead(draw_input, pts[3], pts[0], size=6)
            if arrow_side in ["right", "both"]:
                draw_arrowhead(draw_input, pts[-4], pts[-1], size=6)
                
    elif layout_type == "parallel_above":
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
        y_line = (t_top + t_bottom) // 2
        draw_input.line([(0, y_line), (crop_w - 1, y_line)], fill=0, width=line_width)
        
    elif layout_type == "parallel_double":
        y_above = t_top - random.randint(2, 6)
        y_below = t_bottom + random.randint(2, 6)
        if y_above >= 0:
            draw_input.line([(0, y_above), (crop_w - 1, y_above)], fill=0, width=1)
        if y_below < crop_h:
            draw_input.line([(0, y_below), (crop_w - 1, y_below)], fill=0, width=1)
            
    # Add crossing perpendicular/extension lines in 70% of cases
    if random.random() > 0.3:
        x_line = random.choice([
            random.randint(0, max(5, t_left - 3)),
            random.randint(min(crop_w - 5, t_right + 3), crop_w - 1)
        ])
        angle_rad = math.radians(random.uniform(75, 105))
        h_half = crop_h // 2
        dx_val = int(h_half / math.tan(angle_rad))
        p1 = (x_line - dx_val, 0)
        p2 = (x_line + dx_val, crop_h - 1)
        draw_input.line([p1, p2], fill=0, width=1)

    # Apply OBB rectification alignment error
    angle = random.uniform(-4, 4)
    pad_val = 15
    padded_input = ImageOps.expand(input_crop, border=pad_val, fill=255)
    padded_target = ImageOps.expand(target_crop, border=pad_val, fill=255)
    
    rot_input = padded_input.rotate(angle, resample=Image.BICUBIC, fillcolor=255)
    rot_target = padded_target.rotate(angle, resample=Image.BICUBIC, fillcolor=255)
    
    final_input = rot_input.crop((pad_val, pad_val, pad_val + crop_w, pad_val + crop_h))
    final_target = rot_target.crop((pad_val, pad_val, pad_val + crop_w, pad_val + crop_h))
    
    np_in = np.array(final_input)
    np_tg = np.array(final_target)
    
    if random.random() > 0.6:
        np_in = cv2.GaussianBlur(np_in, (3, 3), 0)
    if random.random() > 0.8:
        noise = np.random.randint(0, 255, np_in.shape, dtype=np_in.dtype)
        np_in = np.where(noise < 3, 0, np_in)
        np_in = np.where(noise > 252, 255, np_in)
        
    _, np_in = cv2.threshold(np_in, 180, 255, cv2.THRESH_BINARY)
    _, np_tg = cv2.threshold(np_tg, 180, 255, cv2.THRESH_BINARY)
    
    input_save_path = os.path.join(output_dir, "inputs", f"pair_{index:06d}.png")
    target_save_path = os.path.join(output_dir, "targets", f"pair_{index:06d}.png")
    
    cv2.imwrite(input_save_path, np_in)
    cv2.imwrite(target_save_path, np_tg)
    print(f"Generated pair {index:06d}: '{text}'")

def generate_dataset_v4(num_pairs=50, output_dir="d:/Internship/OCR_PDF/BoudningBoxCleaning/synthetic_dataset_v4"):
    print(f"Creating directories in: {output_dir}")
    os.makedirs(os.path.join(output_dir, "inputs"), exist_ok=True)
    os.makedirs(os.path.join(output_dir, "targets"), exist_ok=True)
    
    print(f"Generating {num_pairs} training pairs covering all characters, decimals, and special symbols...")
    for idx in range(num_pairs):
        generate_pair_v4(idx, output_dir)
    print("Dataset generation complete!")

if __name__ == "__main__":
    test_dir = "d:/Internship/OCR_PDF/BoudningBoxCleaning/synthetic_dataset_v4_test"
    generate_dataset_v4(num_pairs=10, output_dir=test_dir)
