import cv2
import math
import numpy as np
import os
import random
from PIL import Image, ImageDraw, ImageFont, ImageOps
import sys

sys.path.append("d:/Internship/OCR_PDF/BoudningBoxCleaning/scratch")
from generate_synthetic_dataset_v3 import (
    generate_random_expression,
    get_font,
    draw_arrowhead
)

def generate_curved_only(index, output_dir):
    canvas_size = 400
    clean_canvas = Image.new("L", (canvas_size, canvas_size), 255)
    draw_clean = ImageDraw.Draw(clean_canvas)
    
    text = generate_random_expression()
    font_size = random.randint(18, 28)
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
    
    buffer_percent = random.uniform(0.15, 0.25)
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
    t_bottom = text_bottom - crop_top
    
    line_width = random.choice([1, 2])
    
    # Force layout_type = curved_below
    y_p = t_bottom + random.randint(2, 4)
    x_c = crop_w // 2
    R = random.randint(150, 350)
    
    # Decide orientation: concave up or concave down
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
            
    if len(pts) > 10:
        p_end = pts[-1]
        p_prev = pts[-4]
        draw_arrowhead(draw_input, p_prev, p_end, size=6)
        
    # Rotate slightly
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
    
    _, np_in = cv2.threshold(np_in, 180, 255, cv2.THRESH_BINARY)
    _, np_tg = cv2.threshold(np_tg, 180, 255, cv2.THRESH_BINARY)
    
    input_save_path = os.path.join(output_dir, "inputs", f"curved_{index:02d}.png")
    target_save_path = os.path.join(output_dir, "targets", f"curved_{index:02d}.png")
    
    cv2.imwrite(input_save_path, np_in)
    cv2.imwrite(target_save_path, np_tg)

if __name__ == "__main__":
    test_dir = "d:/Internship/OCR_PDF/BoudningBoxCleaning/scratch/curved_test"
    os.makedirs(os.path.join(test_dir, "inputs"), exist_ok=True)
    os.makedirs(os.path.join(test_dir, "targets"), exist_ok=True)
    
    for idx in range(5):
        generate_curved_only(idx, test_dir)
    print("Forced curved crops generated successfully!")
