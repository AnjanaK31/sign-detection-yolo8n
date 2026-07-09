import os
import random
import math
import numpy as np
import cv2
import json
import multiprocessing
from PIL import Image, ImageDraw, ImageFont

CLASSES = [
    '0', '1', '2', '3', '4', '5', '6', '7', '8', '9',
    'plus_minus',     # ±
    'diameter',       # ⌀
    'radius',         # R
    'Rz',             # Rz
    'Ra',             # Ra
    'perpendicular',  # ⊥
    'parallel',       # ∥
    'circularity',    # ○
    'true_position',  # ⌀
    'arrow',          # Arrow head
    'comma'           # ,
]

CLASS_TO_CHAR = {
    '0': '0', '1': '1', '2': '2', '3': '3', '4': '4',
    '5': '5', '6': '6', '7': '7', '8': '8', '9': '9',
    'radius': 'R',
    'Rz': 'Rz',
    'Ra': 'Ra',
    'comma': ','
}

def get_split_name(page_idx, splits):
    for split_name, (start, end) in splits.items():
        if start <= page_idx < end:
            return split_name
    return "train"

def worker_generate_and_save_page(task_args):
    page_idx, bg_path, num_expr, font_path, yolo_dir, split_name, width, height = task_args
    generator = SyntheticDataGenerator(font_path=font_path)
    page_img, page_colored, labels, gt_details = generator.generate_full_page(bg_path, num_annotations=num_expr)
    
    img_name = f"page_{page_idx}.png"
    img_path = os.path.join(yolo_dir, "images", split_name, img_name)
    page_img.save(img_path)
    
    colored_img_name = f"page_{page_idx}_gt_colored.png"
    colored_img_path = os.path.join(yolo_dir, "images", split_name, colored_img_name)
    page_colored.save(colored_img_path)
    
    label_name = f"page_{page_idx}.txt"
    label_path = os.path.join(yolo_dir, "labels", split_name, label_name)
    with open(label_path, "w", encoding="utf-8") as f:
        for class_idx, pts in labels:
            pts_str = " ".join([f"{p:.6f}" for p in pts])
            f.write(f"{class_idx} {pts_str}\n")
            
    return img_name, gt_details

class SyntheticDataGenerator:
    def __init__(self, base_dir=".", font_path=None):
        self.base_dir = base_dir
        
        if font_path and os.path.exists(font_path):
            self.font_path = font_path
        else:
            font_choices = [
                "C:\\Windows\\Fonts\\seguisym.ttf",
                "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
                "/usr/share/fonts/truetype/freefont/FreeSans.ttf",
                "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",
                "/usr/share/fonts/TTF/DejaVuSans.ttf",
                "/usr/share/fonts/dejavu/DejaVuSans.ttf",
                "C:\\Windows\\Fonts\\arial.ttf",
                "C:\\Windows\\Fonts\\calibri.ttf",
                "C:\\Windows\\Fonts\\segoeui.ttf",
                "C:\\Windows\\Fonts\\tahoma.ttf"
            ]
            self.font_path = None
            for path in font_choices:
                if os.path.exists(path):
                    self.font_path = path
                    break
            
            if self.font_path is None:
                local_font = os.path.join(self.base_dir, "DejaVuSans.ttf")
                if os.path.exists(local_font):
                    self.font_path = local_font
        
        print(f"Using font: {self.font_path if self.font_path else 'Default Pillow Font'}")

    def get_font(self, size):
        if self.font_path:
            try:
                return ImageFont.truetype(self.font_path, size)
            except Exception:
                return ImageFont.load_default()
        return ImageFont.load_default()

    def apply_blueprint_effects(self, page_img):
        img_np = np.array(page_img.convert("RGB"))
        gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
        
        mean_val = np.mean(gray)
        if mean_val < 127:
            gray_input = cv2.bitwise_not(gray)
        else:
            gray_input = gray

        thresh = cv2.adaptiveThreshold(
            gray_input, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2
        )
        
        kernel = np.ones((2, 2), np.uint8)
        if random.random() < 0.35:
            thresh = cv2.erode(thresh, kernel, iterations=1)
        elif random.random() < 0.35:
            thresh = cv2.dilate(thresh, kernel, iterations=1)
            
        if random.random() < 0.4:
            noise = np.random.normal(0, 12, thresh.shape).astype(np.float32)
            noisy = thresh.astype(np.float32) + noise
            thresh = np.clip(noisy, 0, 255).astype(np.uint8)
            
        return Image.fromarray(cv2.cvtColor(thresh, cv2.COLOR_GRAY2RGB))

    def generate_backgrounds(self, output_dir, count=10):
        os.makedirs(output_dir, exist_ok=True)
        width, height = 2400, 3000
        
        for i in range(count):
            # Standard B&W background
            bg = Image.new("RGB", (width, height), (255, 255, 255))
            draw = ImageDraw.Draw(bg)
            
            border_margin = 40
            draw.rectangle(
                (border_margin, border_margin, width - border_margin, height - border_margin),
                outline=(100, 100, 100),
                width=4
            )
            
            tb_w, tb_h = 400, 150
            tb_x = width - border_margin - tb_w
            tb_y = height - border_margin - tb_h
            draw.rectangle((tb_x, tb_y, width - border_margin, height - border_margin), outline=(100, 100, 100), width=3)
            draw.line((tb_x, tb_y + 50, width - border_margin, tb_y + 50), fill=(100, 100, 100), width=2)
            draw.line((tb_x, tb_y + 100, width - border_margin, tb_y + 100), fill=(100, 100, 100), width=2)
            draw.line((tb_x + 150, tb_y, tb_x + 150, height - border_margin), fill=(100, 100, 100), width=2)
            draw.line((tb_x + 280, tb_y + 50, tb_x + 280, height - border_margin), fill=(100, 100, 100), width=2)
            
            self._draw_cad_shapes(draw, width, height, i, border_margin, color_mode=False)
            bg.save(os.path.join(output_dir, f"bg_{i}.png"))

            # GT Colored background (blue shapes, white background)
            bg_colored = Image.new("RGB", (width, height), (255, 255, 255))
            draw_col = ImageDraw.Draw(bg_colored)
            
            draw_col.rectangle(
                (border_margin, border_margin, width - border_margin, height - border_margin),
                outline=(0, 0, 255),
                width=4
            )
            draw_col.rectangle((tb_x, tb_y, width - border_margin, height - border_margin), outline=(0, 0, 255), width=3)
            draw_col.line((tb_x, tb_y + 50, width - border_margin, tb_y + 50), fill=(0, 0, 255), width=2)
            draw_col.line((tb_x, tb_y + 100, width - border_margin, tb_y + 100), fill=(0, 0, 255), width=2)
            draw_col.line((tb_x + 150, tb_y, tb_x + 150, height - border_margin), fill=(0, 0, 255), width=2)
            draw_col.line((tb_x + 280, tb_y + 50, tb_x + 280, height - border_margin), fill=(0, 0, 255), width=2)
            
            self._draw_cad_shapes(draw_col, width, height, i, border_margin, color_mode=True)
            bg_colored.save(os.path.join(output_dir, f"bg_{i}_gt_colored.png"))

    def _draw_arrowhead(self, draw, x1, y1, x2, y2, color=(100, 100, 100), size=12):
        dx = x2 - x1
        dy = y2 - y1
        length = math.hypot(dx, dy)
        if length == 0:
            return
        ux = dx / length
        uy = dy / length
        
        angle = math.pi * 5 / 6
        cos_a, sin_a = math.cos(angle), math.sin(angle)
        
        rx1 = ux * cos_a - uy * sin_a
        ry1 = ux * sin_a + uy * cos_a
        rx2 = ux * cos_a - uy * (-sin_a)
        ry2 = ux * (-sin_a) + uy * cos_a
        
        p1 = (x2 + rx1 * size, y2 + ry1 * size)
        p2 = (x2 + rx2 * size, y2 + ry2 * size)
        draw.polygon([(x2, y2), p1, p2], fill=color)

    def _draw_hatching(self, draw, x0, y0, x1, y1, spacing=16, color=(210, 210, 210), width=1):
        start_intercept = int(x0 + y0)
        end_intercept = int(x1 + y1)
        for intercept in range(start_intercept, end_intercept, spacing):
            pts = []
            for x in [x0, x1]:
                y = intercept - x
                if y0 <= y <= y1:
                    pts.append((x, y))
            for y in [y0, y1]:
                x = intercept - y
                if x0 <= x <= x1:
                    pts.append((x, y))
            pts = list(set(pts))
            if len(pts) >= 2:
                pts.sort()
                draw.line((pts[0][0], pts[0][1], pts[1][0], pts[1][1]), fill=color, width=width)

    def _draw_dimension_line(self, draw, x1, x2, y, offset_y, label, font, color=(100, 100, 100)):
        ext_y1 = y
        ext_y2 = y + offset_y
        
        draw.line((x1, ext_y1, x1, ext_y2 + (8 if offset_y > 0 else -8)), fill=color, width=1)
        draw.line((x2, ext_y1, x2, ext_y2 + (8 if offset_y > 0 else -8)), fill=color, width=1)
        
        dim_y = ext_y2
        draw.line((x1, dim_y, x2, dim_y), fill=color, width=1)
        
        self._draw_arrowhead(draw, x1 + 15, dim_y, x1, dim_y, color=color, size=8)
        self._draw_arrowhead(draw, x2 - 15, dim_y, x2, dim_y, color=color, size=8)
        
        try:
            left, top, right, bottom = font.getbbox(label)
            tw = right - left
            th = bottom - top
        except Exception:
            tw = len(label) * 8
            th = 12
            
        tx = (x1 + x2) // 2 - tw // 2
        ty = dim_y - th - 3 if offset_y > 0 else dim_y + 3
        
        draw.rectangle((tx - 4, ty - 2, tx + tw + 4, ty + th + 2), fill=(255, 255, 255))
        draw.text((tx, ty), label, font=font, fill=color)

    def _draw_cad_shapes(self, draw, width, height, style, margin, color_mode=False):
        if color_mode:
            color = (0, 0, 255)
            thin_color = (0, 0, 255)
            very_thin_color = (0, 0, 255)
        else:
            color = (120, 120, 120)
            thin_color = (180, 180, 180)
            very_thin_color = (220, 220, 220)
        
        min_x, max_x = margin + 50, width - margin - 50
        min_y, max_y = margin + 50, height - margin - 200
        cx, cy = (min_x + max_x) // 2, (min_y + max_y) // 2
        
        font = self.get_font(18)
        
        if style % 5 == 0:
            grid_size = 150
            for x in range(min_x, max_x, grid_size):
                draw.line((x, min_y, x, max_y), fill=very_thin_color, width=1)
            for y in range(min_y, max_y, grid_size):
                draw.line((min_x, y, max_x, y), fill=very_thin_color, width=1)
            
            draw.rectangle((cx - 300, cy - 300, cx + 300, cy + 300), outline=color, width=3)
            draw.ellipse((cx - 200, cy - 200, cx + 200, cy + 200), outline=color, width=2)
            draw.line((cx - 350, cy, cx + 350, cy), fill=thin_color, width=1)
            draw.line((cx, cy - 350, cx, cy + 350), fill=thin_color, width=1)
            
            self._draw_dimension_line(draw, cx - 300, cx + 300, cy - 300, -50, "600", font, color=color)
            self._draw_dimension_line(draw, cx - 200, cx + 200, cy - 200, -30, "⌀ 400", font, color=color)
            
        elif style % 5 == 1:
            draw.rectangle((cx - 150, cy - 400, cx + 150, cy + 400), outline=color, width=3)
            draw.rectangle((cx - 100, cy - 300, cx + 100, cy + 300), outline=color, width=2)
            draw.rectangle((cx - 60, cy - 200, cx + 60, cy + 200), outline=color, width=2)
            draw.line((cx, cy - 450, cx, cy + 450), fill=thin_color, width=1)
            
            self._draw_hatching(draw, cx - 150, cy - 400, cx - 100, cy + 400, spacing=20, color=very_thin_color)
            self._draw_hatching(draw, cx + 100, cy - 400, cx + 150, cy + 400, spacing=20, color=very_thin_color)
            
            self._draw_dimension_line(draw, cx - 150, cx + 150, cy - 400, -40, "300", font, color=color)
            self._draw_dimension_line(draw, cx - 100, cx + 100, cy - 300, -20, "200", font, color=color)
            
        elif style % 5 == 2:
            draw.ellipse((cx - 120, cy - 250, cx + 120, cy - 50), outline=color, width=3)
            draw.ellipse((cx - 120, cy + 50, cx + 120, cy + 250), outline=color, width=3)
            draw.polygon([
                (cx - 60, cy - 150), (cx + 60, cy - 150),
                (cx + 60, cy + 150), (cx - 60, cy + 150)
            ], outline=color, width=2)
            
            draw.ellipse((cx - 50, cy - 150, cx + 50, cy - 50), outline=color, width=2)
            draw.ellipse((cx - 50, cy + 50, cx + 50, cy + 150), outline=color, width=2)
            
            self._draw_hatching(draw, cx - 55, cy - 140, cx + 55, cy + 140, spacing=15, color=very_thin_color)
            
            self._draw_dimension_line(draw, cx - 120, cx + 120, cy - 250, -40, "240", font, color=color)
            self._draw_dimension_line(draw, cx - 50, cx + 50, cy - 150, -20, "⌀ 100", font, color=color)
            
        elif style % 5 == 3:
            draw.polygon([
                (cx - 350, cy - 200), (cx + 350, cy - 200),
                (cx + 250, cy + 200), (cx - 250, cy + 200)
            ], outline=color, width=3)
            
            draw.polygon([(cx - 200, cy - 100), (cx - 100, cy - 100), (cx - 150, cy + 100)], outline=color, width=2)
            draw.polygon([(cx + 100, cy - 100), (cx + 200, cy - 100), (cx + 150, cy + 100)], outline=color, width=2)
            
            draw.line((cx - 400, cy, cx + 400, cy), fill=thin_color, width=1)
            
            self._draw_dimension_line(draw, cx - 350, cx + 350, cy - 200, -50, "700", font, color=color)
            self._draw_dimension_line(draw, cx - 250, cx + 250, cy + 200, 50, "500", font, color=color)
            
        else:
            draw.rectangle((min_x + 50, min_y + 50, max_x - 50, max_y - 50), outline=color, width=3)
            draw.rectangle((min_x + 80, min_y + 80, max_x - 80, max_y - 80), outline=color, width=2)
            
            for i in range(1, 4):
                offset_x = (max_x - min_x) // 4 * i
                offset_y = (max_y - min_y) // 4 * i
                draw.line((min_x + offset_x, min_y, min_x + offset_x, max_y), fill=very_thin_color, width=1)
                draw.line((min_x, min_y + offset_y, max_x, min_y + offset_y), fill=very_thin_color, width=1)
            
            draw.text((min_x + 100, min_y + 100), "DETAIL D", font=font, fill=color)
            draw.text((min_x + 100, min_y + 130), "SCALE 2:1", font=font, fill=color)
            draw.text((max_x - 250, min_y + 100), "REF ONLY", font=font, fill=color)

    def get_rotated_obb(self, cx, cy, w, h, angle):
        rad = math.radians(angle)
        cos_a = math.cos(rad)
        sin_a = math.sin(rad)
        
        local_pts = [
            (-w / 2, -h / 2),
            (w / 2, -h / 2),
            (w / 2, h / 2),
            (-w / 2, h / 2)
        ]
        
        world_pts = []
        for x, y in local_pts:
            rx = cx + x * cos_a + y * sin_a
            ry = cy - x * sin_a + y * cos_a
            world_pts.append((rx, ry))
            
        return world_pts

    def check_collision(self, bbox_list, new_bbox):
        nx_min, ny_min, nx_max, ny_max = new_bbox
        padding = 45
        nx_min -= padding
        ny_min -= padding
        nx_max += padding
        ny_max += padding
        
        for ex_min, ey_min, ex_max, ey_max in bbox_list:
            if not (nx_max < ex_min or nx_min > ex_max or ny_max < ey_min or ny_min > ey_max):
                return True
        return False

    def get_random_expression(self):
        templates = [
            lambda: f"{random.choice(['⌀', 'R', 'Ra', 'Rz'])} {random.randint(1, 9)}",
            lambda: f"{random.choice(['>', '<', '≥'])} {random.choice(['0', '1', '5'])}",
            lambda: f"{random.choice(['90', '45', '30'])}°",
            lambda: f"+{random.choice(['0.01', '0.05', '0.1'])}",
            lambda: f"-{random.choice(['0.02', '0.05', '0.2'])}",
            lambda: ",",
            lambda: f"⌀ {random.randint(5, 120)}{random.choice(['.5', '.0', ''])} ± {random.choice(['0.02', '0.05', '0.1'])}",
            lambda: f"⌀ {random.randint(5, 60)} +{random.choice(['0.05', '0.1', '0.2'])}",
            lambda: f"R {random.randint(2, 50)}{random.choice(['.0', ''])} ± {random.choice(['0.1', '0.2', '0.5'])}",
            lambda: f"{random.choice([15, 30, 45, 60, 90])}° ± {random.choice(['0.5°', '1°'])}",
            lambda: f"M{random.choice([4, 5, 6, 8, 10, 12, 16])} x {random.choice(['0.7', '0.8', '1.0', '1.25', '1.5', '2.0'])}",
            lambda: f"M{random.choice([8, 10, 12, 16])} x {random.choice(['1.25', '1.5', '2.0'])} - {random.choice(['6g', '6H'])}",
            lambda: f"{random.choice(['Ra', 'Rz'])} {random.choice(['0.8', '1.6', '3.2', '6.3', '12.5'])} μm",
            lambda: f"R {random.randint(10, 50)}.0 ± {random.choice(['0.2', '0.5'])} (TYP)",
            lambda: f"⊥ {random.choice(['0.02', '0.05', '0.1'])} A",
            lambda: f"∥ {random.choice(['0.03', '0.05', '0.1'])} A B",
            lambda: f"○ {random.choice(['0.01', '0.05', '0.08'])}",
            lambda: f"⊕ ⌀ {random.choice(['0.05', '0.1', '0.2'])} A B C"
        ]
        return random.choice(templates)()

    def generate_full_page(self, bg_image_path, num_annotations=15):
        bg = Image.open(bg_image_path)
        width, height = bg.size
        
        page_img = bg.copy()
        draw = ImageDraw.Draw(page_img, "RGBA")
        
        # Load colored background
        colored_bg_path = bg_image_path.replace(".png", "_gt_colored.png")
        if os.path.exists(colored_bg_path):
            page_colored = Image.open(colored_bg_path)
        else:
            page_colored = Image.new("RGB", (width, height), (255, 255, 255))
        draw_colored = ImageDraw.Draw(page_colored, "RGBA")
        
        placed_aabbs = []
        labels = []
        gt_details = []
        
        margin = 60
        min_x, max_x = margin + 120, width - margin - 120
        min_y, max_y = margin + 120, height - margin - 250
        
        for _ in range(num_annotations):
            n_lines = random.choice([1, 2, 3])
            font_size = random.randint(18, 38)
            font = self.get_font(font_size)
            
            processed_lines = []
            for _ in range(n_lines):
                text = self.get_random_expression()
                try:
                    left, top, right, bottom = font.getbbox(text)
                    tw = right - left
                    th = bottom - top
                except AttributeError:
                    tw = len(text) * (font_size // 2)
                    th = font_size
                
                pad_canvas = int(max(tw, th) * 2)
                txt_canvas = Image.new("RGBA", (pad_canvas, pad_canvas), (255, 255, 255, 0))
                txt_draw = ImageDraw.Draw(txt_canvas)
                txt_draw.text((pad_canvas // 2 - tw // 2, pad_canvas // 2 - th // 2), text, font=font, fill=(0, 0, 0, 255))
                
                # Red text canvas for the colored page
                txt_canvas_colored = Image.new("RGBA", (pad_canvas, pad_canvas), (255, 255, 255, 0))
                txt_draw_colored = ImageDraw.Draw(txt_canvas_colored)
                txt_draw_colored.text((pad_canvas // 2 - tw // 2, pad_canvas // 2 - th // 2), text, font=font, fill=(255, 0, 0, 255))
                
                processed_lines.append({
                    "canvas": txt_canvas,
                    "canvas_colored": txt_canvas_colored,
                    "text": text,
                    "w": tw,
                    "h": th,
                })
                
            line_spacing = max(line["h"] for line in processed_lines) + 16
            block_h = n_lines * line_spacing
            block_w = max(line["w"] for line in processed_lines)
            
            if random.random() < 0.7:
                line_angle = random.choice([0, 90, 180, 270]) + random.uniform(-5, 5)
            else:
                line_angle = random.uniform(0, 360)
                
            line_angle = line_angle % 360
            rad = math.radians(line_angle)
            cos_a = math.cos(rad)
            sin_a = math.sin(rad)
            
            placed = False
            for _ in range(300):
                cx = random.randint(min_x, max_x)
                cy = random.randint(min_y, max_y)
                
                half_w = (block_w + 16) // 2
                half_h = (block_h + 16) // 2
                
                abs_ang = abs(line_angle)
                if (45 < abs_ang < 135) or (225 < abs_ang < 315):
                    xmin, xmax = cx - half_h, cx + half_h
                    ymin, ymax = cy - half_w, cy + half_w
                else:
                    xmin, xmax = cx - half_w, cx + half_w
                    ymin, ymax = cy - half_h, cy + half_h
                    
                if xmin < min_x or xmax > max_x or ymin < min_y or ymax > max_y:
                    continue
                    
                if not self.check_collision(placed_aabbs, (xmin, ymin, xmax, ymax)):
                    placed_aabbs.append((xmin, ymin, xmax, ymax))
                    placed = True
                    break
                    
            if not placed:
                continue
                
            lines_to_draw = []
            for idx, line in enumerate(processed_lines):
                dy_local = (idx - (n_lines - 1) / 2) * line_spacing
                ccx = cx - dy_local * sin_a
                ccy = cy + dy_local * cos_a
                
                line_w = line["w"]
                line_h = line["h"]
                
                pad_w = 12
                pad_h = 6
                line_obb = self.get_rotated_obb(ccx, ccy, line_w + pad_w, line_h + pad_h, line_angle)
                
                rotated_txt = line["canvas"].rotate(line_angle, expand=True, resample=Image.Resampling.BICUBIC)
                rotated_txt_colored = line["canvas_colored"].rotate(line_angle, expand=True, resample=Image.Resampling.BICUBIC)
                rw, rh = rotated_txt.size
                
                paste_x = int(ccx - rw / 2)
                paste_y = int(ccy - rh / 2)
                
                lines_to_draw.append({
                    "obb": line_obb,
                    "rotated_txt": rotated_txt,
                    "rotated_txt_colored": rotated_txt_colored,
                    "paste_x": paste_x,
                    "paste_y": paste_y,
                    "line": line,
                    "line_obb": line_obb
                })
            
            for item in lines_to_draw:
                draw.polygon([(float(pt[0]), float(pt[1])) for pt in item["obb"]], fill=(255, 255, 255, 255))
                draw_colored.polygon([(float(pt[0]), float(pt[1])) for pt in item["obb"]], fill=(255, 255, 255, 255))
                
            for item in lines_to_draw:
                page_img.paste(item["rotated_txt"], (item["paste_x"], item["paste_y"]), mask=item["rotated_txt"])
                page_colored.paste(item["rotated_txt_colored"], (item["paste_x"], item["paste_y"]), mask=item["rotated_txt_colored"])
                
                norm_pts = []
                for px, py in item["line_obb"]:
                    norm_pts.append(px / width)
                    norm_pts.append(py / height)
                labels.append((0, norm_pts))
                
                gt_details.append({
                    "class": 0,
                    "corners": [[float(pt[0]), float(pt[1])] for pt in item["line_obb"]],
                    "text": item["line"]["text"],
                    "angle": float(line_angle)
                })
                
        preprocessed_page = self.apply_blueprint_effects(page_img)
        return preprocessed_page, page_colored, labels, gt_details

    def generate_test_crop(self, text, line_type="straight", angle=0):
        """Generates standard and big crops along with ground-truth masks for evaluation.
        
        Args:
            text: str, the text/expression to render in the crop.
            line_type: str, 'straight', 'curved', or 'none'.
            angle: float, rotation angle in degrees.
            
        Returns:
            crop_std: PIL Image (standard crop with 10% buffer)
            crop_big: PIL Image (big crop with 40% buffer)
            gt_text_mask: np.ndarray (binary mask of text pixels, in standard crop size)
            gt_line_mask: np.ndarray (binary mask of CAD line pixels, in standard crop size)
        """
        from rectifier import rectify_crop
        
        # Create a large canvas to draw individual layers
        canvas_w, canvas_h = 500, 500
        cx, cy = canvas_w // 2, canvas_h // 2
        
        # 1. Text Layer
        text_layer = Image.new("RGBA", (canvas_w, canvas_h), (255, 255, 255, 0))
        text_draw = ImageDraw.Draw(text_layer)
        font_size = 32
        font = self.get_font(font_size)
        
        # Measure text size
        try:
            left, top, right, bottom = font.getbbox(text)
            tw = right - left
            th = bottom - top
        except Exception:
            tw = len(text) * (font_size // 2)
            th = font_size
            
        # Draw centered text
        text_draw.text((cx - tw // 2, cy - th // 2), text, font=font, fill=(0, 0, 0, 255))
        
        # Rotate text layer
        text_layer = text_layer.rotate(angle, expand=False, resample=Image.Resampling.BICUBIC)
        
        # 2. Line Layer
        line_layer = Image.new("RGBA", (canvas_w, canvas_h), (255, 255, 255, 0))
        line_draw = ImageDraw.Draw(line_layer)
        
        # Draw background CAD line crossing the center
        color = (0, 0, 0, 255)
        thickness = 2
        
        if line_type == "straight":
            # Draw a diagonal line crossing the text
            line_draw.line((cx - 150, cy - 100, cx + 150, cy + 100), fill=color, width=thickness)
        elif line_type == "curved":
            # Draw a curved arc crossing the text (e.g. circle path)
            line_draw.arc((cx - 120, cy - 120, cx + 120, cy + 120), start=0, end=180, fill=color, width=thickness)
            
        # Rotate line layer with same angle
        line_layer = line_layer.rotate(angle + random.uniform(-10, 10), expand=False, resample=Image.Resampling.BICUBIC)
        
        # 3. Combine Layers
        combined = Image.new("RGB", (canvas_w, canvas_h), (255, 255, 255))
        combined.paste(line_layer, (0, 0), mask=line_layer)
        combined.paste(text_layer, (0, 0), mask=text_layer)
        
        # 4. Extract rectified crops using rectifier
        crop_std = rectify_crop(
            combined,
            bbox_metrics={'cx': cx, 'cy': cy, 'w': tw, 'h': th, 'angle': angle},
            buffer_percent=0.20
        )
        
        crop_big = rectify_crop(
            combined,
            bbox_metrics={'cx': cx, 'cy': cy, 'w': tw, 'h': th, 'angle': angle},
            buffer_percent=0.40
        )
        
        # 5. Extract rectified ground truth masks
        text_rgb = Image.new("RGB", (canvas_w, canvas_h), (255, 255, 255))
        text_rgb.paste(text_layer, (0, 0), mask=text_layer)
        
        line_rgb = Image.new("RGB", (canvas_w, canvas_h), (255, 255, 255))
        line_rgb.paste(line_layer, (0, 0), mask=line_layer)
        
        rectified_text_std = rectify_crop(
            text_rgb,
            bbox_metrics={'cx': cx, 'cy': cy, 'w': tw, 'h': th, 'angle': angle},
            buffer_percent=0.20
        )
        
        rectified_line_std = rectify_crop(
            line_rgb,
            bbox_metrics={'cx': cx, 'cy': cy, 'w': tw, 'h': th, 'angle': angle},
            buffer_percent=0.20
        )
        
        # Convert rectified ground truth images to binary masks
        text_np = np.array(rectified_text_std.convert("L"))
        _, gt_text_mask = cv2.threshold(text_np, 127, 255, cv2.THRESH_BINARY_INV)
        
        line_np = np.array(rectified_line_std.convert("L"))
        _, gt_line_mask = cv2.threshold(line_np, 127, 255, cv2.THRESH_BINARY_INV)
        
        # Exclude text pixels from line mask to keep them disjoint
        gt_line_mask = cv2.subtract(gt_line_mask, gt_text_mask)
        
        return crop_std, crop_big, gt_text_mask, gt_line_mask


    def save_dataset(self, num_pdfs=16, pages_per_pdf=5, min_expr=30, max_expr=50, pdf_dir="pdfs", yolo_dir="dataset_yolo", cores=None):
        bg_dir = "temp_backgrounds"
        self.generate_backgrounds(bg_dir, count=10)
        
        total_pages = num_pdfs * pages_per_pdf
        print(f"Generating {total_pages} total pages across {num_pdfs} PDFs...")
        os.makedirs(pdf_dir, exist_ok=True)
        
        train_end = max(1, int(round(0.75 * total_pages)))
        val_end = max(train_end + 1, train_end + int(round(0.15 * total_pages)))
        if val_end >= total_pages:
            val_end = total_pages - 1
            if val_end <= train_end:
                train_end = max(1, val_end - 1)
        
        splits = {
            "train": (0, train_end),
            "val": (train_end, val_end),
            "test": (val_end, total_pages)
        }
        
        for name, (start, end) in splits.items():
            os.makedirs(os.path.join(yolo_dir, "images", name), exist_ok=True)
            os.makedirs(os.path.join(yolo_dir, "labels", name), exist_ok=True)
            
        gt_database = {}
        backgrounds = [os.path.join(bg_dir, f"bg_{i}.png") for i in range(10)]
        
        if cores is None:
            cores = max(1, multiprocessing.cpu_count() - 1)
        print(f"Parallel processing enabled using {cores} CPU core(s)...")
        
        task_args = []
        for page_idx in range(total_pages):
            bg_path = backgrounds[page_idx % len(backgrounds)]
            num_expr = random.randint(min_expr, max_expr)
            split_name = get_split_name(page_idx, splits)
            
            task_args.append((
                page_idx, 
                bg_path, 
                num_expr, 
                self.font_path, 
                yolo_dir, 
                split_name, 
                2400,
                3000
            ))
            
        with multiprocessing.Pool(processes=cores) as pool:
            results = list(pool.imap_unordered(worker_generate_and_save_page, task_args))
                        
        for img_name, gt_details in results:
            gt_database[img_name] = gt_details
                        
        gt_json_path = os.path.join(yolo_dir, "ground_truth.json")
        with open(gt_json_path, "w", encoding="utf-8") as f:
            json.dump(gt_database, f, indent=2, ensure_ascii=False)
        print(f"Saved ground truth JSON database to: {gt_json_path}")
        
        print(f"Compiling {num_pdfs} multi-page PDFs...")
        for pdf_idx in range(num_pdfs):
            pdf_pages = []
            for page_offset in range(pages_per_pdf):
                page_idx = pdf_idx * pages_per_pdf + page_offset
                split_name = get_split_name(page_idx, splits)
                img_path = os.path.join(yolo_dir, "images", split_name, f"page_{page_idx}.png")
                if os.path.exists(img_path):
                    pdf_pages.append(Image.open(img_path).convert("RGB"))
            
            pdf_path = os.path.join(pdf_dir, f"blueprint_{pdf_idx}.pdf")
            if pdf_pages:
                pdf_pages[0].save(pdf_path, save_all=True, append_images=pdf_pages[1:], format="PDF")
                for img in pdf_pages:
                    img.close()
            
        data_yaml_content = f"""path: {os.path.abspath(yolo_dir)}
train: images/train
val: images/val
test: images/test

names:
  0: symbol
"""
        with open("data.yaml", "w", encoding="utf-8") as f:
            f.write(data_yaml_content)
            
        print("YOLO dataset generation complete!")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Synthetic B&W CAD Dataset Generator")
    parser.add_argument("--num-pdfs", "-n", type=int, default=2, help="Number of PDFs to generate")
    parser.add_argument("--pages-per-pdf", "-p", type=int, default=3, help="Number of pages in each PDF")
    parser.add_argument("--min-expr", type=int, default=15, help="Minimum expressions per page")
    parser.add_argument("--max-expr", type=int, default=25, help="Maximum expressions per page")
    parser.add_argument("--yolo-dir", default="dataset_yolo", help="Output directory for YOLO dataset")
    parser.add_argument("--pdf-dir", default="pdfs", help="Output directory for generated PDFs")
    parser.add_argument("--font-path", default=None, help="Path to a custom TrueType font (.ttf)")
    parser.add_argument("--cores", type=int, default=None, help="Number of CPU cores to use for parallel generation")
    args = parser.parse_args()
    
    generator = SyntheticDataGenerator(font_path=args.font_path)
    generator.save_dataset(
        num_pdfs=args.num_pdfs,
        pages_per_pdf=args.pages_per_pdf,
        min_expr=args.min_expr,
        max_expr=args.max_expr,
        pdf_dir=args.pdf_dir,
        yolo_dir=args.yolo_dir,
        cores=args.cores
    )
