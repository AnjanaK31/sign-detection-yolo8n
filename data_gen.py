import os
import random
import math
import numpy as np
import cv2
from PIL import Image, ImageDraw, ImageFont

class SyntheticDataGenerator:
    def __init__(self, base_dir="."):
        self.base_dir = base_dir
        
        # Look for standard Windows fonts, or fall back to default
        font_choices = [
            "C:\\Windows\\Fonts\\arial.ttf",
            "C:\\Windows\\Fonts\\calibri.ttf",
            "C:\\Windows\\Fonts\\segoeui.ttf",
            "C:\\Windows\\Fonts\\tahoma.ttf",
            "C:\\Windows\\Fonts\\cour.ttf"
        ]
        self.font_path = None
        for path in font_choices:
            if os.path.exists(path):
                self.font_path = path
                break
        
        print(f"Using font: {self.font_path if self.font_path else 'Default Pillow Font'}")

    def get_font(self, size):
        if self.font_path:
            try:
                return ImageFont.truetype(self.font_path, size)
            except Exception:
                return ImageFont.load_default()
        return ImageFont.load_default()

    def generate_backgrounds(self, output_dir, count=20):
        """Generates 'count' distinct, text-free engineering/blueprint background images."""
        os.makedirs(output_dir, exist_ok=True)
        width, height = 1600, 2000
        
        for i in range(count):
            bg = Image.new("RGB", (width, height), (255, 255, 255))
            draw = ImageDraw.Draw(bg)
            
            # 1. Draw outer boundary frame
            border_margin = 40
            draw.rectangle(
                (border_margin, border_margin, width - border_margin, height - border_margin),
                outline=(0, 0, 0),
                width=4
            )
            
            # 2. Draw generic empty title block in the bottom right corner
            tb_w, tb_h = 400, 150
            tb_x = width - border_margin - tb_w
            tb_y = height - border_margin - tb_h
            draw.rectangle((tb_x, tb_y, width - border_margin, height - border_margin), outline=(0, 0, 0), width=3)
            draw.line((tb_x, tb_y + 50, width - border_margin, tb_y + 50), fill=(0, 0, 0), width=2)
            draw.line((tb_x, tb_y + 100, width - border_margin, tb_y + 100), fill=(0, 0, 0), width=2)
            draw.line((tb_x + 150, tb_y, tb_x + 150, height - border_margin), fill=(0, 0, 0), width=2)
            draw.line((tb_x + 280, tb_y + 50, tb_x + 280, height - border_margin), fill=(0, 0, 0), width=2)
            
            # 3. Draw style-specific CAD/blueprint content
            self._draw_cad_shapes(draw, width, height, i, border_margin)
            
            bg.save(os.path.join(output_dir, f"bg_{i}.png"))
        
        print(f"Generated {count} text-free backgrounds in {output_dir}")

    def _draw_cad_shapes(self, draw, width, height, style, margin):
        """Draws various CAD shapes depending on the style index to create realistic blueprint elements."""
        color = (80, 80, 80)
        thin_color = (180, 180, 180)
        
        min_x, max_x = margin + 50, width - margin - 50
        min_y, max_y = margin + 50, height - margin - 200
        
        if style % 20 == 0:
            grid_size = 80
            for x in range(min_x, max_x, grid_size):
                draw.line((x, min_y, x, max_y), fill=thin_color, width=1)
            for y in range(min_y, max_y, grid_size):
                draw.line((min_x, y, max_x, y), fill=thin_color, width=1)
                
        elif style % 20 == 1:
            cx, cy = (min_x + max_x) // 2, (min_y + max_y) // 2
            for r in [100, 200, 300, 450]:
                draw.ellipse((cx - r, cy - r, cx + r, cy + r), outline=color, width=2)
            draw.line((min_x, cy, max_x, cy), fill=color, width=1)
            draw.line((cx, min_y, cx, max_y), fill=color, width=1)
            for angle in [30, 45, 60, 120, 135, 150]:
                rad = math.radians(angle)
                dx, dy = int(500 * math.cos(rad)), int(500 * math.sin(rad))
                draw.line((cx - dx, cy - dy, cx + dx, cy + dy), fill=color, width=1)
                
        elif style % 20 == 2:
            cy = (min_y + max_y) // 2
            stages = [
                (min_x + 100, cy - 80, min_x + 300, cy + 80),
                (min_x + 300, cy - 150, min_x + 600, cy + 150),
                (min_x + 600, cy - 100, min_x + 900, cy + 100),
                (min_x + 900, cy - 130, min_x + 1200, cy + 130),
                (min_x + 1200, cy - 70, max_x - 100, cy + 70)
            ]
            for (x1, y1, x2, y2) in stages:
                draw.rectangle((x1, y1, x2, y2), outline=color, width=3)
                if x1 == min_x + 600:
                    for hx in range(x1, x2, 15):
                        draw.line((hx, y1, hx + 15, y2), fill=thin_color, width=1)
            draw.line((min_x + 50, cy, max_x - 50, cy), fill=color, width=1)

        elif style % 20 == 3:
            y_bus1, y_bus2 = min_y + 150, max_y - 150
            draw.line((min_x, y_bus1, max_x, y_bus1), fill=color, width=3)
            draw.line((min_x, y_bus2, max_x, y_bus2), fill=color, width=3)
            rung_spacing = (max_x - min_x) // 6
            for col in range(1, 6):
                x = min_x + col * rung_spacing
                draw.line((x, y_bus1, x, y_bus2), fill=color, width=2)
                cy_rung = (y_bus1 + y_bus2) // 2
                if col in [1, 3, 5]:
                    draw.rectangle((x - 20, cy_rung - 30, x + 20, cy_rung + 30), outline=(255, 255, 255), fill=(255, 255, 255))
                    draw.line((x - 20, cy_rung - 10, x + 20, cy_rung - 10), fill=color, width=2)
                    draw.line((x - 20, cy_rung + 10, x + 20, cy_rung + 10), fill=color, width=2)
                elif col in [2, 4]:
                    draw.rectangle((x - 25, cy_rung - 40, x + 25, cy_rung + 40), outline=color, fill=(255, 255, 255), width=2)

        elif style % 20 == 4:
            draw.rectangle((min_x, min_y, max_x, max_y), outline=color, width=4)
            draw.rectangle((min_x + 15, min_y + 15, max_x - 15, max_y - 15), outline=color, width=2)
            x_mid = (min_x + max_x) // 2
            y_mid = (min_y + max_y) // 2
            draw.line((x_mid, min_y, x_mid, max_y), fill=color, width=4)
            draw.line((x_mid + 15, min_y + 15, x_mid + 15, max_y - 15), fill=color, width=2)
            draw.line((min_x, y_mid, x_mid, y_mid), fill=color, width=4)
            draw.arc((x_mid - 80, y_mid - 80, x_mid, y_mid), start=0, end=90, fill=color, width=2)
            draw.line((x_mid - 80, y_mid, x_mid, y_mid), fill=color, width=2)

        elif style % 20 == 5:
            for offset in [200, 450, 700]:
                draw.line((min_x, min_y + offset, max_x, min_y + offset), fill=color, width=2)
                draw.line((min_x + offset, min_y, min_x + offset, max_y), fill=color, width=2)
                vx, vy = min_x + offset, min_y + offset
                draw.polygon([(vx - 20, vy - 15), (vx - 20, vy + 15), (vx, vy)], outline=color, fill=(255, 255, 255))
                draw.polygon([(vx + 20, vy - 15), (vx + 20, vy + 15), (vx, vy)], outline=color, fill=(255, 255, 255))
            draw.rectangle((min_x + 800, min_y + 200, min_x + 1100, min_y + 600), outline=color, fill=(255, 255, 255), width=3)
            draw.ellipse((min_x + 800, min_y + 150, min_x + 1100, min_y + 250), outline=color, fill=(255, 255, 255), width=3)
            draw.ellipse((min_x + 800, min_y + 550, min_x + 1100, min_y + 650), outline=color, fill=(255, 255, 255), width=3)

        elif style % 20 == 6:
            spacing = 100
            for d in range(-max_y, max_x, spacing):
                draw.line((d, min_y, d + int((max_y - min_y) * 1.732), max_y), fill=thin_color, width=1)
                draw.line((d + int((max_y - min_y) * 1.732), min_y, d, max_y), fill=thin_color, width=1)

        elif style % 20 == 7:
            cx, cy = (min_x + max_x) // 2, (min_y + max_y) // 2
            draw.ellipse((cx - 80, cy - 80, cx + 80, cy + 80), outline=color, width=2)
            draw.ellipse((cx - 150, cy - 150, cx + 150, cy + 150), outline=color, width=2)
            draw.ellipse((cx - 300, cy - 300, cx + 300, cy + 300), outline=color, width=2)
            for angle in range(0, 360, 15):
                rad = math.radians(angle)
                x1 = cx + int(300 * math.cos(rad))
                y1 = cy + int(300 * math.sin(rad))
                x2 = cx + int(340 * math.cos(rad))
                y2 = cy + int(340 * math.sin(rad))
                draw.line((x1, y1, x2, y2), fill=color, width=3)

        elif style % 20 == 8:
            poly1 = [(min_x + 100, min_y + 100), (min_x + 500, min_y + 100), (min_x + 400, min_y + 600), (min_x + 100, min_y + 400)]
            poly2 = [(min_x + 600, min_y + 200), (max_x - 100, min_y + 200), (max_x - 200, max_y - 200), (min_x + 700, max_y - 200)]
            draw.polygon(poly1, outline=color, width=3)
            draw.polygon(poly2, outline=color, width=3)
            for k in range(-500, 1000, 20):
                draw.line((min_x + k, min_y, min_x + k + 500, min_y + 500), fill=thin_color, width=1)
            for k in range(0, 2000, 20):
                draw.line((min_x + k, min_y, min_x + k - 600, min_y + 600), fill=thin_color, width=1)

        elif style % 20 == 9:
            col_x1 = min_x + 200
            col_x2 = max_x - 200
            y_foundation = max_y - 50
            draw.line((col_x1, min_y + 100, col_x1, y_foundation), fill=color, width=5)
            draw.line((col_x2, min_y + 100, col_x2, y_foundation), fill=color, width=5)
            draw.line((col_x1 - 100, min_y + 100, col_x2 + 100, min_y + 100), fill=color, width=5)
            draw.line((col_x1, min_y + 220, col_x2, min_y + 220), fill=color, width=4)
            truss_step = (col_x2 - col_x1) // 6
            for step in range(6):
                x_start = col_x1 + step * truss_step
                x_end = x_start + truss_step
                draw.line((x_start, min_y + 100, x_end, min_y + 220), fill=color, width=3)
                draw.line((x_start, min_y + 220, x_end, min_y + 100), fill=color, width=3)

        else:
            grid_size = 150
            for x in range(min_x, max_x, grid_size):
                draw.line((x, margin, x, margin + 20), fill=color, width=2)
                draw.line((x, height - margin, x, height - margin - 20), fill=color, width=2)
            for y in range(min_y, max_y, grid_size):
                draw.line((margin, y, margin + 20, y), fill=color, width=2)
                draw.line((width - margin, y, width - margin - 20, y), fill=color, width=2)
            
            random.seed(style)
            for _ in range(5):
                shape_type = random.choice(["rect", "circle", "line"])
                sx = random.randint(min_x, max_x - 300)
                sy = random.randint(min_y, max_y - 300)
                sw = random.randint(100, 300)
                sh = random.randint(100, 300)
                if shape_type == "rect":
                    draw.rectangle((sx, sy, sx + sw, sy + sh), outline=color, width=2)
                elif shape_type == "circle":
                    draw.ellipse((sx, sy, sx + sw, sy + sw), outline=color, width=2)
                elif shape_type == "line":
                    draw.line((sx, sy, sx + sw, sy + sh), fill=color, width=2)

    def get_rotated_obb(self, cx, cy, w, h, angle):
        """Calculates the 4 corner points of a bounding box rotated by angle (in degrees, counter-clockwise)."""
        rad = math.radians(angle)
        cos_a = math.cos(rad)
        sin_a = math.sin(rad)
        
        # Local coordinates relative to center (Clockwise order: TL, TR, BR, BL)
        local_pts = [
            (-w / 2, -h / 2), # TL
            (w / 2, -h / 2),  # TR
            (w / 2, h / 2),   # BR
            (-w / 2, h / 2)   # BL
        ]
        
        world_pts = []
        for x, y in local_pts:
            rx = cx + x * cos_a + y * sin_a
            ry = cy - x * sin_a + y * cos_a
            world_pts.append((rx, ry))
            
        return world_pts

    def check_collision(self, bbox_list, new_bbox):
        """Checks if a new axis-aligned bounding box overlaps with existing bounding boxes.
        bbox format: (xmin, ymin, xmax, ymax) with some extra padding/buffer.
        """
        nx_min, ny_min, nx_max, ny_max = new_bbox
        padding = 15
        nx_min -= padding
        ny_min -= padding
        nx_max += padding
        ny_max += padding
        
        for ex_min, ey_min, ex_max, ey_max in bbox_list:
            if not (nx_max < ex_min or nx_min > ex_max or ny_max < ey_min or ny_min > ey_max):
                return True
        return False

    def generate_equation_text(self):
        """Generates a realistic CAD/blueprint engineering expression/equation string."""
        templates = [
            # Ra roughness
            lambda: f"Ra {random.choice([0.1, 0.2, 0.4, 0.8, 1.6, 3.2, 6.3, 12.5])}",
            # Rz roughness
            lambda: f"Rz {random.choice([1.6, 3.2, 6.3, 12.5, 25, 50, 100])}",
            # Diameter with tolerance
            lambda: f"Ø {random.randint(5, 120)} ± {random.choice([0.01, 0.02, 0.05, 0.1, 0.2])}",
            # Diameter with fit
            lambda: f"Ø {random.randint(5, 120)}{random.choice(['H7', 'g6', 'f7', 'js6', 'k6'])}",
            # Simple diameter
            lambda: f"Ø {random.randint(5, 120)}",
            # Radius
            lambda: f"R {random.randint(2, 50)}",
            # Small diameter/hole
            lambda: f"ø {random.randint(2, 20)} ± {random.choice([0.05, 0.1, 0.2])}",
            # Small diameter fit
            lambda: f"ø {random.randint(2, 20)}{random.choice(['H7', 'h6', 'js7'])}",
            # General dimension with tolerance
            lambda: f"{random.randint(10, 500)} ± {random.choice([0.05, 0.1, 0.15, 0.2, 0.5])}",
            # Parallelism tolerance
            lambda: f"II {random.choice([0.01, 0.02, 0.05, 0.1])} A",
            # Position tolerance
            lambda: f"o {random.choice([0.01, 0.02, 0.05, 0.1])} A B"
        ]
        return random.choice(templates)()

    def generate_full_page(self, bg_image_path, num_equations=20, num_distractors=15):
        """Loads a background, rotates it randomly, scatters non-overlapping equations and distractors,
        and returns the image + OBB coordinates of the equations.
        """
        bg = Image.open(bg_image_path)
        bg_rot = bg.rotate(random.choice([0, 90, 180, 270]), expand=True)
        width, height = bg_rot.size
        
        placed_aabbs = []
        labels = []
        
        margin = 60
        min_x, max_x = margin + 50, width - margin - 50
        min_y, max_y = margin + 50, height - margin - 220
        
        # 1. Place Equations (class 0)
        for _ in range(num_equations):
            expr = self.generate_equation_text()
            font_size = random.randint(28, 44)
            font = self.get_font(font_size)
            
            try:
                left, top, right, bottom = font.getbbox(expr)
                w = max(right - left, 1)
                h = max(bottom - top, 1)
            except AttributeError:
                w, h = len(expr) * font_size // 2, font_size
                
            txt_canvas = Image.new("RGBA", (w + 20, h + 20), (255, 255, 255, 0))
            txt_draw = ImageDraw.Draw(txt_canvas)
            txt_draw.text((10, 10), expr, font=font, fill=(0, 0, 0, 255))
            
            angle = random.uniform(0, 360)
            rotated_txt = txt_canvas.rotate(angle, expand=True, resample=Image.BICUBIC)
            rw, rh = rotated_txt.size
            
            for _ in range(100):
                cx = random.randint(min_x, max_x)
                cy = random.randint(min_y, max_y)
                
                xmin = cx - rw // 2
                ymin = cy - rh // 2
                xmax = cx + rw // 2
                ymax = cy + rh // 2
                
                if xmin < min_x or xmax > max_x or ymin < min_y or ymax > max_y:
                    continue
                    
                new_aabb = (xmin, ymin, xmax, ymax)
                if not self.check_collision(placed_aabbs, new_aabb):
                    bg_rot.paste(rotated_txt, (xmin, ymin), mask=rotated_txt)
                    placed_aabbs.append(new_aabb)
                    
                    obb_pts = self.get_rotated_obb(cx, cy, w + 20, h + 20, angle)
                    norm_pts = []
                    for px, py in obb_pts:
                        norm_pts.append(px / width)
                        norm_pts.append(py / height)
                        
                    labels.append((0, norm_pts)) # Class is 0 (equation)
                    break
                    
        # 2. Place distractors
        distractor_patterns = [
            lambda: f"{random.randint(1, 150)}",
            lambda: f"{random.randint(1, 99)}.{random.choice(['0', '5', '25', '75', '05', '12', '8'])}",
            lambda: f"X {random.randint(2, 6)}",
            lambda: f"REF {random.randint(10, 99)}",
            lambda: f"{random.randint(1, 5)}x {random.randint(10, 45)}°"
        ]
        
        for _ in range(num_distractors):
            dist_str = random.choice(distractor_patterns)()
            font_size = random.randint(22, 36)
            font = self.get_font(font_size)
            
            try:
                left, top, right, bottom = font.getbbox(dist_str)
                w = max(right - left, 1)
                h = max(bottom - top, 1)
            except AttributeError:
                w, h = len(dist_str) * font_size // 2, font_size
                
            txt_canvas = Image.new("RGBA", (w + 10, h + 10), (255, 255, 255, 0))
            txt_draw = ImageDraw.Draw(txt_canvas)
            txt_draw.text((5, 5), dist_str, font=font, fill=(0, 0, 0, 255))
            
            angle = random.uniform(0, 360)
            rotated_txt = txt_canvas.rotate(angle, expand=True, resample=Image.BICUBIC)
            rw, rh = rotated_txt.size
            
            for _ in range(50):
                cx = random.randint(min_x, max_x)
                cy = random.randint(min_y, max_y)
                
                xmin = cx - rw // 2
                ymin = cy - rh // 2
                xmax = cx + rw // 2
                ymax = cy + rh // 2
                
                if xmin < min_x or xmax > max_x or ymin < min_y or ymax > max_y:
                    continue
                    
                new_aabb = (xmin, ymin, xmax, ymax)
                if not self.check_collision(placed_aabbs, new_aabb):
                    bg_rot.paste(rotated_txt, (xmin, ymin), mask=rotated_txt)
                    placed_aabbs.append(new_aabb)
                    break
                    
        # 3. Add Gaussian noise to entire page
        np_img = np.array(bg_rot, dtype=np.float32)
        noise = np.random.normal(0, random.uniform(3, 8), np_img.shape)
        np_img = np.clip(np_img + noise, 0, 255).astype(np.uint8)
        
        return Image.fromarray(np_img), labels
    
    def save_dataset(self, pdf_dir="pdfs", yolo_dir="dataset_yolo"):
        """Generates the dataset:
        1. Generates backgrounds.
        2. Generates 600 labeled page images (split into 400 train, 150 val, 50 test).
        3. Compiles the pages into 200 PDFs of 3 pages each.
        4. Writes the YAML config for YOLO training.
        """
        bg_dir = "temp_backgrounds"
        self.generate_backgrounds(bg_dir, count=20)
        
        print("Generating YOLO OBB & PDF dataset...")
        os.makedirs(pdf_dir, exist_ok=True)
        
        splits = {
            "train": (0, 400),
            "val": (400, 550),
            "test": (550, 600)
        }
        
        for name, (start, end) in splits.items():
            os.makedirs(os.path.join(yolo_dir, "images", name), exist_ok=True)
            os.makedirs(os.path.join(yolo_dir, "labels", name), exist_ok=True)
            
        all_pages = []
        backgrounds = [os.path.join(bg_dir, f"bg_{i}.png") for i in range(20)]
        
        for page_idx in range(600):
            bg_path = backgrounds[page_idx % len(backgrounds)]
            page_img, labels = self.generate_full_page(bg_path, num_equations=random.randint(15, 25), num_distractors=15)
            all_pages.append((page_img, labels))
            if (page_idx + 1) % 50 == 0 or page_idx == 0:
                print(f"Generated page {page_idx + 1}/600")
            
        # Save images/labels into YOLO splits
        for split_name, (start, end) in splits.items():
            print(f"Saving images/labels for split: {split_name}...")
            for idx in range(start, end):
                page_img, labels = all_pages[idx]
                
                # Save image
                img_name = f"page_{idx}.png"
                img_path = os.path.join(yolo_dir, "images", split_name, img_name)
                page_img.save(img_path)
                
                # Save YOLO OBB label format
                label_name = f"page_{idx}.txt"
                label_path = os.path.join(yolo_dir, "labels", split_name, label_name)
                
                with open(label_path, "w", encoding="utf-8") as f:
                    for class_idx, pts in labels:
                        pts_str = " ".join([f"{p:.6f}" for p in pts])
                        f.write(f"{class_idx} {pts_str}\n")
                        
        # Save 200 PDFs, each compiling 3 pages
        print("Compiling 200 3-page PDFs...")
        for pdf_idx in range(200):
            p1 = all_pages[pdf_idx * 3][0]
            p2 = all_pages[pdf_idx * 3 + 1][0]
            p3 = all_pages[pdf_idx * 3 + 2][0]
            
            pdf_path = os.path.join(pdf_dir, f"blueprint_{pdf_idx}.pdf")
            p1.save(pdf_path, save_all=True, append_images=[p2, p3], format="PDF")
            
        # Write data.yaml for YOLOv8 OBB
        data_yaml_content = f"""path: {os.path.abspath(yolo_dir)}
train: images/train
val: images/val
test: images/test

names:
  0: equation
"""
        with open("data.yaml", "w", encoding="utf-8") as f:
            f.write(data_yaml_content)
            
        print("Data generation complete! 600 pages generated and 200 PDFs are ready.")

if __name__ == "__main__":
    generator = SyntheticDataGenerator()
    generator.save_dataset()
