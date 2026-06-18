import os
import random
import math
import numpy as np
import cv2
import json
import multiprocessing
from PIL import Image, ImageDraw, ImageFont

# 21 classes matching requirements
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
    """
    Worker task: generates a single page in a separate process and saves images/labels directly to disk.
    This bypasses pickling issues and keeps RAM usage extremely low.
    """
    page_idx, bg_path, num_expr, font_path, yolo_dir, split_name, width, height = task_args
    
    # Initialize process-local generator
    generator = SyntheticDataGenerator(font_path=font_path)
    
    page_img, labels, gt_details = generator.generate_full_page(bg_path, num_annotations=num_expr)
    
    # Save the page image
    img_name = f"page_{page_idx}.png"
    img_path = os.path.join(yolo_dir, "images", split_name, img_name)
    page_img.save(img_path)
    
    # Save the label coordinates
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
                "C:\\Windows\\Fonts\\tahoma.ttf",
                "C:\\Windows\\Fonts\\cour.ttf"
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
                else:
                    print("⚠️ No system font found. Downloading DejaVuSans.ttf for Unicode symbol support...")
                    try:
                        import urllib.request
                        url = "https://raw.githubusercontent.com/dejavu-fonts/dejavu-fonts/master/resources/DejaVuSans.ttf"
                        # Set a user-agent to bypass potential blocks
                        req = urllib.request.Request(
                            url, 
                            headers={'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64)'}
                        )
                        with urllib.request.urlopen(req) as response:
                            with open(local_font, 'wb') as out_file:
                                out_file.write(response.read())
                        self.font_path = local_font
                        print("✅ Successfully downloaded DejaVuSans.ttf")
                    except Exception as e:
                        print(f"❌ Failed to download font: {e}. Falling back to default Pillow font (which may render squares).")
        
        print(f"Using font: {self.font_path if self.font_path else 'Default Pillow Font'}")

    def get_font(self, size):
        if self.font_path:
            try:
                return ImageFont.truetype(self.font_path, size)
            except Exception:
                return ImageFont.load_default()
        return ImageFont.load_default()

    def apply_blueprint_effects(self, page_img):
        """Applies realistic scan and thresholding artifacts: dilation, erosion, and Gaussian noise."""
        img_np = np.array(page_img.convert("RGB"))
        gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
        
        # Binarize first via adaptive threshold with auto-inversion
        mean_val = np.mean(gray)
        if mean_val < 127:
            gray_input = cv2.bitwise_not(gray)
        else:
            gray_input = gray

        thresh = cv2.adaptiveThreshold(
            gray_input, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2
        )
        
        # Randomly erode or dilate white background (which thickens or thins black text strokes)
        kernel = np.ones((2, 2), np.uint8)
        if random.random() < 0.35:
            # Erode white background -> thickens black text (bold)
            thresh = cv2.erode(thresh, kernel, iterations=1)
        elif random.random() < 0.35:
            # Dilate white background -> thins black text (mimics scan drop-outs / broken strokes)
            thresh = cv2.dilate(thresh, kernel, iterations=1)
            
        # Add slight scanning paper noise
        if random.random() < 0.4:
            noise = np.random.normal(0, 12, thresh.shape).astype(np.float32)
            noisy = thresh.astype(np.float32) + noise
            thresh = np.clip(noisy, 0, 255).astype(np.uint8)
            
        return Image.fromarray(cv2.cvtColor(thresh, cv2.COLOR_GRAY2RGB))

    def generate_backgrounds(self, output_dir, count=10):
        """Generates text-free blueprint backgrounds."""
        os.makedirs(output_dir, exist_ok=True)
        width, height = 2400, 3000
        
        for i in range(count):
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
            
            self._draw_cad_shapes(draw, width, height, i, border_margin)
            bg.save(os.path.join(output_dir, f"bg_{i}.png"))
        
        print(f"Generated {count} text-free backgrounds in {output_dir}")

    def _draw_arrowhead(self, draw, x1, y1, x2, y2, color=(100, 100, 100), size=12):
        """Draws a filled arrowhead at (x2, y2) pointing in the direction of the line from x1,y1 to x2,y2."""
        dx = x2 - x1
        dy = y2 - y1
        length = math.hypot(dx, dy)
        if length == 0:
            return
        ux = dx / length
        uy = dy / length
        
        # Wing angles (150 degrees rotation)
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
        """Draws parallel diagonal hatching lines inside the rectangle (x0, y0, x1, y1)."""
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
        """Draws a horizontal dimension line with extension lines, arrowheads, and a label."""
        ext_y1 = y
        ext_y2 = y + offset_y
        
        # Extension lines
        draw.line((x1, ext_y1, x1, ext_y2 + (8 if offset_y > 0 else -8)), fill=color, width=1)
        draw.line((x2, ext_y1, x2, ext_y2 + (8 if offset_y > 0 else -8)), fill=color, width=1)
        
        # Dimension line
        dim_y = ext_y2
        draw.line((x1, dim_y, x2, dim_y), fill=color, width=1)
        
        # Arrowheads
        self._draw_arrowhead(draw, x1 + 15, dim_y, x1, dim_y, color=color, size=8)
        self._draw_arrowhead(draw, x2 - 15, dim_y, x2, dim_y, color=color, size=8)
        
        # Label text
        try:
            left, top, right, bottom = font.getbbox(label)
            tw = right - left
            th = bottom - top
        except Exception:
            tw = len(label) * 8
            th = 12
            
        tx = (x1 + x2) // 2 - tw // 2
        ty = dim_y - th - 3 if offset_y > 0 else dim_y + 3
        
        # Clear background under text to keep it readable
        draw.rectangle((tx - 4, ty - 2, tx + tw + 4, ty + th + 2), fill=(255, 255, 255))
        draw.text((tx, ty), label, font=font, fill=color)

    def _draw_cad_shapes(self, draw, width, height, style, margin):
        color = (120, 120, 120)
        thin_color = (180, 180, 180)
        very_thin_color = (220, 220, 220)
        
        min_x, max_x = margin + 50, width - margin - 50
        min_y, max_y = margin + 50, height - margin - 200
        cx, cy = (min_x + max_x) // 2, (min_y + max_y) // 2
        
        font = self.get_font(18)
        
        # Style-specific drawing base
        if style % 5 == 0:
            # Grid lines
            grid_size = 150
            for x in range(min_x, max_x, grid_size):
                draw.line((x, min_y, x, max_y), fill=very_thin_color, width=1)
            for y in range(min_y, max_y, grid_size):
                draw.line((min_x, y, max_x, y), fill=very_thin_color, width=1)
            
            # Simple outline of a flange plate
            draw.rectangle((cx - 300, cy - 300, cx + 300, cy + 300), outline=color, width=3)
            draw.ellipse((cx - 200, cy - 200, cx + 200, cy + 200), outline=color, width=2)
            # Centerlines
            draw.line((cx - 350, cy, cx + 350, cy), fill=thin_color, width=1)
            draw.line((cx, cy - 350, cx, cy + 350), fill=thin_color, width=1)
            
            # Dimension line clutter
            self._draw_dimension_line(draw, cx - 300, cx + 300, cy - 300, -50, "600", font, color=color)
            self._draw_dimension_line(draw, cx - 200, cx + 200, cy - 200, -30, "⌀ 400", font, color=color)
            
        elif style % 5 == 1:
            # Cylinder / Shaft drawing with cross-hatching sections
            draw.rectangle((cx - 150, cy - 400, cx + 150, cy + 400), outline=color, width=3)
            # Shoulders
            draw.rectangle((cx - 100, cy - 300, cx + 100, cy + 300), outline=color, width=2)
            draw.rectangle((cx - 60, cy - 200, cx + 60, cy + 200), outline=color, width=2)
            
            # Centerline
            draw.line((cx, cy - 450, cx, cy + 450), fill=thin_color, width=1)
            
            # Hatching sections on side walls
            self._draw_hatching(draw, cx - 150, cy - 400, cx - 100, cy + 400, spacing=20, color=very_thin_color)
            self._draw_hatching(draw, cx + 100, cy - 400, cx + 150, cy + 400, spacing=20, color=very_thin_color)
            
            # Dimensions
            self._draw_dimension_line(draw, cx - 150, cx + 150, cy - 400, -40, "300", font, color=color)
            self._draw_dimension_line(draw, cx - 100, cx + 100, cy - 300, -20, "200", font, color=color)
            
        elif style % 5 == 2:
            # Mechanical link arm
            draw.ellipse((cx - 120, cy - 250, cx + 120, cy - 50), outline=color, width=3)
            draw.ellipse((cx - 120, cy + 50, cx + 120, cy + 250), outline=color, width=3)
            # Connecting web
            draw.polygon([
                (cx - 60, cy - 150), (cx + 60, cy - 150),
                (cx + 60, cy + 150), (cx - 60, cy + 150)
            ], outline=color, width=2)
            
            # Holes in link ends
            draw.ellipse((cx - 50, cy - 150, cx + 50, cy - 50), outline=color, width=2)
            draw.ellipse((cx - 50, cy + 50, cx + 50, cy + 150), outline=color, width=2)
            
            # Web cross-hatching
            self._draw_hatching(draw, cx - 55, cy - 140, cx + 55, cy + 140, spacing=15, color=very_thin_color)
            
            # Dimensions
            self._draw_dimension_line(draw, cx - 120, cx + 120, cy - 250, -40, "240", font, color=color)
            self._draw_dimension_line(draw, cx - 50, cx + 50, cy - 150, -20, "⌀ 100", font, color=color)
            
        elif style % 5 == 3:
            # Mounting bracket / structural plate
            draw.polygon([
                (cx - 350, cy - 200), (cx + 350, cy - 200),
                (cx + 250, cy + 200), (cx - 250, cy + 200)
            ], outline=color, width=3)
            
            # Triangular holes
            draw.polygon([(cx - 200, cy - 100), (cx - 100, cy - 100), (cx - 150, cy + 100)], outline=color, width=2)
            draw.polygon([(cx + 100, cy - 100), (cx + 200, cy - 100), (cx + 150, cy + 100)], outline=color, width=2)
            
            # Centerlines
            draw.line((cx - 400, cy, cx + 400, cy), fill=thin_color, width=1)
            
            # Dimensions
            self._draw_dimension_line(draw, cx - 350, cx + 350, cy - 200, -50, "700", font, color=color)
            self._draw_dimension_line(draw, cx - 250, cx + 250, cy + 200, 50, "500", font, color=color)
            
        else:
            # Assembly drawing frame
            draw.rectangle((min_x + 50, min_y + 50, max_x - 50, max_y - 50), outline=color, width=3)
            draw.rectangle((min_x + 80, min_y + 80, max_x - 80, max_y - 80), outline=color, width=2)
            
            # Centerlines and layout grid lines
            for i in range(1, 4):
                offset_x = (max_x - min_x) // 4 * i
                offset_y = (max_y - min_y) // 4 * i
                draw.line((min_x + offset_x, min_y, min_x + offset_x, max_y), fill=very_thin_color, width=1)
                draw.line((min_x, min_y + offset_y, max_x, min_y + offset_y), fill=very_thin_color, width=1)
            
            # Clutter text and stamps
            draw.text((min_x + 100, min_y + 100), "DETAIL D", font=font, fill=color)
            draw.text((min_x + 100, min_y + 130), "SCALE 2:1", font=font, fill=color)
            draw.text((max_x - 250, min_y + 100), "REF ONLY", font=font, fill=color)

    def get_rotated_obb(self, cx, cy, w, h, angle):
        """Calculates 4 corners in TL, TR, BR, BL order rotated by angle (degrees, counter-clockwise)."""
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

    def generate_single_char_image(self, class_name, img_size=64):
        """Fallback function stub."""
        return Image.new("RGB", (img_size, img_size), (255, 255, 255))

    def generate_classifier_dataset(self, output_dir="dataset_classifier", train_count=10, val_count=2):
        """Fallback function stub."""
        pass

    def get_random_expression(self):
        """Generates a random engineering callout text using the target symbols with highly variable lengths."""
        templates = [
            # 1. Very short (1 to 5 chars)
            lambda: f"{random.choice(['⌀', 'R', 'Ra', 'Rz'])} {random.randint(1, 9)}",
            lambda: f"{random.choice(['>', '<', '≥'])} {random.choice(['0', '1', '5'])}",
            lambda: f"{random.choice(['90', '45', '30'])}°",
            lambda: f"+{random.choice(['0.01', '0.05', '0.1'])}",
            lambda: f"-{random.choice(['0.02', '0.05', '0.2'])}",
            lambda: ",",
            
            # 2. Standard (6 to 20 chars)
            lambda: f"⌀ {random.randint(5, 120)}{random.choice(['.5', '.0', ''])} ± {random.choice(['0.02', '0.05', '0.1'])}",
            lambda: f"⌀ {random.randint(5, 60)} +{random.choice(['0.05', '0.1', '0.2'])}",
            lambda: f"R {random.randint(2, 50)}{random.choice(['.0', ''])} ± {random.choice(['0.1', '0.2', '0.5'])}",
            lambda: f"{random.choice([15, 30, 45, 60, 90])}° ± {random.choice(['0.5°', '1°'])}",
            lambda: f"M{random.choice([4, 5, 6, 8, 10, 12, 16])} x {random.choice(['0.7', '0.8', '1.0', '1.25', '1.5', '2.0'])}",
            lambda: f"M{random.choice([8, 10, 12, 16])} x {random.choice(['1.25', '1.5', '2.0'])} - {random.choice(['6g', '6H'])}",
            lambda: f"{random.choice(['Ra', 'Rz'])} {random.choice(['0.8', '1.6', '3.2', '6.3', '12.5'])} μm",
            lambda: f"{random.randint(2, 20)}{random.choice(['.0', '.5', ''])}″ × {random.randint(5, 50)}{random.choice(['.0', '.5', ''])}″",
            lambda: f"{random.choice(['>', '<', '≥'])} {random.choice(['0.05', '0.1', '1.0', '10.0', '25.0'])}",
            
            # 3. Very long (20 to 45 chars)
            lambda: f"⌀ {random.randint(20, 150)}.0 +{random.choice(['0.02', '0.05'])} / -{random.choice(['0.01', '0.03'])} {random.choice(['(THRU)', '(THRU ALL)', '(2x HOLES)'])}",
            lambda: f"M{random.choice([10, 12, 16, 20])} x {random.choice(['1.5', '2.0'])} - {random.choice(['6g', '6H'])} - {random.choice(['20 DEEP', '30 DEEP', 'THRU'])}",
            lambda: f"{random.randint(1, 10)}.5″ × {random.randint(5, 40)}.0″ × {random.choice(['0.25″', '0.50″', '0.12″'])} {random.choice(['MAX', 'MIN', 'TYP'])}",
            lambda: f"R {random.randint(10, 50)}.0 ± {random.choice(['0.2', '0.5'])} {random.choice(['(4x PLACES)', '(TYP)', '(MAX)'])}",
            lambda: f"⊥ {random.choice(['0.02', '0.05', '0.1'])} A",
            lambda: f"∥ {random.choice(['0.03', '0.05', '0.1'])} A B",
            lambda: f"○ {random.choice(['0.01', '0.05', '0.08'])}",
            lambda: f"⊕ ⌀ {random.choice(['0.05', '0.1', '0.2'])} A B C"
        ]
        return random.choice(templates)()
        """Generates a preprocessed 64x64 crop of a single class for classifier training."""
        img = Image.new("RGBA", (img_size * 2, img_size * 2), (255, 255, 255, 0))
        draw = ImageDraw.Draw(img)
        
        # Draw random background grid/CAD lines and circles to simulate blueprint clutter
        if random.random() < 0.5:
            num_lines = random.randint(1, 3)
            for _ in range(num_lines):
                lx1 = random.randint(0, img_size * 2)
                ly1 = random.randint(0, img_size * 2)
                lx2 = random.randint(0, img_size * 2)
                ly2 = random.randint(0, img_size * 2)
                draw.line((lx1, ly1, lx2, ly2), fill=(160, 160, 160, 255), width=random.randint(1, 2))
                
        if random.random() < 0.3:
            rx = random.randint(0, img_size * 2)
            ry = random.randint(0, img_size * 2)
            r = random.randint(10, 50)
            draw.ellipse((rx - r, ry - r, rx + r, ry + r), outline=(160, 160, 160, 255), width=random.randint(1, 2))
            
        angle = random.uniform(-15, 15) if class_name != 'arrow' else random.uniform(0, 360)
        ccx, ccy = img_size, img_size
        
        if class_name == 'arrow':
            arrow_size = random.randint(24, 36)
            self.draw_arrowhead(draw, ccx, ccy, arrow_size, angle, color=(0, 0, 0))
        elif class_name in ['perpendicular', 'parallel', 'circularity', 'diameter', 'true_position', 'plus_minus']:
            symbol_size = random.randint(28, 40)
            thickness = max(2, symbol_size // 12)
            draw_symbol_geometrically(draw, class_name, ccx, ccy, symbol_size, color=(0, 0, 0, 255), thickness=thickness)
            img = img.rotate(angle, expand=False, resample=Image.BICUBIC)
        else:
            char_str = CLASS_TO_CHAR[class_name]
            font_size = random.randint(28, 42)
            font = self.get_font(font_size)
            
            try:
                left, top, right, bottom = font.getbbox(char_str)
                w = right - left
                h = bottom - top
            except AttributeError:
                w = font_size // 2
                h = font_size
                
            draw.text((ccx - w // 2, ccy - h // 2), char_str, font=font, fill=(0, 0, 0, 255))
            img = img.rotate(angle, expand=False, resample=Image.BICUBIC)
            
        final_img = Image.new("RGB", (img_size, img_size), (255, 255, 255))
        offset_x = random.randint(-4, 4)
        offset_y = random.randint(-4, 4)
        final_img.paste(img, (-img_size // 2 + offset_x, -img_size // 2 + offset_y), mask=img)
        
        return self.apply_adaptive_threshold(final_img)

    def generate_classifier_dataset(self, output_dir="dataset_classifier", train_count=500, val_count=100):
        """Generates synthetic character/symbol crops for training the MobileNetV3 model."""
        print(f"Generating classifier dataset ({train_count} train / {val_count} val samples per class)...")
        
        for name in ["train", "val"]:
            for c in CLASSES:
                os.makedirs(os.path.join(output_dir, name, c), exist_ok=True)
                
        for c in CLASSES:
            for i in range(train_count):
                img = self.generate_single_char_image(c)
                img.save(os.path.join(output_dir, "train", c, f"img_{i}.png"))
            for i in range(val_count):
                img = self.generate_single_char_image(c)
                img.save(os.path.join(output_dir, "val", c, f"img_{i}.png"))
                
        print("Classifier dataset generation complete!")

    def generate_full_page(self, bg_image_path, num_annotations=15):
        """Generates a full page by rendering expressions dynamically with proper AABB offsets and fonts."""
        bg = Image.open(bg_image_path)
        width, height = bg.size
        
        page_img = bg.copy()
        draw = ImageDraw.Draw(page_img, "RGBA")
        
        placed_aabbs = []
        labels = []
        gt_details = []
        
        margin = 60
        min_x, max_x = margin + 120, width - margin - 120
        min_y, max_y = margin + 120, height - margin - 250
        
        for _ in range(num_annotations):
            # Select 1 to 3 stacked lines
            n_lines = random.choice([1, 2, 3])
            
            # Draw and measure each line in the block
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
                
                # Draw text onto a transparent RGBA line canvas
                pad_canvas = int(max(tw, th) * 2)
                txt_canvas = Image.new("RGBA", (pad_canvas, pad_canvas), (255, 255, 255, 0))
                txt_draw = ImageDraw.Draw(txt_canvas)
                txt_draw.text((pad_canvas // 2 - tw // 2, pad_canvas // 2 - th // 2), text, font=font, fill=(0, 0, 0, 255))
                
                processed_lines.append({
                    "canvas": txt_canvas,
                    "text": text,
                    "w": tw,
                    "h": th,
                })
                
            # Compute block specifications
            line_spacing = max(line["h"] for line in processed_lines) + 16
            block_h = n_lines * line_spacing
            block_w = max(line["w"] for line in processed_lines)
            
            # Select baseline rotation (70% standard with wobble, 30% arbitrary)
            if random.random() < 0.7:
                line_angle = random.choice([0, 90, 180, 270]) + random.uniform(-5, 5)
            else:
                line_angle = random.uniform(0, 360)
                
            # Wrap to [0, 360)
            line_angle = line_angle % 360
            
            rad = math.radians(line_angle)
            cos_a = math.cos(rad)
            sin_a = math.sin(rad)
            
            # Find collision-free place
            placed = False
            for _ in range(300):
                cx = random.randint(min_x, max_x)
                cy = random.randint(min_y, max_y)
                
                # Bounding box of the entire block (AABB check based on line_angle)
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
                
            # First, calculate all drawing details and OBBs for the block to avoid render order conflicts
            lines_to_draw = []
            for idx, line in enumerate(processed_lines):
                # Local vertical offset (perpendicular to line direction)
                dy_local = (idx - (n_lines - 1) / 2) * line_spacing
                
                # World center of this line crop
                ccx = cx - dy_local * sin_a
                ccy = cy + dy_local * cos_a
                
                line_w = line["w"]
                line_h = line["h"]
                
                # Bounding box with margins of padding around expression for noise-free detection
                pad_w = 12
                pad_h = 6
                line_obb = self.get_rotated_obb(ccx, ccy, line_w + pad_w, line_h + pad_h, line_angle)
                
                # Rotate and prepare the transparent text crop
                rotated_txt = line["canvas"].rotate(line_angle, expand=True, resample=Image.Resampling.BICUBIC)
                rw, rh = rotated_txt.size
                
                paste_x = int(ccx - rw / 2)
                paste_y = int(ccy - rh / 2)
                
                lines_to_draw.append({
                    "obb": line_obb,
                    "rotated_txt": rotated_txt,
                    "paste_x": paste_x,
                    "paste_y": paste_y,
                    "line": line,
                    "line_obb": line_obb
                })
            
            # Step 1: Draw all background clearing white polygons first (so they don't overwrite any text)
            for item in lines_to_draw:
                draw.polygon([(float(pt[0]), float(pt[1])) for pt in item["obb"]], fill=(255, 255, 255, 255))
                
            # Step 2: Paste all the rotated text canvases on top
            for item in lines_to_draw:
                page_img.paste(item["rotated_txt"], (item["paste_x"], item["paste_y"]), mask=item["rotated_txt"])
                
                # Save YOLO OBB coordinate format
                norm_pts = []
                for px, py in item["line_obb"]:
                    norm_pts.append(px / width)
                    norm_pts.append(py / height)
                labels.append((0, norm_pts))
                
                # Save ground truth details
                gt_details.append({
                    "class": 0,
                    "corners": [[float(pt[0]), float(pt[1])] for pt in item["line_obb"]],
                    "text": item["line"]["text"],
                    "angle": float(line_angle)
                })
                
        preprocessed_page = self.apply_blueprint_effects(page_img)
        return preprocessed_page, labels, gt_details

    def save_dataset(self, num_pdfs=16, pages_per_pdf=5, min_expr=30, max_expr=50, pdf_dir="pdfs", yolo_dir="dataset_yolo", cores=None):
        """Compiles the full YOLO OBB dataset and outputs ground_truth.json using parallel processing."""
        bg_dir = "temp_backgrounds"
        self.generate_backgrounds(bg_dir, count=10)
        
        total_pages = num_pdfs * pages_per_pdf
        print(f"Generating {total_pages} total pages across {num_pdfs} PDFs (each having {pages_per_pdf} pages)...")
        os.makedirs(pdf_dir, exist_ok=True)
        
        # Calculate training, validation, and test split indices
        # Standard splits: 75% train, 15% val, 10% test
        train_end = max(1, int(round(0.75 * total_pages)))
        val_end = max(train_end + 1, train_end + int(round(0.15 * total_pages)))
        # clamp to bounds
        if val_end >= total_pages:
            val_end = total_pages - 1
            if val_end <= train_end:
                train_end = max(1, val_end - 1)
        
        splits = {
            "train": (0, train_end),
            "val": (train_end, val_end),
            "test": (val_end, total_pages)
        }
        
        print(f"Split distribution -> Train: {splits['train'][1] - splits['train'][0]} pages, "
              f"Val: {splits['val'][1] - splits['val'][0]} pages, "
              f"Test: {splits['test'][1] - splits['test'][0]} pages")
              
        for name, (start, end) in splits.items():
            os.makedirs(os.path.join(yolo_dir, "images", name), exist_ok=True)
            os.makedirs(os.path.join(yolo_dir, "labels", name), exist_ok=True)
            
        gt_database = {}
        backgrounds = [os.path.join(bg_dir, f"bg_{i}.png") for i in range(10)]
        
        # Determine number of processes/cores
        import multiprocessing
        if cores is None:
            cores = max(1, multiprocessing.cpu_count() - 1)
        print(f"Parallel processing enabled using {cores} CPU core(s)...")
        
        # Prepare task arguments
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
                2400, # width
                3000  # height
            ))
            
        # Run parallel tasks
        with multiprocessing.Pool(processes=cores) as pool:
            try:
                from tqdm import tqdm
                results = list(tqdm(pool.imap_unordered(worker_generate_and_save_page, task_args), total=len(task_args), desc="Generating Pages"))
            except ImportError:
                results = []
                for idx, res in enumerate(pool.imap_unordered(worker_generate_and_save_page, task_args)):
                    results.append(res)
                    if (idx + 1) % 10 == 0 or idx == 0 or idx == len(task_args) - 1:
                        print(f"Generated page {idx + 1}/{len(task_args)}")
                        
        # Gathers ground truth
        for img_name, gt_details in results:
            gt_database[img_name] = gt_details
                        
        # Save ground truth JSON database
        gt_json_path = os.path.join(yolo_dir, "ground_truth.json")
        with open(gt_json_path, "w", encoding="utf-8") as f:
            json.dump(gt_database, f, indent=2, ensure_ascii=False)
        print(f"Saved ground truth JSON database to: {gt_json_path}")
        
        print(f"Compiling {num_pdfs} multi-page PDFs ({pages_per_pdf} pages each)...")
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
                # Explicitly close images to free resources
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
    parser.add_argument("--num-pdfs", "-n", type=int, default=16, help="Number of PDFs to generate")
    parser.add_argument("--pages-per-pdf", "-p", type=int, default=5, help="Number of pages in each PDF")
    parser.add_argument("--min-expr", type=int, default=30, help="Minimum expressions per page")
    parser.add_argument("--max-expr", type=int, default=50, help="Maximum expressions per page")
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
