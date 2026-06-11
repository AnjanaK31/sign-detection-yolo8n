import os
import random
import math
import numpy as np
import cv2
from PIL import Image, ImageDraw, ImageFont
from tqdm import tqdm

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

def draw_symbol_geometrically(draw, symbol_name, cx, cy, size, color=(0, 0, 0, 255), thickness=3):
    """Draws geometric control and tolerance symbols as vectors to prevent font missing glyph boxes."""
    w = size
    h = size
    
    if symbol_name == 'perpendicular':
        # ⊥ (Perpendicularity)
        # Vertical line
        draw.line((cx, cy - h/2, cx, cy + h/3), fill=color, width=thickness)
        # Horizontal base line
        draw.line((cx - w/2, cy + h/3, cx + w/2, cy + h/3), fill=color, width=thickness)
        
    elif symbol_name == 'parallel':
        # ∥ (Parallelism - two sloped parallel lines)
        slope_offset = w / 5
        draw.line((cx - w/4 + slope_offset, cy - h/2, cx - w/4 - slope_offset, cy + h/2), fill=color, width=thickness)
        draw.line((cx + w/4 + slope_offset, cy - h/2, cx + w/4 - slope_offset, cy + h/2), fill=color, width=thickness)
        
    elif symbol_name == 'circularity':
        # ○ (Circularity - empty circle)
        r = w / 2 * 0.8
        draw.ellipse((cx - r, cy - r, cx + r, cy + r), outline=color, width=thickness)
        
    elif symbol_name == 'diameter':
        # ⌀ (Diameter - oval with a diagonal slash)
        rx = w / 2 * 0.55
        ry = h / 2 * 0.8
        draw.ellipse((cx - rx, cy - ry, cx + rx, cy + ry), outline=color, width=thickness)
        # Diagonal slash from bottom-left to top-right
        slash_len_x = rx * 1.3
        slash_len_y = ry * 1.3
        draw.line((cx - slash_len_x, cy + slash_len_y, cx + slash_len_x, cy - slash_len_y), fill=color, width=thickness)
        
    elif symbol_name == 'true_position':
        # ⌀ (True Position - circle with a diagonal slash)
        r = w / 2 * 0.75
        draw.ellipse((cx - r, cy - r, cx + r, cy + r), outline=color, width=thickness)
        # Diagonal slash from bottom-left to top-right
        slash_len = r * 1.3
        draw.line((cx - slash_len, cy + slash_len, cx + slash_len, cy - slash_len), fill=color, width=thickness)
        
    elif symbol_name == 'plus_minus':
        # ± (Plus-Minus)
        plus_w = w * 0.4
        plus_h = h * 0.4
        # Plus horizontal
        draw.line((cx - plus_w/2, cy - h/3, cx + plus_w/2, cy - h/3), fill=color, width=thickness)
        # Plus vertical
        draw.line((cx, cy - h/3 - plus_h/2, cx, cy - h/3 + plus_h/2), fill=color, width=thickness)
        # Minus horizontal
        draw.line((cx - plus_w/2, cy + h/4, cx + plus_w/2, cy + h/4), fill=color, width=thickness)

class SyntheticDataGenerator:
    def __init__(self, base_dir="."):
        self.base_dir = base_dir
        
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

    def apply_adaptive_threshold(self, img_pil):
        """Phase 1: Applies Grayscale and Adaptive Thresholding to ensure high contrast."""
        img_np = np.array(img_pil.convert("RGB"))
        gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
        thresh = cv2.adaptiveThreshold(
            gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2
        )
        return Image.fromarray(cv2.cvtColor(thresh, cv2.COLOR_GRAY2RGB))

    def generate_backgrounds(self, output_dir, count=10):
        """Generates 'count' distinct, text-free engineering/blueprint background images."""
        os.makedirs(output_dir, exist_ok=True)
        width, height = 1600, 2000
        
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

    def _draw_cad_shapes(self, draw, width, height, style, margin):
        color = (120, 120, 120)
        thin_color = (200, 200, 200)
        
        min_x, max_x = margin + 50, width - margin - 50
        min_y, max_y = margin + 50, height - margin - 200
        
        if style % 5 == 0:
            grid_size = 100
            for x in range(min_x, max_x, grid_size):
                draw.line((x, min_y, x, max_y), fill=thin_color, width=1)
            for y in range(min_y, max_y, grid_size):
                draw.line((min_x, y, max_x, y), fill=thin_color, width=1)
        elif style % 5 == 1:
            cx, cy = (min_x + max_x) // 2, (min_y + max_y) // 2
            for r in [150, 300, 450]:
                draw.ellipse((cx - r, cy - r, cx + r, cy + r), outline=color, width=2)
            draw.line((min_x, cy, max_x, cy), fill=color, width=1)
            draw.line((cx, min_y, cx, max_y), fill=color, width=1)
        elif style % 5 == 2:
            cy = (min_y + max_y) // 2
            draw.rectangle((min_x + 100, cy - 100, max_x - 100, cy + 100), outline=color, width=3)
            draw.line((min_x + 50, cy, max_x - 50, cy), fill=color, width=1)
        elif style % 5 == 3:
            y_bus1, y_bus2 = min_y + 200, max_y - 200
            draw.line((min_x, y_bus1, max_x, y_bus1), fill=color, width=3)
            draw.line((min_x, y_bus2, max_x, y_bus2), fill=color, width=3)
            rung_spacing = (max_x - min_x) // 5
            for col in range(1, 5):
                x = min_x + col * rung_spacing
                draw.line((x, y_bus1, x, y_bus2), fill=color, width=2)
        else:
            draw.rectangle((min_x, min_y, max_x, max_y), outline=color, width=4)
            draw.rectangle((min_x + 20, min_y + 20, max_x - 20, max_y - 20), outline=color, width=2)

    def get_rotated_obb(self, cx, cy, w, h, angle):
        """Calculates the 4 corner points of a bounding box rotated by angle (in degrees, counter-clockwise)."""
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
        padding = 10
        nx_min -= padding
        ny_min -= padding
        nx_max += padding
        ny_max += padding
        
        for ex_min, ey_min, ex_max, ey_max in bbox_list:
            if not (nx_max < ex_min or nx_min > ex_max or ny_max < ey_min or ny_min > ey_max):
                return True
        return False

    def draw_arrowhead(self, draw, cx, cy, size, angle_deg, color=(0, 0, 0)):
        """Draws a solid filled arrowhead at (cx, cy) pointing at angle_deg."""
        rad = math.radians(angle_deg)
        pts = [
            (size / 2, 0),
            (-size / 2, -size / 3),
            (-size / 2, size / 3)
        ]
        cos_a = math.cos(rad)
        sin_a = math.sin(rad)
        world_pts = []
        for x, y in pts:
            rx = cx + x * cos_a + y * sin_a
            ry = cy - x * sin_a + y * cos_a
            world_pts.append((rx, ry))
        draw.polygon(world_pts, fill=color)
        return world_pts

    def generate_single_char_image(self, class_name, img_size=64):
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

    def generate_classifier_dataset(self, output_dir="dataset_classifier", train_count=150, val_count=40):
        """Generates synthetic character/symbol crops for training the MobileNetV3 model."""
        if os.path.exists(os.path.join(output_dir, "train", "0")) and len(os.listdir(os.path.join(output_dir, "train", "0"))) > 0:
            print("Classifier dataset already exists! Skipping generation.")
            return
            
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
        """Generates a complete blueprint page image and YOLO-OBB labels for characters/symbols."""
        bg = Image.open(bg_image_path)
        width, height = bg.size
        
        page_img = bg.copy()
        draw = ImageDraw.Draw(page_img)
        
        placed_aabbs = []
        labels = []
        
        margin = 60
        min_x, max_x = margin + 100, width - margin - 100
        min_y, max_y = margin + 100, height - margin - 250
        
        templates = [
            lambda: {
                "chars": ["diameter"] + list(str(random.randint(10, 99))) + ["plus_minus", "0", "comma", str(random.randint(1, 9))],
                "has_leader": True
            },
            lambda: {
                "chars": [random.choice(["Ra", "Rz"]), " "] + list(f"{random.randint(0, 12)},{random.choice([0,1,2,5,8])}"),
                "has_leader": False
            },
            lambda: {
                "chars": [random.choice(["perpendicular", "parallel", "circularity", "true_position"]), " "] + list(f"0,0{random.randint(1,9)}"),
                "has_leader": True
            }
        ]
        
        for _ in range(num_annotations):
            tpl = random.choice(templates)()
            chars = tpl["chars"]
            
            font_size = random.randint(28, 40)
            font = self.get_font(font_size)
            
            line_angle = random.choice([0, 90, 180, 270]) + random.uniform(-10, 10)
            rad = math.radians(line_angle)
            cos_a = math.cos(rad)
            sin_a = math.sin(rad)
            
            char_widths = []
            char_heights = []
            for c in chars:
                if c == " ":
                    char_widths.append(font_size // 2)
                    char_heights.append(font_size)
                elif c in CLASS_TO_CHAR:
                    char_str = CLASS_TO_CHAR[c]
                    try:
                        left, top, right, bottom = font.getbbox(char_str)
                        char_widths.append(max(right - left, 10))
                        char_heights.append(max(bottom - top, 20))
                    except AttributeError:
                        char_widths.append(font_size // 2)
                        char_heights.append(font_size)
                else:
                    char_widths.append(font_size)
                    char_heights.append(font_size)
                    
            total_w = sum(char_widths)
            max_h = max(char_heights)
            
            placed = False
            for _ in range(50):
                cx = random.randint(min_x, max_x)
                cy = random.randint(min_y, max_y)
                
                r_diag = max(total_w, max_h)
                xmin, xmax = cx - r_diag, cx + r_diag
                ymin, ymax = cy - r_diag, cy + r_diag
                
                if xmin < min_x or xmax > max_x or ymin < min_y or ymax > max_y:
                    continue
                    
                if not self.check_collision(placed_aabbs, (xmin, ymin, xmax, ymax)):
                    placed_aabbs.append((xmin, ymin, xmax, ymax))
                    placed = True
                    break
                    
            if not placed:
                continue
                
            current_x = -total_w / 2
            for idx, c in enumerate(chars):
                w_c = char_widths[idx]
                h_c = char_heights[idx]
                
                ccx = cx + (current_x + w_c / 2) * cos_a
                ccy = cy - (current_x + w_c / 2) * sin_a
                
                if c != " ":
                    canvas_size = int(max(w_c, h_c) * 2)
                    char_canvas = Image.new("RGBA", (canvas_size, canvas_size), (255, 255, 255, 0))
                    char_draw = ImageDraw.Draw(char_canvas)
                    
                    if c in CLASS_TO_CHAR:
                        char_str = CLASS_TO_CHAR[c]
                        char_draw.text((canvas_size // 2 - w_c // 2, canvas_size // 2 - h_c // 2), char_str, font=font, fill=(0, 0, 0, 255))
                    else:
                        symbol_size = min(w_c, h_c)
                        thickness = max(2, symbol_size // 12)
                        draw_symbol_geometrically(char_draw, c, canvas_size // 2, canvas_size // 2, symbol_size, color=(0, 0, 0, 255), thickness=thickness)
                        
                    rotated_char = char_canvas.rotate(line_angle, expand=True, resample=Image.BICUBIC)
                    rw, rh = rotated_char.size
                    
                    paste_x = int(ccx - rw / 2)
                    paste_y = int(ccy - rh / 2)
                    page_img.paste(rotated_char, (paste_x, paste_y), mask=rotated_char)
                    
                current_x += w_c
                
            # Generate a single bounding box for the entire expression
            # The expression is centered at cx, cy, with size total_w x max_h
            expr_obb_pts = self.get_rotated_obb(cx, cy, total_w, max_h, line_angle)
            norm_pts = []
            for px, py in expr_obb_pts:
                norm_pts.append(px / width)
                norm_pts.append(py / height)
            labels.append((0, norm_pts))
                
        preprocessed_page = self.apply_adaptive_threshold(page_img)
        return preprocessed_page, labels
        
    def save_dataset(self, pdf_dir="pdfs", yolo_dir="dataset_yolo"):
        """Compiles the full YOLO dataset (images and OBB labels) and PDFs."""
        bg_dir = "temp_backgrounds"
        if not os.path.exists(bg_dir):
            self.generate_backgrounds(bg_dir, count=10)
        
        print("Checking YOLO OBB dataset status...")
        os.makedirs(pdf_dir, exist_ok=True)
        
        splits = {
            "train": (0, 4000),
            "val": (4000, 4800),
            "test": (4800, 5000)
        }
        
        import concurrent.futures
        import json
        
        gt_path = os.path.join(yolo_dir, "ground_truth.json")
        pdf_pages_map = {}
        
        if os.path.exists(gt_path):
            print("✅ YOLO dataset images already exist! Skipping image generation...")
            for split_name in splits.keys():
                split_dir = os.path.join(yolo_dir, "images", split_name)
                if not os.path.exists(split_dir): continue
                for img_name in os.listdir(split_dir):
                    if img_name.endswith(".png"):
                        parts = img_name.replace(".png", "").split("_")
                        pdf_idx = int(parts[1])
                        page_in_pdf = int(parts[3])
                        if pdf_idx not in pdf_pages_map:
                            pdf_pages_map[pdf_idx] = {}
                        pdf_pages_map[pdf_idx][page_in_pdf] = os.path.join(split_dir, img_name)
        else:
            for name, (start, end) in splits.items():
                os.makedirs(os.path.join(yolo_dir, "images", name), exist_ok=True)
                os.makedirs(os.path.join(yolo_dir, "labels", name), exist_ok=True)
            
        backgrounds = [os.path.join(bg_dir, f"bg_{i}.png") for i in range(10)]
        
        print("Generating YOLO OBB dataset with Multiprocessing...")
        
        tasks = []
        for idx in range(5000):
            bg_path = backgrounds[idx % len(backgrounds)]
            if idx < splits["train"][1]:
                split_name = "train"
            elif idx < splits["val"][1]:
                split_name = "val"
            else:
                split_name = "test"
                
            pdf_idx = idx // 5
            page_in_pdf = (idx % 5) + 1
            tasks.append((idx, bg_path, split_name, pdf_idx, page_in_pdf, yolo_dir))
            
        ground_truth_db = {}
        
        import concurrent.futures
            
        pdf_pages_map = {}
        
        # Run multiprocessing pool
        max_workers = os.cpu_count() or 4
            print("🚀 Launching parallel workers for Image Generation...")
            with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
                futures = {executor.submit(process_page_worker, task): task for task in tasks}
                for future in tqdm(concurrent.futures.as_completed(futures), total=len(tasks), desc="Generating Pages"):
                    img_name, absolute_gt, pdf_idx, page_in_pdf, img_path = future.result()
                    ground_truth_db[img_name] = absolute_gt
                    
                    if pdf_idx not in pdf_pages_map:
                        pdf_pages_map[pdf_idx] = {}
                    pdf_pages_map[pdf_idx][page_in_pdf] = img_path
                    
            # Save GT JSON database
            with open(gt_path, "w") as f:
                json.dump(ground_truth_db, f, indent=2)
                
        print("Compiling sample 5-page PDFs with Multiprocessing...")
        pdf_tasks = []
        for pdf_idx in range(1000):
            # Check if PDF already exists
            pdf_path = os.path.join(pdf_dir, f"blueprint_{pdf_idx}.pdf")
            if not os.path.exists(pdf_path) and len(pdf_pages_map.get(pdf_idx, {})) == 5:
                pdf_tasks.append((pdf_idx, pdf_pages_map[pdf_idx], pdf_dir))
                
        if len(pdf_tasks) > 0:
            max_workers = os.cpu_count() or 4
            print(f"🚀 Launching {max_workers} parallel workers for PDF Compilation...")
            with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
                futures = {executor.submit(compile_pdf_worker, task): task for task in pdf_tasks}
                for future in tqdm(concurrent.futures.as_completed(futures), total=len(pdf_tasks), desc="Compiling PDFs"):
                    pass
        else:
            print("✅ All PDFs already compiled!")
            
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

# Top-level function to avoid multiprocessing pickling issues
def process_page_worker(task):
    idx, bg_path, split_name, pdf_idx, page_in_pdf, y_dir = task
    # Initialize local generator
    generator = SyntheticDataGenerator()
    page_img, labels = generator.generate_full_page(bg_path, num_annotations=15)
    width, height = page_img.size
    
    img_name = f"blueprint_{pdf_idx}_page_{page_in_pdf}.png"
    img_path = os.path.join(y_dir, "images", split_name, img_name)
    page_img.save(img_path)
    
    label_name = f"blueprint_{pdf_idx}_page_{page_in_pdf}.txt"
    label_path = os.path.join(y_dir, "labels", split_name, label_name)
    
    absolute_gt = []
    with open(label_path, "w", encoding="utf-8") as f:
        for class_idx, pts in labels:
            pts_str = " ".join([f"{p:.6f}" for p in pts])
            f.write(f"{class_idx} {pts_str}\n")
            
            abs_corners = [
                [pts[0] * width, pts[1] * height],
                [pts[2] * width, pts[3] * height],
                [pts[4] * width, pts[5] * height],
                [pts[6] * width, pts[7] * height]
            ]
            absolute_gt.append({
                "class": class_idx,
                "corners": abs_corners
            })
            
    return img_name, absolute_gt, pdf_idx, page_in_pdf, img_path

# Top-level function for multiprocessing PDF compilation
def compile_pdf_worker(task):
    pdf_idx, pages_map, pdf_dir = task
    p1 = Image.open(pages_map[1])
    p2 = Image.open(pages_map[2])
    p3 = Image.open(pages_map[3])
    p4 = Image.open(pages_map[4])
    p5 = Image.open(pages_map[5])
    pdf_path = os.path.join(pdf_dir, f"blueprint_{pdf_idx}.pdf")
    p1.save(pdf_path, save_all=True, append_images=[p2, p3, p4, p5], format="PDF")
    return pdf_idx

if __name__ == "__main__":
    generator = SyntheticDataGenerator()
    generator.generate_classifier_dataset()
    generator.save_dataset()
