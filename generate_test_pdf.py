import os
import math
import random
import json
import numpy as np
import cv2
from PIL import Image, ImageDraw, ImageFont

# Path to fonts
FONT_CHOICES = [
    "C:\\Windows\\Fonts\\seguisym.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    "/usr/share/fonts/truetype/freefont/FreeSans.ttf",
    "DejaVuSans.ttf",
    "C:\\Windows\\Fonts\\arial.ttf",
    "C:\\Windows\\Fonts\\calibri.ttf",
    "C:\\Windows\\Fonts\\segoeui.ttf"
]

def load_font(size):
    for p in FONT_CHOICES:
        if os.path.exists(p):
            try:
                return ImageFont.truetype(p, size)
            except Exception:
                pass
    return ImageFont.load_default()

# 34 Meaningful engineering expressions using the target symbols
EXPRESSIONS = [
    "⌀ 12.5 ± 0.05",
    "⌀ 50.0 +0.02",
    "⌀ 6.3 -0.05",
    "M10 x 1.5 - 6g",
    "M12 x 1.75 - 6H",
    "M6 x 1.0",
    "R 5.0 ± 0.1",
    "R 15.0 ± 0.5",
    "Ra 3.2 μm",
    "Rz 12.5 μm",
    "Ra 1.6",
    "45° ± 0.5°",
    "90° ± 1°",
    "30° ± 0.2°",
    "12\" × 24\"",
    "6\" × 12\"",
    "2.5\" × 4.0\"",
    "> 10.0",
    "< 0.05",
    "≥ 25.0",
    "⌀ 20 +0.1 / -0.05",
    "M20 x 2.0 - 6g",
    "⌀ 8.0 H7",
    "R 2.0 MAX",
    "Rz 6.3 μm",
    "> 18.5 μm",
    "< 0.005",
    "15.5\" × 30.0\"",
    "⌀ 100 ± 0.15",
    "1.5\" × 3.0\" × 0.25\"",
    "+0.05",
    "+0.1 / -0.1",
    "R 10.0",
    "M8 x 1.25 - 6H"
]

def get_rotated_obb(cx, cy, w, h, angle):
    rad = math.radians(angle)
    cos_a = math.cos(rad)
    sin_a = math.sin(rad)
    
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

def check_collision(bbox_list, new_bbox):
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

def draw_arrowhead(draw, x1, y1, x2, y2, color=(100, 100, 100), size=12):
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

def draw_hatching(draw, x0, y0, x1, y1, spacing=16, color=(210, 210, 210), width=1):
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

def draw_dimension_line(draw, x1, x2, y, offset_y, label, font, color=(100, 100, 100)):
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
    draw_arrowhead(draw, x1 + 15, dim_y, x1, dim_y, color=color, size=8)
    draw_arrowhead(draw, x2 - 15, dim_y, x2, dim_y, color=color, size=8)
    
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

def generate_blueprint_page():
    width, height = 1600, 2000
    page_img = Image.new("RGB", (width, height), (255, 255, 255))
    draw = ImageDraw.Draw(page_img, "RGBA")
    
    # Draw standard CAD border
    border = 40
    color = (120, 120, 120)
    thin_color = (180, 180, 180)
    very_thin_color = (220, 220, 220)
    
    draw.rectangle((border, border, width - border, height - border), outline=color, width=4)
    
    # Concentric circles in center
    cx, cy = width // 2, height // 2
    for r in [150, 300, 450]:
        draw.ellipse((cx - r, cy - r, cx + r, cy + r), outline=color, width=2)
    draw.line((border + 50, cy, width - border - 50, cy), fill=thin_color, width=1)
    draw.line((cx, border + 50, cx, height - border - 200), fill=thin_color, width=1)
    
    # Draw some grid lines
    grid_size = 200
    for x in range(border + 100, width - border, grid_size):
        draw.line((x, border + 50, x, height - border - 200), fill=very_thin_color, width=1)
    for y in range(border + 100, height - border - 200, grid_size):
        draw.line((border + 50, y, width - border - 50, y), fill=very_thin_color, width=1)
        
    # Draw details: some shaft shoulders on left/right of center
    draw.rectangle((cx - 400, cy - 100, cx + 400, cy + 100), outline=color, width=2)
    draw_hatching(draw, cx - 400, cy - 100, cx - 300, cy + 100, spacing=15, color=very_thin_color)
    draw_hatching(draw, cx + 300, cy - 100, cx + 400, cy + 100, spacing=15, color=very_thin_color)
    
    # Title box at bottom-right
    tb_w, tb_h = 400, 150
    tb_x = width - border - tb_w
    tb_y = height - border - tb_h
    draw.rectangle((tb_x, tb_y, width - border, height - border), outline=color, width=3)
    draw.line((tb_x, tb_y + 50, width - border, tb_y + 50), fill=color, width=2)
    draw.line((tb_x, tb_y + 100, width - border, tb_y + 100), fill=color, width=2)
    draw.line((tb_x + 150, tb_y, tb_x + 150, height - border), fill=color, width=2)
    draw.line((tb_x + 280, tb_y + 50, tb_x + 280, height - border), fill=color, width=2)
    
    # Detail box at bottom-left
    draw.rectangle((border + 50, height - border - 350, border + 450, height - border - 50), outline=color, width=2)
    
    # Standard dimensions
    font_dim = load_font(18)
    draw_dimension_line(draw, cx - 400, cx + 400, cy - 100, -50, "800", font_dim, color=color)
    draw_dimension_line(draw, cx - 300, cx + 300, cy - 100, -30, "600", font_dim, color=color)
    
    # Non-target dimension labels
    draw.text((border + 70, height - border - 330), "DETAIL A", font=font_dim, fill=color)
    draw.text((border + 70, height - border - 300), "SCALE 1:1", font=font_dim, fill=color)
    draw.text((tb_x + 10, tb_y + 15), "SHAPE NIPPLE ASSEMBLY", font=font_dim, fill=color)
    draw.text((tb_x + 10, tb_y + 65), "DRW NO: CN-021", font=font_dim, fill=color)
    draw.text((tb_x + 10, tb_y + 115), "REF: 12.5-30", font=font_dim, fill=color)
    
    placed_aabbs = []
    
    min_x, max_x = border + 100, width - border - 100
    min_y, max_y = border + 100, height - border - 250
    
    # Distribute the 34 expressions
    expressions_to_draw = []
    for idx, text in enumerate(EXPRESSIONS):
        font_size = random.randint(28, 52)
        font = load_font(font_size)
        
        # Get dimensions
        try:
            left, top, right, bottom = font.getbbox(text)
            tw = right - left
            th = bottom - top
        except AttributeError:
            tw = len(text) * (font_size // 2)
            th = font_size
            
        # Select rotation angle
        angle = random.choice([0, 90, 180, 270]) + random.uniform(-5, 5)
        
        # Find collision-free place
        placed = False
        for _ in range(300):
            ccx = random.randint(min_x, max_x)
            ccy = random.randint(min_y, max_y)
            
            # Bounding box of the expression (AABB check based on angle)
            half_w = (tw + 16) // 2
            half_h = (th + 16) // 2
            
            # If text is vertical, swap width and height for AABB check
            abs_ang = abs(angle)
            if (45 < abs_ang < 135) or (225 < abs_ang < 315):
                xmin, xmax = ccx - half_h, ccx + half_h
                ymin, ymax = ccy - half_w, ccy + half_w
            else:
                xmin, xmax = ccx - half_w, ccx + half_w
                ymin, ymax = ccy - half_h, ccy + half_h
            
            if xmin < min_x or xmax > max_x or ymin < min_y or ymax > max_y:
                continue
                
            if not check_collision(placed_aabbs, (xmin, ymin, xmax, ymax)):
                placed_aabbs.append((xmin, ymin, xmax, ymax))
                placed = True
                break
                
        if not placed:
            continue
        
        # Draw transparent text canvas
        pad_canvas = int(max(tw, th) * 2)
        txt_canvas = Image.new("RGBA", (pad_canvas, pad_canvas), (255, 255, 255, 0))
        txt_draw = ImageDraw.Draw(txt_canvas)
        
        # Draw text centered in canvas
        txt_draw.text((pad_canvas // 2 - tw // 2, pad_canvas // 2 - th // 2), text, font=font, fill=(0, 0, 0, 255))
        
        # Rotate text canvas
        rotated_txt = txt_canvas.rotate(angle, expand=True, resample=Image.Resampling.BICUBIC)
        rw, rh = rotated_txt.size
        
        # Calculate OBB with padding
        pad_w = 12
        pad_h = 6
        obb = get_rotated_obb(ccx, ccy, tw + pad_w, th + pad_h, angle)
        
        expressions_to_draw.append({
            "obb": obb,
            "rotated_txt": rotated_txt,
            "rw": rw,
            "rh": rh,
            "ccx": ccx,
            "ccy": ccy,
            "text": text,
            "angle": angle
        })
        
    # Step 1: Draw all background clearing white polygons first
    for item in expressions_to_draw:
        draw.polygon([(float(pt[0]), float(pt[1])) for pt in item["obb"]], fill=(255, 255, 255, 255))
        
    # Step 2: Paste all the rotated text canvases on top
    gt_details = []
    for item in expressions_to_draw:
        paste_x = int(item["ccx"] - item["rw"] / 2)
        paste_y = int(item["ccy"] - item["rh"] / 2)
        page_img.paste(item["rotated_txt"], (paste_x, paste_y), mask=item["rotated_txt"])
        
        # Save GT details
        gt_details.append({
            "class": 0,
            "corners": [[float(pt[0]), float(pt[1])] for pt in item["obb"]],
            "text": item["text"],
            "angle": float(item["angle"])
        })
        
    # Apply adaptive threshold to binarize drawing page
    preprocessed_page = Image.new("RGB", (width, height), (255, 255, 255))
    img_np = np.array(page_img.convert("RGB"))
    gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
    thresh = cv2.adaptiveThreshold(
        gray, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        11, 2
    )
    preprocessed_page = Image.fromarray(cv2.cvtColor(thresh, cv2.COLOR_GRAY2RGB))
    
    return preprocessed_page, gt_details

def gt_database_entry(gt_details):
    return gt_details

def main():
    os.makedirs("pdfs", exist_ok=True)
    os.makedirs("output_pipeline/visualizations", exist_ok=True)
    
    # Generate blueprint page
    print("Generating page 1 for test_2.pdf with 30+ expressions...")
    page_img, gt_details = generate_blueprint_page()
    
    # Save PDF
    pdf_path = "pdfs/test_2.pdf"
    page_img.save(pdf_path, format="PDF")
    print(f"Saved PDF to: {pdf_path}")
    
    # Save high-res PNG for visualization/checking
    png_path = "output_pipeline/visualizations/test_2_page_1.png"
    page_img.save(png_path)
    print(f"Saved page preview PNG to: {png_path}")
    
    # Write a test ground truth JSON specifically for verifying this PDF page
    test_gt_db = {
        "test_2.pdf_page_1.png": gt_details
    }
    test_gt_path = "dataset_yolo/test_2_gt.json"
    os.makedirs(os.path.dirname(test_gt_path), exist_ok=True)
    with open(test_gt_path, "w", encoding="utf-8") as f:
        json.dump(test_gt_db, f, indent=2, ensure_ascii=False)
    print(f"Saved test ground truth database to: {test_gt_path}")
    
    # Also append this page to dataset_yolo/ground_truth.json if it exists
    main_gt_path = "dataset_yolo/ground_truth.json"
    if os.path.exists(main_gt_path):
        try:
            with open(main_gt_path, "r", encoding="utf-8") as f:
                db = json.load(f)
            db["test_2.pdf_page_1.png"] = gt_details
            with open(main_gt_path, "w", encoding="utf-8") as f:
                json.dump(db, f, indent=2, ensure_ascii=False)
            print("Successfully appended test page entries to main ground_truth.json")
        except Exception as e:
            print(f"Warning: Could not update main ground_truth.json: {e}")

if __name__ == "__main__":
    main()
