"""
Generate crop_viewer.html — injects all crop metadata + image paths as JSON
into the HTML template so it works as a standalone local file viewer.
"""
import os, json, re
import cv2

ORIG_DIR     = r"d:\Internship\OCR_PDF\INTRA_cleaning\msil_original_crops"
CLEAN_DIR    = r"d:\Internship\OCR_PDF\INTRA_cleaning\msil_cleaned_crops"
TEMPLATE     = r"d:\Internship\OCR_PDF\INTRA_cleaning\crop_viewer_template.html"
OUT_HTML     = r"d:\Internship\OCR_PDF\INTRA_cleaning\crop_viewer.html"

# Erase counts from the last run (copied from script output)
# We'll compute them dynamically by comparing pixel counts instead
def count_white_diff(orig_path, clean_path):
    """Count pixels that became white in cleaned vs original."""
    a = cv2.imread(orig_path,  cv2.IMREAD_GRAYSCALE)
    b = cv2.imread(clean_path, cv2.IMREAD_GRAYSCALE)
    if a is None or b is None or a.shape != b.shape:
        return 0
    # pixels that are dark in original but white in cleaned = erased
    mask = (a < 200) & (b >= 200)
    return int(mask.sum())

print("Building crop viewer ...")
crops = []

orig_files  = sorted(f for f in os.listdir(ORIG_DIR)  if f.endswith('.png'))
clean_files = sorted(f for f in os.listdir(CLEAN_DIR) if f.endswith('.png'))

# Match by filename
orig_map  = {f: f for f in orig_files}
clean_map = {f: f for f in clean_files}
all_names = sorted(set(orig_map) | set(clean_map))

for fname in all_names:
    # Parse idx and label from filename like crop_076_Ø7_HOLE.png
    m = re.match(r'crop_(\d+)_(.+)\.png', fname)
    if not m:
        continue
    idx   = int(m.group(1))
    label = m.group(2).replace('_', ' ').replace('deg', '°').replace('+-', '±')

    orig_path  = os.path.join(ORIG_DIR,  fname)
    clean_path = os.path.join(CLEAN_DIR, fname)

    # Get dimensions
    img = cv2.imread(orig_path if os.path.exists(orig_path) else clean_path)
    h, w = img.shape[:2] if img is not None else (0, 0)

    # Count erased pixels
    erased = 0
    if os.path.exists(orig_path) and os.path.exists(clean_path):
        erased = count_white_diff(orig_path, clean_path)

    # Use relative paths (both folders are siblings of the HTML)
    crops.append({
        "idx":   idx,
        "label": label,
        "orig":  f"msil_original_crops/{fname}",
        "clean": f"msil_cleaned_crops/{fname}",
        "w": w, "h": h,
        "erased": erased,
    })
    print(f"  [{idx:3d}] {label:<22s}  erased={erased:5d}px")

# Inject into template
with open(TEMPLATE, 'r', encoding='utf-8') as f:
    html = f.read()

crops_json = json.dumps(crops, ensure_ascii=False)
html = html.replace('__CROPS_JSON__', crops_json)

with open(OUT_HTML, 'w', encoding='utf-8') as f:
    f.write(html)

print(f"\nDone! Open in browser:")
print(f"  {OUT_HTML}")
print(f"  ({len(crops)} crops injected)")
