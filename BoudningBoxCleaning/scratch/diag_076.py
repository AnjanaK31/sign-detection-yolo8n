"""
Diagnostic: show raw crop vs cleaned crop with CC analysis overlay for crop_076 (Ø7 HOLE)
"""
import cv2, numpy as np, json, sys, os
from PIL import Image
Image.MAX_IMAGE_PIXELS = None

sys.path.append(r"d:\Internship\OCR_PDF\BoudningBoxCleaning")
from clean_page_expressions import rectify_crop

IMG_PATH   = r"d:\Internship\OCR_PDF\INTRA_cleaning\msil.png"
LABEL_PATH = r"d:\Internship\OCR_PDF\INTRA_cleaning\Label.txt"
OUT_DIR    = r"C:\Users\lalit\.gemini\antigravity-ide\brain\ccd5a870-b411-4e9c-979b-e6d8eb2a58f6"

CROP_IDX = 76   # Ø7 HOLE
LINE_AR_THRESH  = 4.0
LINE_LEN_THRESH = 30
INTERS_DILATE   = 4
BUFFER          = 0.30

# Load
orig_bgr = cv2.imread(IMG_PATH)
orig_np  = cv2.cvtColor(orig_bgr, cv2.COLOR_BGR2RGB)
orig_pil = Image.fromarray(orig_np)

with open(LABEL_PATH, "r", encoding="utf-8") as f:
    for line in f:
        parts = line.strip().split("\t")
        if "msil.png" in parts[0]:
            boxes = json.loads(parts[1])
            break

box = boxes[CROP_IDX]
print(f"Crop {CROP_IDX}: '{box['transcription']}'")
print(f"Points: {box['points']}")

pts_np = np.array(box["points"], dtype=np.float32)
rect   = cv2.minAreaRect(pts_np)
cx, cy = rect[0]
w_box, h_box = rect[1]
angle  = rect[2]
print(f"minAreaRect: center=({cx:.0f},{cy:.0f}) size=({w_box:.0f}x{h_box:.0f}) angle={angle:.1f}")

bbox_metrics = {"cx": cx, "cy": cy, "w": w_box, "h": h_box, "angle": angle}
crop_color   = rectify_crop(orig_pil, bbox_metrics=bbox_metrics, buffer_percent=BUFFER)
crop_np      = np.array(crop_color)
ch, cw       = crop_np.shape[:2]
print(f"Crop size: {cw} x {ch}")

# Save raw crop
raw_bgr = cv2.cvtColor(crop_np, cv2.COLOR_RGB2BGR)
cv2.imwrite(os.path.join(OUT_DIR, "diag_076_raw.png"), raw_bgr)

# CC analysis
crop_gray = cv2.cvtColor(crop_np, cv2.COLOR_RGB2GRAY)
_, thresh  = cv2.threshold(crop_gray, 200, 255, cv2.THRESH_BINARY_INV)
num_cc, cc_labels, stats, _ = cv2.connectedComponentsWithStats(thresh, connectivity=8)

print(f"\nCC analysis — {num_cc-1} components:")
line_mask = np.zeros((ch, cw), dtype=np.uint8)
text_mask = np.zeros((ch, cw), dtype=np.uint8)

for lbl in range(1, num_cc):
    bw     = stats[lbl, cv2.CC_STAT_WIDTH]
    bh     = stats[lbl, cv2.CC_STAT_HEIGHT]
    area   = stats[lbl, cv2.CC_STAT_AREA]
    long_s = max(bw, bh)
    short_s= min(bw, bh)
    ar     = long_s / short_s if short_s > 0 else 0
    comp   = (cc_labels == lbl).astype(np.uint8) * 255
    kind   = "LINE" if ar >= LINE_AR_THRESH and long_s >= LINE_LEN_THRESH else "text"
    if kind == "LINE":
        line_mask = cv2.bitwise_or(line_mask, comp)
    else:
        text_mask = cv2.bitwise_or(text_mask, comp)
    print(f"  CC {lbl:3d}: {bw:4d}x{bh:4d} AR={ar:5.1f} area={area:5d}  -> {kind}")

ik           = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (INTERS_DILATE*2+1, INTERS_DILATE*2+1))
line_dilated = cv2.dilate(line_mask, ik)
inters_mask  = cv2.bitwise_and(line_dilated, text_mask)
protect_mask = cv2.bitwise_or(text_mask, inters_mask)
erase_mask   = cv2.bitwise_and(line_mask, cv2.bitwise_not(protect_mask))

# Save CC overlay (green=LINE, red=text, orange=intersection)
overlay = crop_np.copy()
overlay[line_mask  > 0] = [0, 200, 0]
overlay[text_mask  > 0] = [200, 80, 0]
overlay[inters_mask> 0] = [255, 165, 0]
cv2.imwrite(os.path.join(OUT_DIR, "diag_076_cc_overlay.png"), cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))

# Save cleaned
cleaned = crop_np.copy()
cleaned[erase_mask > 0] = [255, 255, 255]
cv2.imwrite(os.path.join(OUT_DIR, "diag_076_cleaned.png"), cv2.cvtColor(cleaned, cv2.COLOR_RGB2BGR))
print(f"\nErased: {int(np.sum(erase_mask>0))} px")
print("Saved: diag_076_raw.png, diag_076_cc_overlay.png, diag_076_cleaned.png")
