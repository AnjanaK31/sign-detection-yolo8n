"""
Generate original (uncleaned) crops of msil.png for the web viewer.
Saves to msil_original_crops/ alongside msil_cleaned_crops/
"""
import sys, os, json
import cv2
import numpy as np
from PIL import Image
Image.MAX_IMAGE_PIXELS = None

sys.path.append(r"d:\Internship\OCR_PDF\BoudningBoxCleaning")
from clean_page_expressions import rectify_crop

IMG_PATH   = r"d:\Internship\OCR_PDF\INTRA_cleaning\msil.png"
LABEL_PATH = r"d:\Internship\OCR_PDF\INTRA_cleaning\Label.txt"
OUT_DIR    = r"d:\Internship\OCR_PDF\INTRA_cleaning\msil_original_crops"
BUFFER     = 0.30

os.makedirs(OUT_DIR, exist_ok=True)
print("Loading msil.png ...")
orig_bgr = cv2.imread(IMG_PATH)
orig_np  = cv2.cvtColor(orig_bgr, cv2.COLOR_BGR2RGB)
orig_pil = Image.fromarray(orig_np)
H, W     = orig_np.shape[:2]
print(f"  {W} x {H} px")

with open(LABEL_PATH, "r", encoding="utf-8") as f:
    for line in f:
        parts = line.strip().split("\t")
        if "msil.png" in parts[0]:
            boxes = json.loads(parts[1])
            break

print(f"Saving {len(boxes)} original crops ...")
for idx, box in enumerate(boxes):
    transcription = box.get("transcription", f"box_{idx}")
    points = box.get("points", [])
    if len(points) != 4:
        continue
    pts_np = np.array(points, dtype=np.float32)
    rect   = cv2.minAreaRect(pts_np)
    cx, cy = rect[0]
    w_box, h_box = rect[1]
    angle  = rect[2]
    bbox_metrics = {"cx": cx, "cy": cy, "w": w_box, "h": h_box, "angle": angle}
    crop_color   = rectify_crop(orig_pil, bbox_metrics=bbox_metrics, buffer_percent=BUFFER)
    crop_np      = np.array(crop_color)
    clean_name = (transcription
                  .replace("/","_").replace(":","_").replace("°","deg")
                  .replace("×","x").replace("±","+-").replace(" ","_")
                  .replace("(","").replace(")",""))
    fname = f"crop_{idx:03d}_{clean_name}.png"
    cv2.imwrite(os.path.join(OUT_DIR, fname), cv2.cvtColor(crop_np, cv2.COLOR_RGB2BGR))
    print(f"  [{idx:3d}] {transcription}")

print(f"\nDone. Saved to {OUT_DIR}")
