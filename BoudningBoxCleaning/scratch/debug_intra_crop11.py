"""
Debug what's happening inside clean_crop_lines for crop 11 (Ø242).
Dumps all intermediate masks so we can see exactly WHY pixels are erased.
"""
import sys, os, json, math
import cv2
import numpy as np
from PIL import Image

sys.path.append(os.path.abspath(r"d:\Internship\OCR_PDF\BoudningBoxCleaning"))
from clean_page_expressions import rectify_crop

OUT = r"d:\Internship\OCR_PDF\BoudningBoxCleaning\scratch\debug_crop11"
os.makedirs(OUT, exist_ok=True)

def save(name, arr):
    """Save a numpy array as a PNG for inspection."""
    if arr.dtype != np.uint8:
        arr = (arr.astype(np.float32) / arr.max() * 255).astype(np.uint8) if arr.max() > 0 else arr.astype(np.uint8)
    if len(arr.shape) == 2:
        arr = cv2.cvtColor(arr, cv2.COLOR_GRAY2BGR)
    cv2.imwrite(os.path.join(OUT, name + ".png"), arr)
    print(f"  Saved {name}.png  shape={arr.shape}  nonzero={np.count_nonzero(arr[:,:,0] if len(arr.shape)==3 else arr)}")

# ── Load page & build page-level masks ─────────────────────────────────────────
img_path   = r"d:\Internship\OCR_PDF\INTRA_cleaning\intra.png"
label_path = r"d:\Internship\OCR_PDF\INTRA_cleaning\Label.txt"
orig_img   = Image.open(img_path)
img_gray   = cv2.cvtColor(np.array(orig_img.convert("RGB")), cv2.COLOR_RGB2GRAY)

print("Running page-level MSER...")
mser = cv2.MSER_create(5, 30, 15000)
regions, _ = mser.detectRegions(img_gray)
mser_char_mask = np.zeros(img_gray.shape, np.uint8)
mser_line_mask = np.zeros(img_gray.shape, np.uint8)
max_mser_char_dim = 0.0
for r in regions:
    rect = cv2.minAreaRect(r)
    box_pts = cv2.boxPoints(rect)
    v1 = box_pts[1]-box_pts[0]; v2 = box_pts[2]-box_pts[1]
    L1=np.linalg.norm(v1); L2=np.linalg.norm(v2)
    Lmin=min(L1,L2); Lmax=max(L1,L2)
    ar = Lmax/Lmin if Lmin else 0
    box_pts = np.intp(box_pts)
    if 5<=L1<=80 and 5<=L2<=80 and ar<=1.5:
        cv2.drawContours(mser_char_mask,[box_pts],0,255,-1)
        max_mser_char_dim = max(max_mser_char_dim, Lmax)
    elif Lmin<=80 and Lmax>=5 and ar>1.5:
        cv2.drawContours(mser_line_mask,[box_pts],0,255,-1)
if max_mser_char_dim==0: max_mser_char_dim=30.0
print(f"  max_mser_char_dim={max_mser_char_dim:.1f}")

_, thresh_page = cv2.threshold(img_gray, 180, 255, cv2.THRESH_BINARY)
preprocessed_img = Image.fromarray(cv2.cvtColor(thresh_page, cv2.COLOR_GRAY2RGB))
thresh_page_inv = cv2.bitwise_not(thresh_page)
_, labels_im_page, stats_page, _ = cv2.connectedComponentsWithStats(thresh_page_inv, connectivity=8)
small_pg = np.zeros_like(thresh_page_inv)
large_pg = np.zeros_like(thresh_page_inv)
for lbl in range(1, stats_page.shape[0]):
    w_c=stats_page[lbl,cv2.CC_STAT_WIDTH]; h_c=stats_page[lbl,cv2.CC_STAT_HEIGHT]
    a_c=stats_page[lbl,cv2.CC_STAT_AREA]
    cm=(labels_im_page==lbl).astype(np.uint8)*255
    if w_c<=50 and h_c<=50 and a_c<=350: small_pg=cv2.bitwise_or(small_pg,cm)
    else: large_pg=cv2.bitwise_or(large_pg,cm)

# ── Get crop 11 bbox ────────────────────────────────────────────────────────────
intra_boxes = None
with open(label_path,"r",encoding="utf-8") as f:
    for line in f:
        parts=line.strip().split("\t")
        if len(parts)>=2 and "intra.png" in parts[0]:
            intra_boxes=json.loads(parts[1]); break

box = intra_boxes[11]
print(f"Box 11 transcription: {box['transcription']}")
pts_np = np.array(box["points"], dtype=np.float32)
rect   = cv2.minAreaRect(pts_np)
cx,cy  = rect[0]; w_box,h_box=rect[1]; angle=rect[2]
bm = {'cx':cx,'cy':cy,'w':w_box,'h':h_box,'angle':angle}

# ── Crop all masks ──────────────────────────────────────────────────────────────
crop_std   = rectify_crop(preprocessed_img,               bm, 0.20)
crop_mchar = rectify_crop(Image.fromarray(mser_char_mask), bm, 0.20)
crop_mline = rectify_crop(Image.fromarray(mser_line_mask), bm, 0.20)
crop_small = rectify_crop(Image.fromarray(small_pg),       bm, 0.20)
crop_large = rectify_crop(Image.fromarray(large_pg),       bm, 0.20)
crop_color = rectify_crop(orig_img.convert("RGB"),         bm, 0.20)

img_std  = np.array(crop_std.convert("L"))
_,thresh_std = cv2.threshold(img_std,180,255,cv2.THRESH_BINARY_INV)

mser_char_np = np.array(crop_mchar.convert("L"))
mser_line_np = np.array(crop_mline.convert("L"))
small_np     = np.array(crop_small.convert("L"))
large_np     = np.array(crop_large.convert("L"))

# Build masks exactly as clean_crop_lines does
text_prot = np.zeros_like(thresh_std)
text_prot[mser_char_np>127]=255

mline = np.zeros_like(thresh_std); mline[mser_line_np>127]=255
mline = cv2.bitwise_and(mline, thresh_std)
mline_after_sub = cv2.bitwise_and(mline, cv2.bitwise_not(text_prot))

llarge = np.zeros_like(thresh_std); llarge[large_np>127]=255
llarge = cv2.bitwise_and(llarge, thresh_std)
llarge_after_sub = cv2.bitwise_and(llarge, cv2.bitwise_not(text_prot))

ssmall = np.zeros_like(thresh_std); ssmall[small_np>127]=255
ssmall = cv2.bitwise_and(ssmall, thresh_std)

print("\nCrop 11 mask stats:")
print(f"  thresh_std nonzero:        {np.count_nonzero(thresh_std)}")
print(f"  text_protection nonzero:   {np.count_nonzero(text_prot)}")
print(f"  mser_line (raw) nonzero:   {np.count_nonzero(mline)}")
print(f"  mser_line (after sub):     {np.count_nonzero(mline_after_sub)}")
print(f"  large_islands (raw):       {np.count_nonzero(llarge)}")
print(f"  large_islands (after sub): {np.count_nonzero(llarge_after_sub)}")
print(f"  small_islands:             {np.count_nonzero(ssmall)}")

# Save everything
save("00_thresh_std",        thresh_std)
save("01_text_protection",   text_prot)
save("02_mser_line_raw",     mline)
save("03_mser_line_after_sub", mline_after_sub)
save("04_large_islands_raw", llarge)
save("05_large_islands_after_sub", llarge_after_sub)
save("06_small_islands",     ssmall)
save("07_orig_color",        np.array(crop_color))

# Show CC analysis of thresh_std
num_cc, cc_labels, cc_stats, _ = cv2.connectedComponentsWithStats(thresh_std, connectivity=8)
print(f"\nConnected components in thresh_std: {num_cc-1}")
for lbl in range(1, num_cc):
    w=cc_stats[lbl,cv2.CC_STAT_WIDTH]; h=cc_stats[lbl,cv2.CC_STAT_HEIGHT]
    a=cc_stats[lbl,cv2.CC_STAT_AREA]
    comp=(cc_labels==lbl).astype(np.uint8)*255
    has_mser_char = np.any(cv2.bitwise_and(comp,text_prot))
    is_in_mline   = np.any(cv2.bitwise_and(comp,mline_after_sub))
    is_in_llarge  = np.any(cv2.bitwise_and(comp,llarge_after_sub))
    print(f"  CC{lbl}: w={w} h={h} area={a}  has_mser_char={has_mser_char}  in_mline={is_in_mline}  in_llarge={is_in_llarge}")
