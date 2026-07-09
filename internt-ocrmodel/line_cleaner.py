import cv2
import numpy as np
import math
from PIL import Image

def skeletonize(img: np.ndarray) -> np.ndarray:
    """Applies Zhang-Suen thinning algorithm to guarantee a thin 1-pixel wide skeleton.
    Expects input image: binary, foreground = 255 (white), background = 0 (black).
    """
    binary = (img > 0).astype(np.uint8)
    while True:
        # Step 1
        padded = np.pad(binary, 1, mode='constant', constant_values=0)
        P2 = padded[:-2, 1:-1]
        P3 = padded[:-2, 2:]
        P4 = padded[1:-1, 2:]
        P5 = padded[2:, 2:]
        P6 = padded[2:, 1:-1]
        P7 = padded[2:, :-2]
        P8 = padded[1:-1, :-2]
        P9 = padded[:-2, :-2]
        
        B = P2 + P3 + P4 + P5 + P6 + P7 + P8 + P9
        A = ((P2 == 0) & (P3 == 1)).astype(np.uint8) + \
            ((P3 == 0) & (P4 == 1)).astype(np.uint8) + \
            ((P4 == 0) & (P5 == 1)).astype(np.uint8) + \
            ((P5 == 0) & (P6 == 1)).astype(np.uint8) + \
            ((P6 == 0) & (P7 == 1)).astype(np.uint8) + \
            ((P7 == 0) & (P8 == 1)).astype(np.uint8) + \
            ((P8 == 0) & (P9 == 1)).astype(np.uint8) + \
            ((P9 == 0) & (P2 == 1)).astype(np.uint8)
            
        cond1 = (B >= 2) & (B <= 6)
        cond2 = (A == 1)
        cond3 = (P2 * P4 * P6 == 0)
        cond4 = (P4 * P6 * P8 == 0)
        
        to_delete = (binary == 1) & cond1 & cond2 & cond3 & cond4
        if not np.any(to_delete):
            step1_changed = False
        else:
            binary[to_delete] = 0
            step1_changed = True
            
        # Step 2
        padded = np.pad(binary, 1, mode='constant', constant_values=0)
        P2 = padded[:-2, 1:-1]
        P3 = padded[:-2, 2:]
        P4 = padded[1:-1, 2:]
        P5 = padded[2:, 2:]
        P6 = padded[2:, 1:-1]
        P7 = padded[2:, :-2]
        P8 = padded[1:-1, :-2]
        P9 = padded[:-2, :-2]
        
        B = P2 + P3 + P4 + P5 + P6 + P7 + P8 + P9
        A = ((P2 == 0) & (P3 == 1)).astype(np.uint8) + \
            ((P3 == 0) & (P4 == 1)).astype(np.uint8) + \
            ((P4 == 0) & (P5 == 1)).astype(np.uint8) + \
            ((P5 == 0) & (P6 == 1)).astype(np.uint8) + \
            ((P6 == 0) & (P7 == 1)).astype(np.uint8) + \
            ((P7 == 0) & (P8 == 1)).astype(np.uint8) + \
            ((P8 == 0) & (P9 == 1)).astype(np.uint8) + \
            ((P9 == 0) & (P2 == 1)).astype(np.uint8)
            
        cond1 = (B >= 2) & (B <= 6)
        cond2 = (A == 1)
        cond3 = (P2 * P4 * P8 == 0)
        cond4 = (P2 * P6 * P8 == 0)
        
        to_delete2 = (binary == 1) & cond1 & cond2 & cond3 & cond4
        if not np.any(to_delete2):
            step2_changed = False
        else:
            binary[to_delete2] = 0
            step2_changed = True
            
        if not step1_changed and not step2_changed:
            break
            
    return (binary * 255).astype(np.uint8)

def get_skeleton_junctions(skel: np.ndarray) -> np.ndarray:
    """Finds true junction points in the skeleton where 3 or more branches meet.
    Uses topological transition counts to prevent diagonal staircases from being classified as junctions.
    """
    binary = (skel > 0).astype(np.uint8)
    padded = np.pad(binary, 1, mode='constant', constant_values=0)
    
    P2 = padded[:-2, 1:-1]
    P3 = padded[:-2, 2:]
    P4 = padded[1:-1, 2:]
    P5 = padded[2:, 2:]
    P6 = padded[2:, 1:-1]
    P7 = padded[2:, :-2]
    P8 = padded[1:-1, :-2]
    P9 = padded[:-2, :-2]
    
    # T = count of 0-to-1 transitions in the 8-neighborhood sequence
    T = ((P2 == 0) & (P3 == 1)).astype(np.uint8) + \
        ((P3 == 0) & (P4 == 1)).astype(np.uint8) + \
        ((P4 == 0) & (P5 == 1)).astype(np.uint8) + \
        ((P5 == 0) & (P6 == 1)).astype(np.uint8) + \
        ((P6 == 0) & (P7 == 1)).astype(np.uint8) + \
        ((P7 == 0) & (P8 == 1)).astype(np.uint8) + \
        ((P8 == 0) & (P9 == 1)).astype(np.uint8) + \
        ((P9 == 0) & (P2 == 1)).astype(np.uint8)
        
    junctions = (binary == 1) & (T >= 3)
    return (junctions * 255).astype(np.uint8)


def split_skeleton_into_branches(skel: np.ndarray) -> list:
    """Splits the skeleton at junction points. Returns (branches, junctions).
    Dilates the junctions slightly to ensure branches are completely disconnected.
    """
    junctions = get_skeleton_junctions(skel)
    
    # Dilate junctions by 3x3 kernel to cleanly disconnect intersecting branches
    junc_dilate_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    junctions_dilated = cv2.dilate(junctions, junc_dilate_kernel)
    
    skel_no_junc = cv2.subtract(skel, junctions_dilated)
    
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(skel_no_junc, connectivity=8)
    
    branches = []
    for label in range(1, num_labels):
        ys, xs = np.where(labels == label)
        if len(xs) > 0:
            pts = np.column_stack((xs, ys))
            branches.append(pts)
            
    return branches, junctions


def fit_circle(pts: np.ndarray):
    """Fits a circle to a set of points (x, y) using linear least-squares."""
    x = pts[:, 0]
    y = pts[:, 1]
    N = len(pts)
    
    A = np.column_stack((x, y, np.ones(N)))
    B = -x**2 - y**2
    
    try:
        W, residuals, rank, s = np.linalg.lstsq(A, B, rcond=None)
        a, b, c = W
        xc = -a / 2.0
        yc = -b / 2.0
        r_sq = xc**2 + yc**2 - c
        if r_sq < 0:
            return None, None, None, float('inf')
        R = math.sqrt(r_sq)
        
        dists = np.sqrt((x - xc)**2 + (y - yc)**2)
        errors = np.abs(dists - R)
        mean_err = np.mean(errors)
        return xc, yc, R, mean_err
    except np.linalg.LinAlgError:
        return None, None, None, float('inf')

def fit_polynomial(pts: np.ndarray):
    """Fits quadratic polynomials y = f(x) and x = f(y)."""
    x = pts[:, 0].astype(np.float64)
    y = pts[:, 1].astype(np.float64)
    
    try:
        p_y = np.polyfit(x, y, 2)
        pred_y = np.polyval(p_y, x)
        err_y = np.mean(np.abs(y - pred_y))
    except Exception:
        err_y = float('inf')
        
    try:
        p_x = np.polyfit(y, x, 2)
        pred_x = np.polyval(p_x, y)
        err_x = np.mean(np.abs(x - pred_x))
    except Exception:
        err_x = float('inf')
        
    return min(err_y, err_x)

def classify_branch_shape(pts: np.ndarray, thresh_px: float = 1.2) -> tuple:
    """Classifies a branch path as 'straight', 'curved', or 'unknown'."""
    N = len(pts)
    if N < 5:
        return "noise", 0.0
        
    x = pts[:, 0]
    y = pts[:, 1]
    coords = np.column_stack((x, y))
    
    # 1. Straight Line Fit via PCA
    cov = np.cov(coords, rowvar=False)
    if cov.ndim < 2 or np.isnan(cov).any():
        return "noise", 0.0
        
    evals, evecs = np.linalg.eigh(cov)
    min_eval = max(0.0, min(evals))
    sigma_pca = math.sqrt(min_eval)
    
    if sigma_pca < thresh_px:
        score = 1.0 - (sigma_pca / thresh_px)
        return "straight", score
        
    # 2. Circle/Arc Fit
    xc, yc, R, err_circle = fit_circle(pts)
    if err_circle < thresh_px and R is not None and R < max(np.max(x) - np.min(x), np.max(y) - np.min(y)) * 5:
        # Small radius check (protect character loops like ○, 0, R)
        if R <= 22.0:
            return "unknown", 0.0
        score = 1.0 - (err_circle / thresh_px)
        return "curved", score
        
    # 3. Quadratic Polynomial Fit
    err_poly = fit_polynomial(pts)
    if err_poly < thresh_px:
        score = 1.0 - (err_poly / thresh_px)
        return "curved", score
        
    return "unknown", 0.0

def map_crop_to_page_coordinates(pts_crop, cx, cy, w, h, angle_deg, buffer_percent=0.20):
    """Maps coordinates from a rectified standard crop back to full page coordinates."""
    if len(pts_crop) == 0:
        return np.empty((0, 2))
    bw = w * (1.0 + buffer_percent)
    bh = h * (1.0 + buffer_percent)
    diag = math.sqrt(bw**2 + bh**2)
    crop_size = int(math.ceil(diag)) + 20
    half_size = crop_size // 2
    
    bx_min = int(round(half_size - bw / 2))
    by_min = int(round(half_size - bh / 2))
    
    pts_warped = pts_crop + np.array([bx_min, by_min])
    
    rad = math.radians(-angle_deg)
    cos_a = math.cos(rad)
    sin_a = math.sin(rad)
    
    dx = pts_warped[:, 0] - half_size
    dy = pts_warped[:, 1] - half_size
    
    x_padded = half_size + dx * cos_a - dy * sin_a
    y_padded = half_size + dx * sin_a + dy * cos_a
    
    x_min = int(round(cx - half_size))
    y_min = int(round(cy - half_size))
    
    x_page = x_padded + x_min
    y_page = y_padded + y_min
    
    return np.column_stack((x_page, y_page))

def clean_patch_lines(crop_pil_std: Image.Image, crop_pil_big: Image.Image, line_err_thresh: float = 1.2,
                      bbox_metrics: dict = None, labels_im: np.ndarray = None, protected_labels: set = None) -> tuple:
    """Traces boundary crossing components, separates CAD lines from text, and erases them."""
    img_std = np.array(crop_pil_std.convert("L"))
    img_big = np.array(crop_pil_big.convert("L"))
    
    H_std, W_std = img_std.shape
    H_big, W_big = img_big.shape
    
    dx = (W_big - W_std) // 2
    dy = (H_big - H_std) // 2
    
    _, thresh_std = cv2.threshold(img_std, 127, 255, cv2.THRESH_BINARY_INV)
    _, thresh_big = cv2.threshold(img_big, 127, 255, cv2.THRESH_BINARY_INV)
    
    # Distance transform for thickness verification
    dist_transform = cv2.distanceTransform(thresh_std, cv2.DIST_L2, 5)
    
    # Just remove isolated single-pixel noise, preserving commas, periods, and small characters
    for t in [thresh_std, thresh_big]:
        num, labels, stats, _ = cv2.connectedComponentsWithStats(t, connectivity=8)
        for label in range(1, num):
            if stats[label, cv2.CC_STAT_AREA] < 2:
                t[labels == label] = 0
                
    # Crossing components mask
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(thresh_big, connectivity=8)
    boundary_mask = np.ones((H_big, W_big), dtype=np.uint8) * 255
    boundary_mask[dy:dy+H_std, dx:dx+W_std] = 0
    inner_mask = np.zeros((H_big, W_big), dtype=np.uint8)
    inner_mask[dy:dy+H_std, dx:dx+W_std] = 255
    
    crossing_mask_std = np.zeros((H_std, W_std), dtype=np.uint8)
    for label in range(1, num_labels):
        comp_mask = (labels == label)
        if np.any(comp_mask & (boundary_mask == 255)) and np.any(comp_mask & (inner_mask == 255)):
            crossing_mask_std[comp_mask[dy:dy+H_std, dx:dx+W_std]] = 255
            
    # Skeletonize standard crop
    skel_std = skeletonize(thresh_std)
    branches, junctions = split_skeleton_into_branches(skel_std)
    
    # Classify branches and separate border-touching from internal branches
    border_branches = []
    internal_branches = []
    
    # Mask of protected text/symbol branches
    protected_text_skel_mask = np.zeros((H_std, W_std), dtype=np.uint8)
    
    border_margin = 3
    
    for idx, branch_pts in enumerate(branches):
        # Calculate thickness of this branch
        branch_thickness = 2 * np.mean([dist_transform[pt[1], pt[0]] for pt in branch_pts])
        is_thick_text = (branch_thickness >= 2.6)
        
        # Check if page-level label protection is active for this branch
        is_page_protected = False
        if bbox_metrics is not None and labels_im is not None and protected_labels is not None:
            cx = bbox_metrics.get('cx')
            cy = bbox_metrics.get('cy')
            w = bbox_metrics.get('w')
            h = bbox_metrics.get('h')
            angle_deg = bbox_metrics.get('angle')
            
            pts_page = map_crop_to_page_coordinates(branch_pts, cx, cy, w, h, angle_deg, buffer_percent=0.20)
            
            protected_points_count = 0
            img_h_page, img_w_page = labels_im.shape
            for pt_page in pts_page:
                px_x = int(round(pt_page[0]))
                px_y = int(round(pt_page[1]))
                if 0 <= px_x < img_w_page and 0 <= px_y < img_h_page:
                    label_val = labels_im[px_y, px_x]
                    if label_val in protected_labels:
                        protected_points_count += 1
            
            if len(branch_pts) > 0 and (protected_points_count / len(branch_pts)) > 0.30:
                is_page_protected = True
                
        # Protect thick strokes and page-level symbol islands
        if is_thick_text or is_page_protected:
            for pt in branch_pts:
                protected_text_skel_mask[pt[1], pt[0]] = 255
            continue
            
        # Must overlap with crossing components
        overlap_count = sum(1 for pt in branch_pts if crossing_mask_std[pt[1], pt[0]] > 0)
        if (overlap_count / len(branch_pts)) < 0.25:
            continue
            
        shape_type, score = classify_branch_shape(branch_pts, thresh_px=line_err_thresh)
        if shape_type not in ["straight", "curved"]:
            continue
            
        # Check if it touches the border of standard crop
        touches_border = np.any((branch_pts[:, 0] < border_margin) | 
                                (branch_pts[:, 0] > W_std - 1 - border_margin) | 
                                (branch_pts[:, 1] < border_margin) | 
                                (branch_pts[:, 1] > H_std - 1 - border_margin))
                                
        branch_info = {
            "pts": branch_pts,
            "type": shape_type,
            "score": score
        }
        
        if touches_border:
            border_branches.append(branch_info)
        else:
            internal_branches.append(branch_info)
            
    # Collect points to erase
    erase_skel_mask = np.zeros((H_std, W_std), dtype=np.uint8)
    deleted_lines = []
    
    # 1. Always delete border-touching CAD lines
    for b in border_branches:
        for pt in b["pts"]:
            erase_skel_mask[pt[1], pt[0]] = 255
        deleted_lines.append(b)
        
    # 2. For internal branches, only delete if they align/fit with border-touching branches
    # This prevents deleting internal character strokes (like the crossbar of 'H' or curves of 'B')
    if len(border_branches) > 0 and len(internal_branches) > 0:
        # Fit models on the combined border points
        border_pts = np.vstack([b["pts"] for b in border_branches])
        
        # Check straight line fit of border points
        border_x = border_pts[:, 0]
        border_y = border_pts[:, 1]
        
        is_border_straight = False
        p_line = None
        # Fit PCA/line
        cov = np.cov(border_pts, rowvar=False)
        if cov.ndim == 2 and not np.isnan(cov).any():
            evals, evecs = np.linalg.eigh(cov)
            if math.sqrt(max(0.0, min(evals))) < 2.0:
                is_border_straight = True
                # Fit a line y = mx + c or x = my + c
                if np.std(border_x) > np.std(border_y):
                    p_line = np.polyfit(border_x, border_y, 1)
                    line_dir = 'x'
                else:
                    p_line = np.polyfit(border_y, border_x, 1)
                    line_dir = 'y'
                    
        # Fit circle on border points
        xc, yc, R, err_circle = fit_circle(border_pts)
        is_border_circle = err_circle < 2.0 and R is not None
        
        for b in internal_branches:
            pts = b["pts"]
            should_delete = False
            
            if is_border_straight and p_line is not None:
                # Check distance to fitted straight line
                if line_dir == 'x':
                    dist = np.mean(np.abs(pts[:, 1] - np.polyval(p_line, pts[:, 0])))
                else:
                    dist = np.mean(np.abs(pts[:, 0] - np.polyval(p_line, pts[:, 1])))
                if dist < 2.2:
                    should_delete = True
                    
            elif is_border_circle and R is not None:
                # Check distance to fitted circle
                dists = np.sqrt((pts[:, 0] - xc)**2 + (pts[:, 1] - yc)**2)
                dist = np.mean(np.abs(dists - R))
                if dist < 2.2:
                    should_delete = True
                    
            if should_delete:
                for pt in pts:
                    erase_skel_mask[pt[1], pt[0]] = 255
                deleted_lines.append(b)
 
                
    # Dilate erase skeleton mask
    dilate_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    erase_mask_thick = cv2.dilate(erase_skel_mask, dilate_kernel, iterations=1)
    erase_mask_final = cv2.bitwise_and(thresh_std, erase_mask_thick)
    
    # Junction protection (don't erase near intersection junctions)
    junc_protect_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    protected_zone = cv2.dilate(junctions, junc_protect_kernel)
    
    # Dilate protected text mask slightly to protect text strokes from erosion bleeding
    protected_text_mask_dilated = cv2.dilate(protected_text_skel_mask, cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3)))
    
    # Combine junctions and protected text zones
    all_protected_zones = cv2.bitwise_or(protected_zone, protected_text_mask_dilated)
    
    erase_mask_protected = cv2.bitwise_and(erase_mask_final, cv2.bitwise_not(all_protected_zones))
    
    cleaned_thresh_std = cv2.subtract(thresh_std, erase_mask_protected)
    cleaned_std = cv2.bitwise_not(cleaned_thresh_std)
    
    # Review image
    review_img = np.ones((H_std, W_std, 3), dtype=np.uint8) * 255
    text_mask = cleaned_thresh_std
    line_mask = erase_mask_protected
    
    review_img[text_mask > 0] = [255, 0, 0]   # Text = Red
    review_img[line_mask > 0] = [0, 0, 255]   # Lines = Blue
    
    details = {
        "deleted_lines": [{"type": b["type"], "score": float(b["score"])} for b in deleted_lines],
        "total_deleted_pixels": int(np.sum(line_mask > 0)),
        "total_text_pixels": int(np.sum(text_mask > 0))
    }
    
    return Image.fromarray(cleaned_std), review_img, details

def evaluate_cleaning(pred_text_mask: np.ndarray, pred_line_mask: np.ndarray, 
                      gt_text_mask: np.ndarray, gt_line_mask: np.ndarray) -> dict:
    p_txt = pred_text_mask > 0
    p_line = pred_line_mask > 0
    gt_txt = gt_text_mask > 0
    gt_line = gt_line_mask > 0
    
    tot_gt_txt = np.sum(gt_txt)
    tpr = np.sum(p_txt & gt_txt) / tot_gt_txt if tot_gt_txt > 0 else 1.0
    
    tot_gt_line = np.sum(gt_line)
    ldr = np.sum(p_line & gt_line) / tot_gt_line if tot_gt_line > 0 else 1.0
    
    fdr = np.sum(p_line & gt_txt) / tot_gt_txt if tot_gt_txt > 0 else 0.0
    llr = np.sum(p_txt & gt_line) / tot_gt_line if tot_gt_line > 0 else 0.0
    
    f1 = 2 * (tpr * ldr) / (tpr + ldr) if (tpr + ldr) > 0 else 0.0
    
    return {
        "text_preservation_rate": float(tpr),
        "line_deletion_rate": float(ldr),
        "text_false_deletion_rate": float(fdr),
        "line_leakage_rate": float(llr),
        "f1_score": float(f1)
    }
