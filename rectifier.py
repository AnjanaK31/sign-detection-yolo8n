import cv2
import numpy as np
import math
from PIL import Image

def rectify_crop(image, pts=None, bbox_metrics=None, buffer_percent=0.08, target_size=None):
    """Extracts and deskews a rotated bounding box from an image using cv2.getRotationMatrix2D and cv2.warpAffine.
    
    Args:
        image: PIL Image or numpy array (RGB or BGR).
        pts: numpy array of shape (4, 2) containing the 4 corners of the OBB [TL, TR, BR, BL].
             If provided, bbox_metrics will be computed from it.
        bbox_metrics: dict containing keys: 'cx', 'cy', 'w', 'h', 'angle' (angle in degrees).
                      Required if pts is None.
        buffer_percent: float, cropping buffer percentage for context (default 8% for 5-10% range).
        target_size: tuple (width, height) for the output cropped image (optional).
        
    Returns:
        A deskewed, normalized crop as a PIL Image.
    """
    is_pil = isinstance(image, Image.Image)
    if is_pil:
        img_np = np.array(image.convert("RGB"))
    else:
        img_np = image.copy()
        
    # Phase 3 Parser: Retrieve cx, cy, w, h, and angle metrics
    if pts is not None:
        pts = np.array(pts, dtype=np.float32)
        # Center is the mean of corners
        cx, cy = np.mean(pts, axis=0)
        # Width: distance between TL and TR
        w = np.linalg.norm(pts[0] - pts[1])
        # Height: distance between TL and BL
        h = np.linalg.norm(pts[0] - pts[3])
        # Angle of rotation: calculate from the TR - TL edge
        dx, dy = pts[1] - pts[0]
        angle = math.degrees(math.atan2(dy, dx))
    elif bbox_metrics is not None:
        cx = bbox_metrics['cx']
        cy = bbox_metrics['cy']
        w = bbox_metrics['w']
        h = bbox_metrics['h']
        angle = bbox_metrics['angle']
    else:
        raise ValueError("Either 'pts' or 'bbox_metrics' must be provided.")
        
    # Apply a 5-10% cropping buffer for context
    bw = w * (1.0 + buffer_percent)
    bh = h * (1.0 + buffer_percent)
    
    # Calculate crop square size that will contain the full rotated bounding box
    diag = math.sqrt(bw**2 + bh**2)
    crop_size = int(math.ceil(diag)) + 20
    
    # Half size of crop
    half_size = crop_size // 2
    
    # Calculate crop region in the original image (with padding if out of bounds)
    x_min = int(round(cx - half_size))
    y_min = int(round(cy - half_size))
    x_max = x_min + crop_size
    y_max = y_min + crop_size
    
    img_h, img_w = img_np.shape[:2]
    
    # Crop boundaries with padding
    pad_left = max(0, -x_min)
    pad_top = max(0, -y_min)
    pad_right = max(0, x_max - img_w)
    pad_bottom = max(0, y_max - img_h)
    
    # Coordinates inside the original image
    src_x_min = max(0, x_min)
    src_y_min = max(0, y_min)
    src_x_max = min(img_w, x_max)
    src_y_max = min(img_h, y_max)
    
    # Sub-image from original
    sub_img = img_np[src_y_min:src_y_max, src_x_min:src_x_max]
    
    # Create padded image filled with white (255)
    if len(img_np.shape) == 3:
        padded_crop = np.ones((crop_size, crop_size, 3), dtype=np.uint8) * 255
        padded_crop[pad_top:crop_size - pad_bottom, pad_left:crop_size - pad_right] = sub_img
    else:
        padded_crop = np.ones((crop_size, crop_size), dtype=np.uint8) * 255
        padded_crop[pad_top:crop_size - pad_bottom, pad_left:crop_size - pad_right] = sub_img
        
    # Center of rotation is now the center of the padded crop
    rot_center = (half_size, half_size)
    
    # Get rotation matrix (use negative angle because OpenCV rotation is counter-clockwise)
    # But wait, YOLO angle is counter-clockwise, getRotationMatrix2D angle is counter-clockwise.
    # So we pass angle directly.
    M = cv2.getRotationMatrix2D(rot_center, angle, 1.0)
    
    # Perform the warping
    warped = cv2.warpAffine(
        padded_crop, 
        M, 
        (crop_size, crop_size), 
        flags=cv2.INTER_CUBIC, 
        borderMode=cv2.BORDER_CONSTANT, 
        borderValue=(255, 255, 255) if len(img_np.shape) == 2 else (255, 255, 255)
    )
    
    # Extract the final centered crop of size (bw, bh)
    bx_min = int(round(half_size - bw / 2))
    by_min = int(round(half_size - bh / 2))
    bx_max = bx_min + int(round(bw))
    by_max = by_min + int(round(bh))
    
    # Bounds check for the final slice
    bx_min = max(0, bx_min)
    by_min = max(0, by_min)
    bx_max = min(crop_size, bx_max)
    by_max = min(crop_size, by_max)
    
    final_crop = warped[by_min:by_max, bx_min:bx_max]
    
    # Resize to target size if requested
    if target_size is not None and final_crop.size > 0:
        final_crop = cv2.resize(final_crop, target_size, interpolation=cv2.INTER_CUBIC)
        
    if is_pil:
        return Image.fromarray(final_crop)
    return final_crop
