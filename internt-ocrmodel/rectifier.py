import cv2
import numpy as np
import math
from PIL import Image

def rectify_crop(image, pts=None, bbox_metrics=None, buffer_percent=0.08, target_size=None):
    """Extracts and deskews a rotated bounding box from an image using cv2.getRotationMatrix2D and cv2.warpAffine.
    
    Args:
        image: PIL Image or numpy array (RGB or BGR).
        pts: numpy array of shape (4, 2) containing the 4 corners of the OBB [TL, TR, BR, BL].
        bbox_metrics: dict containing keys: 'cx', 'cy', 'w', 'h', 'angle' (angle in degrees).
        buffer_percent: float, cropping buffer percentage for context.
        target_size: tuple (width, height) for the output cropped image (optional).
        
    Returns:
        A deskewed, normalized crop as a PIL Image.
    """
    is_pil = isinstance(image, Image.Image)
    if is_pil:
        img_np = np.array(image.convert("RGB"))
    else:
        img_np = image.copy()
        
    if pts is not None:
        pts = np.array(pts, dtype=np.float32)
        cx, cy = np.mean(pts, axis=0)
        w = np.linalg.norm(pts[0] - pts[1])
        h = np.linalg.norm(pts[0] - pts[3])
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
        
    bw = w * (1.0 + buffer_percent)
    bh = h * (1.0 + buffer_percent)
    
    diag = math.sqrt(bw**2 + bh**2)
    crop_size = int(math.ceil(diag)) + 20
    half_size = crop_size // 2
    
    x_min = int(round(cx - half_size))
    y_min = int(round(cy - half_size))
    x_max = x_min + crop_size
    y_max = y_min + crop_size
    
    img_h, img_w = img_np.shape[:2]
    
    pad_left = max(0, -x_min)
    pad_top = max(0, -y_min)
    pad_right = max(0, x_max - img_w)
    pad_bottom = max(0, y_max - img_h)
    
    src_x_min = max(0, x_min)
    src_y_min = max(0, y_min)
    src_x_max = min(img_w, x_max)
    src_y_max = min(img_h, y_max)
    
    sub_img = img_np[src_y_min:src_y_max, src_x_min:src_x_max]
    
    if len(img_np.shape) == 3:
        padded_crop = np.ones((crop_size, crop_size, 3), dtype=np.uint8) * 255
        padded_crop[pad_top:crop_size - pad_bottom, pad_left:crop_size - pad_right] = sub_img
    else:
        padded_crop = np.ones((crop_size, crop_size), dtype=np.uint8) * 255
        padded_crop[pad_top:crop_size - pad_bottom, pad_left:crop_size - pad_right] = sub_img
        
    rot_center = (half_size, half_size)
    M = cv2.getRotationMatrix2D(rot_center, angle, 1.0)
    
    warped = cv2.warpAffine(
        padded_crop, 
        M, 
        (crop_size, crop_size), 
        flags=cv2.INTER_CUBIC, 
        borderMode=cv2.BORDER_CONSTANT, 
        borderValue=(255, 255, 255) if len(img_np.shape) == 2 else (255, 255, 255)
    )
    
    bx_min = int(round(half_size - bw / 2))
    by_min = int(round(half_size - bh / 2))
    bx_max = bx_min + int(round(bw))
    by_max = by_min + int(round(bh))
    
    bx_min = max(0, bx_min)
    by_min = max(0, by_min)
    bx_max = min(crop_size, bx_max)
    by_max = min(crop_size, by_max)
    
    final_crop = warped[by_min:by_max, bx_min:bx_max]
    
    if target_size is not None and final_crop.size > 0:
        final_crop = cv2.resize(final_crop, target_size, interpolation=cv2.INTER_CUBIC)
        
    if is_pil:
        return Image.fromarray(final_crop)
    return final_crop
