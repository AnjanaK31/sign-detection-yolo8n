import cv2
import numpy as np
from PIL import Image

def rectify_crop(image, pts, target_size=(128, 128)):
    """Extracts and deskews a rotated bounding box from an image using an Affine transformation.
    
    Args:
        image: PIL Image or numpy array (BGR or RGB).
        pts: numpy array of shape (4, 2) containing the 4 corners of the OBB.
             Expected order: [TL, TR, BR, BL].
        target_size: tuple (width, height) for the output cropped image.
        
    Returns:
        A deskewed, normalized crop as a 128x128 PIL Image or numpy array.
    """
    # Convert PIL Image to numpy array
    is_pil = isinstance(image, Image.Image)
    if is_pil:
        img_np = np.array(image)
    else:
        img_np = image.copy()
        
    pts = np.array(pts, dtype=np.float32)
    
    # We use 3 points to define the Affine transformation:
    # TL -> (0, 0)
    # TR -> (target_width, 0)
    # BL -> (0, target_height)
    src_pts = np.float32([pts[0], pts[1], pts[3]])
    
    tw, th = target_size
    dst_pts = np.float32([[0, 0], [tw, 0], [0, th]])
    
    # Compute the Affine transform matrix
    M = cv2.getAffineTransform(src_pts, dst_pts)
    
    # Warp the image to deskew and resize
    crop = cv2.warpAffine(img_np, M, target_size, flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_CONSTANT, borderValue=(255, 255, 255))
    
    if is_pil:
        return Image.fromarray(crop)
    return crop
