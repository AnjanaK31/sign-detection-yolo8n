import os
import random
import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
import cv2
import numpy as np
import io
import base64

def pil_to_base64(img: Image.Image, format="PNG") -> str:
    buffered = io.BytesIO()
    img.save(buffered, format=format)
    img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")
    return f"data:image/{format.lower()};base64,{img_str}"

# Class to character mapping for display and transcription
CLASS_TO_CHAR = {
    '0': '0', '1': '1', '2': '2', '3': '3', '4': '4',
    '5': '5', '6': '6', '7': '7', '8': '8', '9': '9',
    'plus_minus': '+/-',
    'diameter': 'DIA',
    'radius': 'R',
    'Rz': 'Rz',
    'Ra': 'Ra',
    'perpendicular': 'PERP',
    'parallel': 'PARA',
    'circularity': 'CIRC',
    'true_position': 'TP',
    'arrow': 'Arrow',
    'comma': ','
}

# 21 classes matching the requirements document
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

IDX_TO_CLASS = {i: c for i, c in enumerate(CLASSES)}
CLASS_TO_IDX = {c: i for i, c in enumerate(CLASSES)}

class AddGaussianNoise:
    """Adds zero-mean Gaussian noise to a tensor (applied after ToTensor)."""
    def __init__(self, std_range=(0.01, 0.08)):
        self.std_range = std_range

    def __call__(self, tensor):
        std = random.uniform(*self.std_range)
        return (tensor + torch.randn_like(tensor) * std).clamp(0.0, 1.0)

def get_train_transforms(img_size=64):
    return transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.RandomRotation(degrees=15),
        transforms.RandomAffine(
            degrees=0,
            translate=(0.06, 0.06),
            shear=(-8, 8)
        ),
        transforms.ColorJitter(brightness=0.35, contrast=0.35),
        transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 1.5)),
        transforms.ToTensor(),
        AddGaussianNoise(std_range=(0.01, 0.06)),
        transforms.RandomErasing(
            p=0.25,
            scale=(0.01, 0.06),
            ratio=(0.3, 3.0),
            value=1.0
        ),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        ),
    ])

def get_val_transforms(img_size=64):
    return transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        ),
    ])

def get_mobilenet_v3_small(num_classes):
    try:
        from torchvision.models import MobileNet_V3_Small_Weights
        model = models.mobilenet_v3_small(weights=MobileNet_V3_Small_Weights.DEFAULT)
    except Exception:
        try:
            model = models.mobilenet_v3_small(pretrained=True)
        except Exception:
            model = models.mobilenet_v3_small(pretrained=False)
    
    in_features = model.classifier[3].in_features
    model.classifier[3] = nn.Linear(in_features, num_classes)
    return model

class SymbolClassifier:
    """Wrapper class for loading and running predictions with the MobileNetV3 classifier."""
    def __init__(self, model_path=None, img_size=64, device=None):
        self.img_size = img_size
        self.device = device if device else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = get_mobilenet_v3_small(len(CLASSES))
        
        if model_path and os.path.exists(model_path):
            print(f"Loading MobileNetV3 classifier weights from {model_path}...")
            self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        else:
            if model_path:
                print(f"WARNING: Model path {model_path} does not exist. Initializing empty model.")
                
        self.model.to(self.device)
        self.model.eval()
        self.transform = get_val_transforms(self.img_size)
        
    def predict(self, crop_image, target_percent=0.6):
        """Runs inference on a cropped PIL image. Pads and scales the image to match the training dataset scale."""
        if isinstance(crop_image, Image.Image):
            img = crop_image.convert("RGB")
        else:
            img = Image.fromarray(crop_image).convert("RGB")
            
        rotations = [0, 90, 270]
        batch_tensors = []
        
        for rot in rotations:
            if rot == 0:
                rot_img = img
            else:
                rot_img = img.rotate(rot, expand=True)
                
            w, h = rot_img.size
            max_dim = max(w, h, 1)
            
            scale = (self.img_size * target_percent) / max_dim
            new_w = max(1, int(round(w * scale)))
            new_h = max(1, int(round(h * scale)))
            
            resized_img = rot_img.resize((new_w, new_h), Image.BICUBIC)
            
            canvas = Image.new("RGB", (self.img_size, self.img_size), (255, 255, 255))
            paste_x = (self.img_size - new_w) // 2
            paste_y = (self.img_size - new_h) // 2
            canvas.paste(resized_img, (paste_x, paste_y))
            
            tensor = self.transform(canvas)
            batch_tensors.append(tensor)
            
        batch_tensor = torch.stack(batch_tensors).to(self.device)
        
        with torch.no_grad():
            outputs = self.model(batch_tensor)
            probabilities = torch.softmax(outputs, dim=1)
            conf_vals, class_idxs = torch.max(probabilities, dim=1)
            best_rot_idx = torch.argmax(conf_vals).item()
            
            confidence = conf_vals[best_rot_idx].item()
            class_idx = class_idxs[best_rot_idx].item()
            class_name = IDX_TO_CLASS[class_idx]
            
        return class_name, confidence

    def _segment_characters(self, crop_image):
        img_np = np.array(crop_image.convert("L"))
        h_orig, w_orig = img_np.shape
        
        _, thresh = cv2.threshold(img_np, 127, 255, cv2.THRESH_BINARY_INV)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
        thresh_cleaned = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
        
        thresh_center = thresh_cleaned[int(h_orig * 0.1):int(h_orig * 0.9), :]
        proj = np.sum(thresh_center, axis=0)
        
        text_cols = np.where(proj >= 2 * 255)[0]
        
        if len(text_cols) == 0:
            return [], []
            
        segments = []
        start = text_cols[0]
        for i in range(1, len(text_cols)):
            if text_cols[i] > text_cols[i-1] + 2:
                end = text_cols[i-1]
                segments.append((start, end))
                start = text_cols[i]
        segments.append((start, text_cols[-1]))
        
        segments = [seg for seg in segments if (seg[1] - seg[0]) >= 2]
        
        char_crops = []
        for start_x, end_x in segments:
            pad_x = 2
            x_min = max(0, start_x - pad_x)
            x_max = min(w_orig, end_x + pad_x)
            
            seg_thresh = thresh_cleaned[int(h_orig * 0.1):int(h_orig * 0.9), x_min:x_max]
            row_indices = np.where(np.sum(seg_thresh, axis=1) > 0)[0]
            
            if len(row_indices) > 0:
                y_min = int(h_orig * 0.1) + row_indices[0]
                y_max = int(h_orig * 0.1) + row_indices[-1]
                pad_y = 2
                y_min = max(0, y_min - pad_y)
                y_max = min(h_orig, y_max + pad_y)
            else:
                y_min = int(h_orig * 0.1)
                y_max = int(h_orig * 0.9)
                
            char_crop = img_np[y_min:y_max, x_min:x_max]
            char_crops.append(Image.fromarray(char_crop))
            
        return char_crops, segments

    def predict_expression(self, crop_image):
        if not isinstance(crop_image, Image.Image):
            crop_image = Image.fromarray(crop_image)
            
        fill_color = (255, 255, 255) if crop_image.mode == "RGB" else 255
        w, h = crop_image.size
        if h > w:
            crop_image = crop_image.rotate(90, expand=True, fillcolor=fill_color)
            
        char_crops, segments = self._segment_characters(crop_image)
        
        if len(char_crops) == 0:
            best_angle = 0
            best_conf = -1.0
            best_class = ""
            for angle in [0, 90, 180, 270]:
                rotated_char = crop_image.rotate(angle, fillcolor=fill_color)
                class_name, confidence = self.predict(rotated_char)
                if confidence > best_conf:
                    best_conf = confidence
                    best_class = class_name
                    best_angle = angle
            
            char_display = CLASS_TO_CHAR.get(best_class, best_class)
            rectified_char = crop_image.rotate(best_angle, fillcolor=fill_color)
            char_details = [{
                'char': char_display,
                'confidence': float(best_conf),
                'image': pil_to_base64(rectified_char)
            }]
            return char_display, best_conf, rectified_char, char_details
            
        votes = {0: 0, 90: 0, 180: 0, 270: 0}
        conf_sums = {0: 0.0, 90: 0.0, 180: 0.0, 270: 0.0}
        
        for char_crop in char_crops:
            best_char_angle = 0
            best_char_conf = -1.0
            fill_color_char = (255, 255, 255) if char_crop.mode == "RGB" else 255
            for angle in [0, 90, 180, 270]:
                rotated_char = char_crop.rotate(angle, fillcolor=fill_color_char)
                _, confidence = self.predict(rotated_char)
                
                conf_sums[angle] += confidence
                if confidence > best_char_conf:
                    best_char_conf = confidence
                    best_char_angle = angle
            
            votes[best_char_angle] += 1
            
        winning_angle = max(votes.keys(), key=lambda k: (votes[k], conf_sums[k]))
        rectified_crop = crop_image.rotate(winning_angle, fillcolor=fill_color)
        final_char_crops, final_segments = self._segment_characters(rectified_crop)
        
        if len(final_char_crops) == 0:
            class_name, confidence = self.predict(rectified_crop)
            char_display = CLASS_TO_CHAR.get(class_name, class_name)
            char_details = [{
                'char': char_display,
                'confidence': float(confidence),
                'image': pil_to_base64(rectified_crop)
            }]
            return char_display, confidence, rectified_crop, char_details
            
        predicted_chars = []
        confidences = []
        char_details = []
        
        for char_crop in final_char_crops:
            class_name, confidence = self.predict(char_crop)
            char_display = CLASS_TO_CHAR.get(class_name, class_name)
            predicted_chars.append(char_display)
            confidences.append(confidence)
            char_details.append({
                'char': char_display,
                'confidence': float(confidence),
                'image': pil_to_base64(char_crop)
            })
            
        final_str = "".join(predicted_chars)
        avg_conf = np.mean(confidences) if confidences else 0.0
        
        return final_str, avg_conf, rectified_crop, char_details
