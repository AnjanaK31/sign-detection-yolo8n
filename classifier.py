import os
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
    'diameter',       # ∅ (or ⌀)
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

# Mapping from index to class and class to index
IDX_TO_CLASS = {i: c for i, c in enumerate(CLASSES)}
CLASS_TO_IDX = {c: i for i, c in enumerate(CLASSES)}

# Preprocessing transforms for MobileNetV3 inference/training
def get_transforms(img_size=64):
    return transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

def get_mobilenet_v3_small(num_classes):
    """Returns a MobileNetV3-Small model with pretrained weights for rapid transfer learning."""
    try:
        # Try loading via newer weights API
        from torchvision.models import MobileNet_V3_Small_Weights
        model = models.mobilenet_v3_small(weights=MobileNet_V3_Small_Weights.DEFAULT)
    except Exception:
        try:
            # Fallback to older pretrained parameter
            model = models.mobilenet_v3_small(pretrained=True)
        except Exception:
            # Final fallback to random initialization
            model = models.mobilenet_v3_small(pretrained=False)
    
    # Replace the last linear layer of the classifier
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
        self.transform = get_transforms(self.img_size)
        
    def predict(self, crop_image, target_percent=0.6):
        """Runs inference on a cropped PIL image. Pads and scales the image to match the training dataset scale."""
        # Ensure RGB
        if isinstance(crop_image, Image.Image):
            img = crop_image.convert("RGB")
        else:
            # Assume numpy array
            img = Image.fromarray(crop_image).convert("RGB")
            
        w, h = img.size
        max_dim = max(w, h, 1)
        
        # Scale the character so its max dimension is target_percent of self.img_size
        scale = (self.img_size * target_percent) / max_dim
        new_w = max(1, int(round(w * scale)))
        new_h = max(1, int(round(h * scale)))
        
        # Resize using BICUBIC interpolation
        resized_img = img.resize((new_w, new_h), Image.BICUBIC)
        
        # Create a new white canvas and paste the resized image in the center
        canvas = Image.new("RGB", (self.img_size, self.img_size), (255, 255, 255))
        paste_x = (self.img_size - new_w) // 2
        paste_y = (self.img_size - new_h) // 2
        canvas.paste(resized_img, (paste_x, paste_y))
        
        # Transform and convert to batch tensor
        tensor = self.transform(canvas).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            outputs = self.model(tensor)
            probabilities = torch.softmax(outputs, dim=1)
            conf, pred_idx = torch.max(probabilities, dim=1)
            
        class_idx = pred_idx.item()
        confidence = conf.item()
        class_name = IDX_TO_CLASS[class_idx]
        
        return class_name, confidence

    def _segment_characters(self, crop_image):
        """Finds horizontal character segments in the crop and returns lists of character PIL Images and segments."""
        img_np = np.array(crop_image.convert("L"))
        h_orig, w_orig = img_np.shape
        
        # Binarize (assuming white background, black text)
        _, thresh = cv2.threshold(img_np, 127, 255, cv2.THRESH_BINARY_INV)
        
        # Clean noise (speckles) via morphological opening
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
        thresh_cleaned = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
        
        # Project middle 80% height of the cleaned threshold to avoid CAD lines
        thresh_center = thresh_cleaned[int(h_orig * 0.1):int(h_orig * 0.9), :]
        proj = np.sum(thresh_center, axis=0)
        
        # Require at least 2 white pixels in the column to filter remaining noise
        text_cols = np.where(proj >= 2 * 255)[0]
        
        if len(text_cols) == 0:
            return [], []
            
        # Find continuous segments of text columns (gap > 2 pixels)
        segments = []
        start = text_cols[0]
        for i in range(1, len(text_cols)):
            if text_cols[i] > text_cols[i-1] + 2:
                end = text_cols[i-1]
                segments.append((start, end))
                start = text_cols[i]
        segments.append((start, text_cols[-1]))
        
        # Filter out very thin segments (e.g. less than 2 pixels wide)
        segments = [seg for seg in segments if (seg[1] - seg[0]) >= 2]
        
        char_crops = []
        for start_x, end_x in segments:
            # Add horizontal padding for context
            pad_x = 2
            x_min = max(0, start_x - pad_x)
            x_max = min(w_orig, end_x + pad_x)
            
            # Slice the middle 80% height of the cleaned threshold for this segment
            # to find the vertical boundaries of the character
            seg_thresh = thresh_cleaned[int(h_orig * 0.1):int(h_orig * 0.9), x_min:x_max]
            row_indices = np.where(np.sum(seg_thresh, axis=1) > 0)[0]
            
            if len(row_indices) > 0:
                # Map back to original height coordinates
                y_min = int(h_orig * 0.1) + row_indices[0]
                y_max = int(h_orig * 0.1) + row_indices[-1]
                
                # Add vertical padding (e.g. 2 pixels)
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
        """Segment a cropped expression image into individual characters and predict each.
        Determines the best orientation (0, 90, 180, 270) using character-level voting.
        Returns:
            (final_str, avg_conf, rectified_crop_image, char_details)
        """
        # Ensure crop_image is PIL Image
        if not isinstance(crop_image, Image.Image):
            crop_image = Image.fromarray(crop_image)
            
        # Determine background fill color (white)
        fill_color = (255, 255, 255) if crop_image.mode == "RGB" else 255
            
        # Ensure text is horizontal (width >= height)
        w, h = crop_image.size
        if h > w:
            crop_image = crop_image.rotate(90, expand=True, fillcolor=fill_color)
            
        # 1. Segment characters on the horizontal crop to get bounding boxes (x-spans)
        char_crops, segments = self._segment_characters(crop_image)
        
        if len(char_crops) == 0:
            # Fallback to single prediction if no characters could be segmented
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
            
        # 2. For each character segment, test all 4 orientations and vote
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
            
            # Vote for the best angle for this character
            votes[best_char_angle] += 1
            
        # 3. Determine the winning angle
        # Primary key: number of votes. Secondary key (tie-breaker): sum of confidences.
        winning_angle = max(votes.keys(), key=lambda k: (votes[k], conf_sums[k]))
        
        # 4. Rotate the entire crop by the winning angle
        rectified_crop = crop_image.rotate(winning_angle, fillcolor=fill_color)
        
        # 5. Re-segment the rectified crop to get the characters in correct order
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
