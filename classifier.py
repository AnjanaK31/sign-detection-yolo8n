import os
import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image

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
