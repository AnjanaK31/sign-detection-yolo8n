import os
import torch
from ultralytics import YOLO

def train_yolo_obb(epochs=20, imgsz=640, batch_size=4):
    """Initializes and trains a YOLOv8n-OBB model on the synthetic equation dataset."""
    print("Initializing YOLOv8n-OBB model...")
    # Load a pre-trained YOLOv8n-OBB model. It will auto-download the weights if not present.
    model = YOLO("yolov8n-obb.pt")
    
    device = "0" if torch.cuda.is_available() else "cpu"
    print(f"Training YOLOv8n-OBB on device: {device}")
    
    # Train the model
    # We set workers=0 on Windows to prevent potential multiprocessing/dataloader errors.
    model.train(
        data="data.yaml",
        epochs=epochs,
        imgsz=imgsz,
        batch=batch_size,
        device=device,
        workers=0,
        project="yolo_obb_project",
        name="symbol_obb_train"
    )
    print("YOLOv8n-OBB training finished successfully!")

if __name__ == "__main__":
    train_yolo_obb(epochs=20, imgsz=640, batch_size=4)
