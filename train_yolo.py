from ultralytics import YOLO
import torch

def main():
    gpu_count = torch.cuda.device_count()
    is_available = torch.cuda.is_available()
    
    if is_available and gpu_count >= 1:
        device = "0"
        print(f"✅ GPU detected: {torch.cuda.get_device_name(0)}. Training on single GPU to bypass DDP subprocess issues.")
        batch_size = 32 # 32 is highly respectable and fits easily in VRAM
    else:
        device = "cpu"
        print("⚠️ No GPU detected (or CUDA drivers mismatched). Training will fall back to CPU.")
        batch_size = 8

    print(f"Starting YOLO training with batch size {batch_size}...")

    # Load the pretrained YOLOv8n-OBB model
    model = YOLO("yolov8n-obb.pt")

    # Train the model
    model.train(
        data="data.yaml",
        epochs=100,
        imgsz=1280,
        batch=batch_size,
        name="trained_on_chars_1000_pdfs",
        device=device,
        project="runs/obb"
    )
    
    print("\n🎉 Training complete! The best model weights are saved in:")
    print("runs/obb/trained_on_chars_1000_pdfs/weights/best.pt")

if __name__ == "__main__":
    main()
