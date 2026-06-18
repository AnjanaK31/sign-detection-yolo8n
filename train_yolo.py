from ultralytics import YOLO
import torch
import argparse

def main():
    parser = argparse.ArgumentParser(description="YOLO OBB Training Script")
    parser.add_argument("--epochs", type=int, default=100, help="Number of training epochs")
    parser.add_argument("--batch", type=int, default=None, help="Batch size (default: auto-detected based on GPU)")
    parser.add_argument("--model", default="yolov8n-obb.pt", help="Pretrained YOLO model path")
    parser.add_argument("--name", default="trained_on_chars_1000_pdfs", help="Name of training run")
    parser.add_argument("--project", default="runs/obb", help="Project output directory")
    args = parser.parse_args()

    gpu_count = torch.cuda.device_count()
    is_available = torch.cuda.is_available()
    
    if is_available and gpu_count >= 1:
        device = "0"
        print(f"✅ GPU detected: {torch.cuda.get_device_name(0)}. Training on single GPU to bypass DDP subprocess issues.")
        batch_size = 32 if args.batch is None else args.batch
    else:
        device = "cpu"
        print("⚠️ No GPU detected (or CUDA drivers mismatched). Training will fall back to CPU.")
        batch_size = 8 if args.batch is None else args.batch

    print(f"Starting YOLO training with batch size {batch_size} for {args.epochs} epochs...")

    # Load the pretrained YOLO model
    model = YOLO(args.model)

    # Train the model
    model.train(
        data="data.yaml",
        epochs=args.epochs,
        imgsz=1280,
        batch=batch_size,
        name=args.name,
        device=device,
        project=args.project
    )
    
    print("\n🎉 Training complete! The best model weights are saved in:")
    print(f"{args.project}/{args.name}/weights/best.pt")

if __name__ == "__main__":
    main()
