import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from classifier import get_mobilenet_v3_small, get_transforms, CLASSES, CLASS_TO_IDX

def train_classifier(epochs=5, batch_size=32, lr=0.001, dataset_dir="dataset_classifier", model_save_path="classifier_best.pt"):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training MobileNetV3 classifier on device: {device}")
    
    # 1. Prepare data loaders
    img_size = 64
    transform = get_transforms(img_size)
    
    train_dir = os.path.join(dataset_dir, "train")
    val_dir = os.path.join(dataset_dir, "val")
    
    if not os.path.exists(train_dir) or not os.path.exists(val_dir):
        print(f"ERROR: Dataset directories not found at {dataset_dir}. Run data_gen.py first.")
        return
        
    train_dataset = ImageFolder(root=train_dir, transform=transform)
    val_dataset = ImageFolder(root=val_dir, transform=transform)
    
    # Verify dataset indices match classifier classes
    # ImageFolder assigns indices alphabetically, which might not match CLASSES order.
    # We map the ImageFolder indices to our CLASSES indices.
    folder_to_class = {v: k for k, v in train_dataset.class_to_idx.items()}
    mapping_tensor = torch.zeros(len(train_dataset.classes), dtype=torch.long)
    for folder_idx, class_name in folder_to_class.items():
        if class_name in CLASS_TO_IDX:
            mapping_tensor[folder_idx] = CLASS_TO_IDX[class_name]
        else:
            print(f"WARNING: Class name {class_name} not found in CLASS_TO_IDX")
            
    mapping_tensor = mapping_tensor.to(device)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    
    print(f"Loaded {len(train_dataset)} training samples, {len(val_dataset)} validation samples.")
    
    # 2. Initialize Model, Loss, Optimizer
    model = get_mobilenet_v3_small(len(CLASSES))
    model.to(device)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    best_val_acc = 0.0
    
    # 3. Training Loop
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        correct_train = 0
        total_train = 0
        
        for images, labels in train_loader:
            images = images.to(device)
            # Map labels to our custom CLASS_TO_IDX indices
            labels = mapping_tensor[labels.to(device)]
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item() * images.size(0)
            _, predicted = torch.max(outputs, 1)
            correct_train += (predicted == labels).sum().item()
            total_train += labels.size(0)
            
        epoch_loss = running_loss / total_train
        epoch_train_acc = correct_train / total_train
        
        # Validation Phase
        model.eval()
        correct_val = 0
        total_val = 0
        val_loss = 0.0
        
        with torch.no_grad():
            for images, labels in val_loader:
                images = images.to(device)
                labels = mapping_tensor[labels.to(device)]
                
                outputs = model(images)
                loss = criterion(outputs, labels)
                val_loss += loss.item() * images.size(0)
                
                _, predicted = torch.max(outputs, 1)
                correct_val += (predicted == labels).sum().item()
                total_val += labels.size(0)
                
        epoch_val_loss = val_loss / total_val
        epoch_val_acc = correct_val / total_val
        
        print(f"Epoch {epoch+1}/{epochs} - Train Loss: {epoch_loss:.4f}, Train Acc: {epoch_train_acc:.4f} | Val Loss: {epoch_val_loss:.4f}, Val Acc: {epoch_val_acc:.4f}")
        
        # Save best model
        if epoch_val_acc >= best_val_acc:
            best_val_acc = epoch_val_acc
            torch.save(model.state_dict(), model_save_path)
            print(f"--> Saved best model weights with accuracy: {best_val_acc:.4f}")
            
    print(f"Training complete! Best validation accuracy: {best_val_acc:.4f}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Train MobileNetV3 Character Classifier")
    parser.add_argument("--epochs", type=int, default=5, help="Number of training epochs")
    parser.add_argument("--batch", type=int, default=32, help="Batch size")
    parser.add_argument("--lr", type=float, default=0.001, help="Learning rate")
    parser.add_argument("--dataset", default="dataset_classifier", help="Dataset directory")
    parser.add_argument("--save-path", default="classifier_best.pt", help="Path to save best weights")
    args = parser.parse_args()
    
    train_classifier(
        epochs=args.epochs,
        batch_size=args.batch,
        lr=args.lr,
        dataset_dir=args.dataset,
        model_save_path=args.save_path
    )
