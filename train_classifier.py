import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from classifier import (
    get_mobilenet_v3_small,
    get_train_transforms,
    get_val_transforms,
    CLASSES, CLASS_TO_IDX
)


def train_classifier(
    epochs=30,
    batch_size=32,
    lr=0.0005,
    dataset_dir="dataset_classifier",
    model_save_path="classifier_best.pt",
    patience=7,          # early-stopping patience (epochs without improvement)
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training MobileNetV3 classifier on device: {device}")
    print(f"  epochs={epochs}  lr={lr}  batch={batch_size}  patience={patience}")

    # ── Data loaders ─────────────────────────────────────────────────────────
    img_size  = 64
    train_dir = os.path.join(dataset_dir, "train")
    val_dir   = os.path.join(dataset_dir, "val")

    if not os.path.exists(train_dir) or not os.path.exists(val_dir):
        print(f"ERROR: Dataset not found at {dataset_dir}. Run data_gen.py first.")
        return

    # Training uses online augmentation; validation uses clean transforms
    train_dataset = ImageFolder(root=train_dir, transform=get_train_transforms(img_size))
    val_dataset   = ImageFolder(root=val_dir,   transform=get_val_transforms(img_size))

    # Map ImageFolder's alphabetical indices → our CLASSES order
    folder_to_class = {v: k for k, v in train_dataset.class_to_idx.items()}
    mapping_tensor  = torch.zeros(len(train_dataset.classes), dtype=torch.long)
    for folder_idx, class_name in folder_to_class.items():
        if class_name in CLASS_TO_IDX:
            mapping_tensor[folder_idx] = CLASS_TO_IDX[class_name]
        else:
            print(f"WARNING: class '{class_name}' not in CLASS_TO_IDX")
    mapping_tensor = mapping_tensor.to(device)

    train_loader = DataLoader(train_dataset, batch_size=batch_size,
                              shuffle=True,  num_workers=0)
    val_loader   = DataLoader(val_dataset,   batch_size=batch_size,
                              shuffle=False, num_workers=0)

    print(f"  Train: {len(train_dataset)} samples  |  Val: {len(val_dataset)} samples")

    # ── Model, loss, optimiser ────────────────────────────────────────────────
    model     = get_mobilenet_v3_small(len(CLASSES))
    model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    # Cosine-annealing LR: smoothly decays lr → 0 over all epochs
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)

    best_val_acc     = 0.0
    epochs_no_improve = 0

    # ── Training loop ─────────────────────────────────────────────────────────
    for epoch in range(epochs):
        # ── Train ─────────────────────────────────────────────────────────────
        model.train()
        running_loss   = 0.0
        correct_train  = 0
        total_train    = 0

        for images, labels in train_loader:
            images = images.to(device)
            labels = mapping_tensor[labels.to(device)]

            optimizer.zero_grad()
            outputs = model(images)
            loss    = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss  += loss.item() * images.size(0)
            _, predicted   = torch.max(outputs, 1)
            correct_train += (predicted == labels).sum().item()
            total_train   += labels.size(0)

        train_loss = running_loss / total_train
        train_acc  = correct_train / total_train

        # ── Validate ───────────────────────────────────────────────────────────
        model.eval()
        val_loss_sum  = 0.0
        correct_val   = 0
        total_val     = 0

        with torch.no_grad():
            for images, labels in val_loader:
                images  = images.to(device)
                labels  = mapping_tensor[labels.to(device)]
                outputs = model(images)
                loss    = criterion(outputs, labels)
                val_loss_sum  += loss.item() * images.size(0)
                _, predicted   = torch.max(outputs, 1)
                correct_val   += (predicted == labels).sum().item()
                total_val     += labels.size(0)

        val_loss = val_loss_sum / total_val
        val_acc  = correct_val  / total_val

        current_lr = scheduler.get_last_lr()[0]
        scheduler.step()

        marker = ""
        if val_acc >= best_val_acc:
            best_val_acc      = val_acc
            epochs_no_improve = 0
            torch.save(model.state_dict(), model_save_path)
            marker = f"  ✓ saved (best={best_val_acc:.4f})"
        else:
            epochs_no_improve += 1

        print(
            f"Epoch {epoch+1:03d}/{epochs} | "
            f"lr={current_lr:.2e} | "
            f"Train  loss={train_loss:.4f}  acc={train_acc:.4f} | "
            f"Val  loss={val_loss:.4f}  acc={val_acc:.4f}"
            f"{marker}"
        )

        # ── Early stopping ─────────────────────────────────────────────────────
        if epochs_no_improve >= patience:
            print(f"\nEarly stopping triggered — no improvement for {patience} epochs.")
            break

    print(f"\nTraining complete.  Best val accuracy: {best_val_acc:.4f}")
    print(f"Best weights saved to: {model_save_path}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Train MobileNetV3 Character Classifier")
    parser.add_argument("--epochs",    type=int,   default=30)
    parser.add_argument("--batch",     type=int,   default=32)
    parser.add_argument("--lr",        type=float, default=0.0005)
    parser.add_argument("--patience",  type=int,   default=7)
    parser.add_argument("--dataset",   default="dataset_classifier")
    parser.add_argument("--save-path", default="classifier_best.pt")
    args = parser.parse_args()

    train_classifier(
        epochs          = args.epochs,
        batch_size      = args.batch,
        lr              = args.lr,
        dataset_dir     = args.dataset,
        model_save_path = args.save_path,
        patience        = args.patience,
    )

