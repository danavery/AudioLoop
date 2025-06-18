import os
import random
import time

import torch
import torch.nn.functional as F
import torch.optim as optim
from torch import nn
from torch.utils.data import DataLoader

from .models.simple_cnn import SimpleCNN
from .utils.spec_dataset import SpectrogramDataset


def set_seed(seed):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def collate_fn(batch):
    """Simple collate function for fixed-length spectrograms."""
    # Extract data and other fields
    data_list = [item["data"] for item in batch]
    labels = torch.tensor([item["label"] for item in batch])

    # Convert to mono by averaging channels if stereo
    mono_data = []
    for spec in data_list:
        if spec.shape[0] > 1:
            spec = spec.mean(dim=0, keepdim=True)  # Average channels, keep dimension
        mono_data.append(spec)

    # Stack the spectrograms (they should all be the same size now)
    data_tensor = torch.stack(mono_data)

    # Return in the same format as the original batch
    return {
        "data": data_tensor,
        "label": labels,
        "filename": [item["filename"] for item in batch],
        "filepath": [item["filepath"] for item in batch],
        "run": [item["run"] for item in batch]
    }


def train_epoch(model, train_loader, optimizer, criterion, device):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for batch in train_loader:
        features = batch["data"].to(device)
        labels = batch["label"].to(device)

        # If features are spectrograms with shape (batch, n_mels, time), add channel dim:
        if features.ndim == 3:
            features = features.unsqueeze(1)  # shape: (batch, 1, n_mels, time)

        # Forward pass
        outputs = model(features)
        loss = criterion(outputs, labels)

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

        # Calculate accuracy
        _, predicted = torch.max(outputs, dim=1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()



    avg_loss = running_loss / len(train_loader)
    accuracy = correct / total
    return avg_loss, accuracy




def run_training(labels_file="labels.csv", max_epochs=1000, seed=42):
    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Using device: {device}")

    set_seed(seed)
    accuracy = 0.0  # Initialize accuracy

    # Create dataset from precomputed spectrograms
    # Note: This assumes you've run create_all_specs.py first to generate .pt files
    train_dataset = SpectrogramDataset(labels_file=labels_file)
    print(f"Dataset size: {len(train_dataset)}")

    # Determine number of classes from the dataset
    labels = [train_dataset[i]["label"] for i in range(len(train_dataset))]
    num_classes = len(set(labels))
    print(f"Number of classes: {num_classes}")

    train_loader = DataLoader(
        train_dataset,
        batch_size=32,
        shuffle=True,
        num_workers=0,
        pin_memory=False,
        collate_fn=collate_fn
    )

    # Create simpler model for debugging
    model = SimpleCNN(num_classes=num_classes).to(device)

    # Print basic model info
    sample = train_dataset[0]
    sample_shape = sample["data"].shape
    print(f"Sample spectrogram shape: {sample_shape}")
    print(f"Model created with {sum(p.numel() for p in model.parameters())} parameters")

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)

    print("Model parameters:")
    for name, param in model.named_parameters():
        print(f"  {name}: {param.shape}")

    print("Starting training...")
    print("Target: 100% training accuracy")
    print("-" * 50)

    for epoch in range(max_epochs):
        epoch_start_time = time.time()
        avg_loss, accuracy = train_epoch(model, train_loader, optimizer, criterion, device)

        epoch_time = time.time() - epoch_start_time

        if epoch % 5 == 0 or accuracy >= 1.0:
            print(
                f"Epoch {epoch + 1:4d}/{max_epochs} ({epoch_time:.2f}s) - "
                f"Loss: {avg_loss:.4f} - Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)"
            )

            # Debug: print a few predictions
            if epoch % 10 == 0:
                model.eval()
                with torch.no_grad():
                    sample_batch = next(iter(train_loader))
                    features = sample_batch["data"].to(device)
                    labels = sample_batch["label"].to(device)
                    if features.ndim == 3:
                        features = features.unsqueeze(1)

                    outputs = model(features)
                    probs = F.softmax(outputs, dim=-1)
                    preds = torch.argmax(probs, dim=-1)

                    for i in range(min(3, len(labels))):
                        print(f"    Sample {i}: True={labels[i].item()}, Pred={preds[i].item()}, "
                              f"Conf={torch.max(probs[i]).item():.3f}")
                model.train()

        # Check if we've reached 100% accuracy
        if accuracy >= 1.0:
            print()
            print("=" * 60)
            print("🎉 Reached 100% training accuracy!")
            print(f"Training completed in {epoch + 1} epochs")
            print("=" * 60)

            # Save the final model
            os.makedirs("outputs", exist_ok=True)
            model_path = f"outputs/model_100pct_seed_{seed}.pt"
            torch.save(model.state_dict(), model_path)
            print(f"Model saved to: {model_path}")
            break
    else:
        print(f"\nTraining completed {max_epochs} epochs. Final accuracy: {accuracy:.4f}")
        # Save the final model even if we didn't reach 100%
        os.makedirs("outputs", exist_ok=True)
        model_path = f"outputs/model_final_seed_{seed}.pt"
        torch.save(model.state_dict(), model_path)
        print(f"Model saved to: {model_path}")

    return accuracy


if __name__ == "__main__":
    # Simple training run
    final_accuracy = run_training(
        labels_file="labels.csv",
        max_epochs=1000,
        seed=42
    )
    print(f"\nFinal training accuracy: {final_accuracy:.4f}")
