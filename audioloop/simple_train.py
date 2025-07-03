import argparse
import os
import random
import re
import time

import torch
import torch.optim as optim
from torch import nn
from torch.utils.data import DataLoader

from .models.cnn_5layer import SoundCNN
from .utils.data_utils import get_device, simple_collate_fn
from .utils.spectrogram_dataset import SpectrogramDataset
from .utils.stopping_criteria import AccuracyCriterion, PlateauCriterion


def set_seed(seed):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


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

        # Calculate accuracy inline to avoid extra operations
        _, predicted = torch.max(outputs.detach(), dim=1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    avg_loss = running_loss / len(train_loader)
    accuracy = correct / total
    return avg_loss, accuracy


def run_training(
    labels_file="labels.csv",
    max_epochs=1000,
    seed=43,
    batch_size=32,
    learning_rate=1e-3,
    specs_dir="data/all_specs",
    model_path=None,
    version=None,
    use_batchnorm=None,
    stopping_criterion=None,
):
    device = get_device()
    print(f"Using device: {device}")

    set_seed(seed)

    # Create dataset from precomputed spectrograms
    # Note: This assumes you've run create_all_specs.py first to generate .pt files
    train_dataset = SpectrogramDataset(csv_file=labels_file, specs_dir=specs_dir)
    print(f"Dataset size: {len(train_dataset)}")

    # Determine number of classes from the dataset
    labels = [train_dataset[i]["label"] for i in range(len(train_dataset))]
    num_classes = len(set(labels))
    print(f"Number of classes: {num_classes}")

    # Optimize data loader for better performance
    # num_workers=2 enables parallel data loading
    # pin_memory=True speeds up GPU transfer
    # persistent_workers=True avoids worker recreation overhead
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,  # Set to 0 to avoid hanging with persistent workers
        pin_memory=torch.cuda.is_available(),
        collate_fn=simple_collate_fn,
    )

    # Decide whether to use BatchNorm based on dataset size
    if use_batchnorm is None:
        # Auto-decide: BatchNorm works well with larger datasets but can be problematic
        # with very small datasets due to poor running statistics estimation
        use_batchnorm = len(train_dataset) >= 100
        if len(train_dataset) < 100:
            print(f"⚠️  Small dataset ({len(train_dataset)} samples) detected - disabling BatchNorm")

    # Create 5-layer CNN model with appropriate architecture
    model = SoundCNN(num_classes=num_classes, kernel_size=(3, 3), use_batchnorm=use_batchnorm).to(
        device
    )
    if use_batchnorm:
        print("Using model WITH BatchNorm")
    else:
        print("Using model WITHOUT BatchNorm")

    # Print basic model info
    sample = train_dataset[0]
    sample_shape = sample["data"].shape
    print(f"Sample spectrogram shape: {sample_shape}")
    print(f"Model created with {sum(p.numel() for p in model.parameters())} parameters")

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-5)

    # Use default stopping criterion if none provided
    if stopping_criterion is None:
        stopping_criterion = PlateauCriterion(max_epochs=max_epochs)

    # Reset stopping criterion for this training run
    stopping_criterion.reset()

    print("Starting training...")
    print(f"Stopping criterion: {stopping_criterion.__class__.__name__}")
    print("-" * 50)

    # Pre-allocate timing list to avoid memory allocation during training
    epoch_times = []
    accuracy = 0.0
    for epoch in range(max_epochs):
        epoch_start_time = time.time()
        avg_loss, accuracy = train_epoch(model, train_loader, optimizer, criterion, device)

        epoch_time = time.time() - epoch_start_time
        epoch_times.append(epoch_time)

        # Print progress periodically
        if epoch % 10 == 0 or accuracy >= 1.0 or epoch < 5:
            print(
                f"Epoch {epoch + 1:4d}/{max_epochs} ({epoch_time:.2f}s) - "
                f"Loss: {avg_loss:.4f} - Accuracy: {accuracy:.4f} ({accuracy * 100:.2f}%)"
            )

        # Check stopping criterion
        if stopping_criterion.should_stop(epoch, accuracy, avg_loss):
            print()
            print("=" * 60)
            print(f"🛑 Stopping criterion met ({stopping_criterion.__class__.__name__})")
            print(f"Training completed in {epoch + 1} epochs")
            print(f"Final accuracy: {accuracy:.4f}")
            print("=" * 60)
            break
    else:
        print(f"\nTraining completed {max_epochs} epochs. Final accuracy: {accuracy:.4f}")

    # Always save the final model regardless of how training ended
    os.makedirs("outputs", exist_ok=True)
    if model_path is None:
        model_path = f"outputs/model_v{version}.pt"
    torch.save(model.state_dict(), model_path)
    print(f"Model saved to: {model_path}")

    # Clean up and return
    del train_loader
    del model
    torch.cuda.empty_cache() if torch.cuda.is_available() else None

    return accuracy


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a binary audio classification model")
    parser.add_argument("labels_file", help="Path to CSV file with training labels")
    parser.add_argument(
        "-o", "--output", help="Path to save trained model (default: auto-generated in outputs/)"
    )
    parser.add_argument(
        "-v",
        "--version",
        type=int,
        help="Model version number (default: auto-detected from training set filename)",
    )
    parser.add_argument(
        "-e", "--epochs", type=int, default=1000, help="Maximum training epochs (default: 1000)"
    )
    parser.add_argument("-s", "--seed", type=int, default=42, help="Random seed (default: 42)")
    parser.add_argument("-b", "--batch-size", type=int, default=32, help="Batch size (default: 32)")
    parser.add_argument(
        "-lr", "--learning-rate", type=float, default=0.001, help="Learning rate (default: 0.001)"
    )
    parser.add_argument(
        "--specs-dir",
        default="data/all_specs",
        help="Directory containing spectrogram files (default: data/all_specs)",
    )
    parser.add_argument(
        "--no-batchnorm",
        action="store_true",
        help="Disable BatchNorm (auto-disabled for <100 samples)",
    )
    parser.add_argument(
        "--stopping-criterion",
        choices=["accuracy", "plateau"],
        default="plateau",
        help="Stopping criterion to use (default: plateau)",
    )
    parser.add_argument(
        "--patience",
        type=int,
        default=20,
        help="Patience for plateau stopping criterion (default: 20)",
    )
    parser.add_argument(
        "--min-delta",
        type=float,
        default=0.01,
        help="Minimum delta for plateau stopping criterion (default: 0.01)",
    )

    args = parser.parse_args()

    # Auto-detect version from training set filename if not specified
    if args.version is None:
        # Try to extract version from filename like training_set_v2.csv
        match = re.search(r"_v(\d+)\.csv$", args.labels_file)
        if match:
            args.version = int(match.group(1))
            print(f"Auto-detected version {args.version} from training set filename")
        else:
            args.version = 1
            print("Could not detect version from filename, defaulting to version 1")

    # Create stopping criterion based on CLI arguments
    if args.stopping_criterion == "accuracy":
        stopping_criterion = AccuracyCriterion(max_epochs=args.epochs)
    elif args.stopping_criterion == "plateau":
        stopping_criterion = PlateauCriterion(
            patience=args.patience, min_delta=args.min_delta, max_epochs=args.epochs
        )
    else:
        stopping_criterion = PlateauCriterion(max_epochs=args.epochs)

    # Run training with CLI arguments
    accuracy = run_training(
        labels_file=args.labels_file,
        max_epochs=args.epochs,
        seed=args.seed,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        specs_dir=args.specs_dir,
        model_path=args.output,
        version=args.version,
        use_batchnorm=False if args.no_batchnorm else None,
        stopping_criterion=stopping_criterion,
    )
    print(f"\nFinal training accuracy: {accuracy:.4f}")
