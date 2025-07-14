import random
import time

import numpy as np
import torch
import torch.optim as optim
from torch import nn
from torch.utils.data import DataLoader

from .config import AudioLoopConfig
from .models import MODEL_TYPES
from .utils.data_utils import get_device, simple_collate_fn
from .utils.spectrogram_dataset import SpectrogramDataset
from .utils.stopping_criteria import PlateauCriterion


def set_seed(seed):
    """Set random seed for reproducible training."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def train_epoch(model, train_loader, optimizer, criterion, device):
    """Train model for one epoch."""
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for batch in train_loader:
        labels = batch["label"].to(device)

        # Prepare inputs using model's method
        model_inputs = model.prepare_input(batch)

        # Forward pass through model's forward method
        outputs = model.forward(model_inputs)
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


def create_model(model_type: str, num_classes: int, dataset_size: int, **kwargs):
    """Create a model based on the specified type."""
    if model_type not in MODEL_TYPES:
        raise ValueError(f"Unknown model type: {model_type}. Available: {list(MODEL_TYPES.keys())}")

    model_class = MODEL_TYPES[model_type]

    # Create model with appropriate parameters based on type
    if model_type == "soundcnn":
        return model_class(num_classes=num_classes, kernel_size=(3, 3), dataset_size=dataset_size)
    return model_class(num_classes=num_classes)


def run_training(
    config: AudioLoopConfig,
    labels_file: str = "labels.csv",
    max_epochs: int = 1000,
    seed: int = 42,
    batch_size: int = 32,
    learning_rate: float = 1e-3,
    model_path: str | None = None,
    version: int | None = None,
    use_batchnorm: bool | None = None,
    stopping_criterion=None,
    model_type: str = "soundcnn",
):
    """
    Run training for a binary audio classification model.

    Args:
        config: AudioLoopConfig with paths and experiment settings
        labels_file: Path to CSV file with training labels
        max_epochs: Maximum number of training epochs
        seed: Random seed for reproducibility
        batch_size: Training batch size
        learning_rate: Learning rate for optimizer
        model_path: Path to save trained model (auto-generated if None)
        version: Model version number (for auto-generated paths)
        use_batchnorm: Whether to use BatchNorm (auto-decided if None)
        stopping_criterion: Training stopping criterion (PlateauCriterion if None)
        model_type: Type of model to use ("soundcnn" or "simplecnn")

    Returns:
        Final training accuracy
    """
    device = get_device()
    print(f"Using device: {device}")

    set_seed(seed)

    # Create dataset from precomputed spectrograms
    train_dataset = SpectrogramDataset(csv_file=labels_file, specs_dir=str(config.specs_dir))
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
        num_workers=2,
        persistent_workers=True,  # Keep workers alive between epochs
        pin_memory=torch.cuda.is_available(),
        collate_fn=simple_collate_fn,
    )

    # Create model based on specified type
    model = create_model(
        model_type=model_type, num_classes=num_classes, dataset_size=len(train_dataset)
    ).to(device)

    # Print model info
    model_info = model.get_model_info()
    print(f"Using model: {model_info['model_type']}")
    if "use_batchnorm" in model_info:
        if model_info["use_batchnorm"]:
            print("Using model WITH BatchNorm")
        else:
            print("Using model WITHOUT BatchNorm")
        print(
            f"⚠️  Small dataset ({len(train_dataset)} samples) detected - using model WITHOUT BatchNorm"
        )

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

        # Check stopping criterion (this updates internal state)
        should_stop = stopping_criterion.should_stop(epoch, accuracy, avg_loss)

        # Update best model state if criterion indicates we should
        if stopping_criterion.should_update_best_model():
            stopping_criterion.update_best_model(model.state_dict().copy())
            print(f"    💾 Best model updated (epoch {epoch + 1})")

        # Stop if criterion says to stop
        if should_stop:
            print()
            print("=" * 60)
            print(f"🛑 Stopping criterion met ({stopping_criterion.__class__.__name__})")
            print(f"Training completed in {epoch + 1} epochs")
            print(f"Final accuracy: {accuracy:.4f}")
            print("=" * 60)
            break
    else:
        print(f"\nTraining completed {max_epochs} epochs. Final accuracy: {accuracy:.4f}")

    # Save the best model state if available, otherwise save final model
    # This ensures that when early stopping triggers (e.g., patience exhausted),
    # we save the model from the epoch with the best performance, not the final epoch
    config.create_directories()
    if model_path is None:
        model_path = str(config.get_model_path(version or 1))

    best_model_state = stopping_criterion.get_best_model_state()
    if best_model_state is not None:
        # For best model state, we need to load it into the model and save using the new method
        model.load_state_dict(best_model_state)
        model.save_model(model_path)
        print(f"✅ Best model saved to: {model_path}")
    else:
        model.save_model(model_path)
        print(f"📁 Final model saved to: {model_path}")

    # Clean up and return
    del train_loader
    del model
    torch.cuda.empty_cache() if torch.cuda.is_available() else None

    return accuracy
