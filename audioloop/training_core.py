import logging
import random
import time

import numpy as np
import torch
import torch.optim as optim
from torch import nn
from torch.utils.data import DataLoader

from .config import AudioLoopConfig
from .models.model_registry import get_model_class, list_available_models
from .utils.data_utils import get_device, variable_length_collate_fn
from .utils.spectrogram_dataset import SpectrogramDataset
from .utils.stopping_criteria import create_stopping_criterion

# Set up logger for this module
logger = logging.getLogger(__name__)


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
        features = batch["data"].to(device)

        # Standard PyTorch forward pass
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


def create_model(model_type: str, num_classes: int, dataset_size: int, **kwargs):
    """Create a model based on the specified type."""
    try:
        model_class = get_model_class(model_type)
    except ValueError:
        available = list_available_models()
        raise ValueError(
            f"Unknown model type: {model_type}. Available: {', '.join(sorted(available))}"
        ) from None

    # Create model with flexible kwargs - let each model handle its own parameters
    return model_class(num_classes=num_classes, dataset_size=dataset_size, **kwargs)


def run_training(
    config: AudioLoopConfig,
    labels_file: str,
    version: int,
    model_path: str | None = None,
    log_level: int = logging.INFO,
):
    """
    Run training for a binary audio classification model using config-driven parameters.

    Args:
        config: AudioLoopConfig with all training settings (epochs, batch_size, learning_rate, etc.)
        labels_file: Path to CSV file with training labels
        version: Model version number (for workflow tracking)
        model_path: Path to save trained model (uses config.get_model_path(version) if None)
        log_level: Logging level (logging.INFO for normal output, logging.WARNING for quiet)

    Returns:
        Tuple of (final training accuracy, number of epochs trained)
    """
    # Set up logging for this training run
    logger.setLevel(log_level)

    device = get_device()
    logger.info(f"Using device: {device}")

    set_seed(config.seed)

    # Get dataset config for lazy spec generation
    dataset_config = config.get_dataset_config()

    # Create dataset with lazy spec generation support
    train_dataset = SpectrogramDataset(
        csv_file=labels_file,
        specs_dir=str(config.specs_dir),
        dataset_config=dataset_config,  # Enable lazy generation from audio_path
    )
    logger.info(f"Dataset size: {len(train_dataset)}")

    # Determine number of classes from the dataset
    labels = [sample["label"] for sample in train_dataset.samples]
    num_classes = len(set(labels))
    logger.info(f"Number of classes: {num_classes}")

    # Display training set composition (always visible, not just at INFO level)
    positive_count = sum(1 for label in labels if label == 1)
    negative_count = sum(1 for label in labels if label == 0)
    print(
        f"Training set: {len(train_dataset)} total ({positive_count} positive, {negative_count} negative)"
    )

    # Optimize data loader for better performance
    # num_workers=2 enables parallel data loading
    # pin_memory=True speeds up GPU transfer
    # persistent_workers=True avoids worker recreation overhead
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=2,
        persistent_workers=True,  # Keep workers alive between epochs
        pin_memory=torch.cuda.is_available(),
        collate_fn=variable_length_collate_fn,
    )

    # Create model based on config
    model = create_model(
        model_type=config.model_type,
        num_classes=num_classes,
        dataset_size=len(train_dataset),
        **config.model_kwargs,
    ).to(device)

    # Check dataset/model compatibility
    dataset_shape = dataset_config.get_output_shape()

    if not model.can_handle_shape(dataset_shape):
        available_models = list_available_models()
        raise ValueError(
            f"Model '{config.model_type}' cannot handle tensors with shape {dataset_shape} "
            f"from dataset '{config.dataset}'. "
            f"Available models: {available_models}. "
            f"Try a different model with --model-type or use a compatible dataset."
        )

    logger.info(f"✅ Dataset shape {dataset_shape} is compatible with model '{config.model_type}'")

    # Print model info
    model_info = model.get_model_info()
    logger.info(f"Using model: {model_info['model_type']}")
    if "use_batchnorm" in model_info:
        if model_info["use_batchnorm"]:
            logger.info("Using model WITH BatchNorm")
        else:
            logger.info(
                f"⚠️  Small dataset ({len(train_dataset)} samples) detected - using model WITHOUT BatchNorm"
            )

    # Print basic model info
    sample = train_dataset[0]
    if sample is not None:
        sample_shape = sample["data"].shape
        logger.info(f"Sample spectrogram shape: {sample_shape}")
    logger.info(f"Model created with {sum(p.numel() for p in model.parameters())} parameters")

    # Create loss criterion with optional class weighting
    if isinstance(config.class_weighting, float):
        # Fixed class weighting mode
        weight_neg = 1.0
        weight_pos = (1.0 - config.class_weighting) / config.class_weighting
        class_weights = torch.tensor([weight_neg, weight_pos])

        logger.info(f"Fixed class weighting: target {config.class_weighting:.1%} positive")
        logger.info(f"  Class weights: [neg={weight_neg:.2f}, pos={weight_pos:.2f}]")

        criterion = nn.CrossEntropyLoss(weight=class_weights.to(device))

    elif config.class_weighting == "adaptive":
        # Adaptive class weighting mode (calculates from training set)
        labels = [sample["label"] for sample in train_dataset.samples]
        class_counts = torch.bincount(torch.tensor(labels))
        total_samples = len(train_dataset)
        class_weights = total_samples / (len(class_counts) * class_counts.float())

        logger.info("Adaptive class weighting enabled:")
        logger.info(f"  Class counts: {class_counts.tolist()}")
        logger.info(f"  Class weights: {class_weights.tolist()}")
        logger.info(f"  Weight ratio (neg/pos): {class_weights[0] / class_weights[1]:.2f}")

        criterion = nn.CrossEntropyLoss(weight=class_weights.to(device))

    else:
        # No class weighting
        criterion = nn.CrossEntropyLoss()

    optimizer = optim.Adam(model.parameters(), lr=config.learning_rate, weight_decay=1e-5)

    # Create learning rate scheduler
    scheduler = None
    if config.use_lr_scheduler:
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode='max',
            factor=config.lr_scheduler_factor,
            patience=config.lr_scheduler_patience,
            min_lr=config.lr_scheduler_min_lr
        )
        logger.info("Learning rate scheduler enabled:")
        logger.info(f"  Factor: {config.lr_scheduler_factor}")
        logger.info(f"  Patience: {config.lr_scheduler_patience} epochs")
        logger.info(f"  Min LR: {config.lr_scheduler_min_lr}")

    # Create stopping criterion from config
    stopping_criterion = create_stopping_criterion(config)

    # Reset stopping criterion for this training run
    stopping_criterion.reset()

    logger.info("Starting training...")
    logger.info(f"Stopping criterion: {stopping_criterion.__class__.__name__}")
    logger.info("-" * 50)

    # Pre-allocate timing list to avoid memory allocation during training
    epoch_times = []
    accuracy = 0.0
    best_accuracy = None  # Track accuracy of best saved model
    num_epochs = 0  # Track actual number of epochs trained
    for epoch in range(config.max_epochs):
        epoch_start_time = time.time()
        avg_loss, accuracy = train_epoch(model, train_loader, optimizer, criterion, device)

        epoch_time = time.time() - epoch_start_time
        epoch_times.append(epoch_time)

        # Update learning rate scheduler
        if scheduler is not None:
            scheduler.step(accuracy)

        # Print progress periodically
        if epoch % 10 == 0 or accuracy >= 1.0 or epoch < 5:
            current_lr = optimizer.param_groups[0]['lr']
            logger.info(
                f"Epoch {epoch + 1:4d}/{config.max_epochs} ({epoch_time:.2f}s) - "
                f"Loss: {avg_loss:.4f} - Accuracy: {accuracy:.4f} ({accuracy * 100:.2f}%) - "
                f"LR: {current_lr:.2e}"
            )

        # Check stopping criterion (this updates internal state)
        should_stop = stopping_criterion.should_stop(
            epoch, accuracy, avg_loss, train_dataset=train_dataset
        )

        # Update best model state if criterion indicates we should
        if stopping_criterion.should_update_best_model():
            stopping_criterion.update_best_model(model.state_dict().copy())
            best_accuracy = accuracy  # Track best accuracy separately
            logger.info(f"    💾 Best model updated (epoch {epoch + 1})")

        # Stop if criterion says to stop
        if should_stop:
            num_epochs = epoch + 1
            logger.info("")
            logger.info("=" * 60)
            logger.info(f"🛑 Stopping criterion met ({stopping_criterion.__class__.__name__})")
            logger.info(f"Training completed in {num_epochs} epochs")

            # Report best accuracy if available, otherwise final epoch accuracy
            if best_accuracy is not None:
                logger.info(f"Final accuracy: {best_accuracy:.4f} (best saved model)")
            else:
                logger.info(f"Final accuracy: {accuracy:.4f} (final epoch)")
            logger.info("=" * 60)
            break
    else:
        num_epochs = config.max_epochs
        # Report best accuracy if available, otherwise final epoch accuracy
        if best_accuracy is not None:
            logger.info(
                f"\nTraining completed {num_epochs} epochs. Final accuracy: {best_accuracy:.4f} (best saved model)"
            )
        else:
            logger.info(
                f"\nTraining completed {num_epochs} epochs. Final accuracy: {accuracy:.4f} (final epoch)"
            )

    # Save the best model state if available, otherwise save final model
    # This ensures that when early stopping triggers (e.g., patience exhausted),
    # we save the model from the epoch with the best performance, not the final epoch
    config.create_directories()
    if model_path is None:
        model_path = str(config.get_model_path(version))

    best_model_state = stopping_criterion.get_best_model_state()
    if best_model_state is not None:
        # Save the best model state with complete metadata
        model.load_state_dict(best_model_state)
        model_info = model.get_model_info()
        save_dict = {
            "model_state_dict": best_model_state,
            **{
                k: v for k, v in model_info.items() if k != "num_parameters"
            },  # Exclude runtime-only fields
        }
        torch.save(save_dict, model_path)
        logger.info(f"✅ Best model saved to: {model_path}")
    else:
        # Save the final model state with complete metadata
        model_info = model.get_model_info()
        save_dict = {
            "model_state_dict": model.state_dict(),
            **{
                k: v for k, v in model_info.items() if k != "num_parameters"
            },  # Exclude runtime-only fields
        }
        torch.save(save_dict, model_path)
        logger.info(f"📁 Final model saved to: {model_path}")

    # Clean up and return the best accuracy if available, otherwise final accuracy
    del train_loader
    del model
    torch.cuda.empty_cache() if torch.cuda.is_available() else None

    # Return the accuracy and epoch count of the best saved model, not the final epoch
    return (best_accuracy if best_accuracy is not None else accuracy, num_epochs)
