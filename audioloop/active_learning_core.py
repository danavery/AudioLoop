import csv
import logging
import os

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from .models.model_registry import get_model_class, list_available_models
from .utils.candidate_selection import (
    load_predictions,
    print_selection_statistics,
    save_candidates,
)
from .utils.data_utils import entropy, get_device, variable_length_collate_fn
from .utils.spectrogram_dataset import SpectrogramDataset

# Set up logger for this module
logger = logging.getLogger(__name__)


def load_training_set_filenames(training_set_csv):
    """Load filenames from training set CSV."""
    if not training_set_csv or not os.path.exists(training_set_csv):
        return set()

    training_filenames = set()
    with open(training_set_csv) as f:
        reader = csv.DictReader(f)
        for row in reader:
            # Handle both 'filename' and 'filepath' columns (extract filename from filepath)
            filename = row.get("filename")
            if not filename and row.get("filepath"):
                # Extract filename from filepath (e.g., "foo.pt" from "data/specs/foo.pt")
                filename = os.path.basename(row["filepath"])
            if filename:
                training_filenames.add(filename)

    return training_filenames


def load_model(model_path, num_classes, device):
    """Load a trained model from disk using all saved constructor args."""
    # Load checkpoint with all metadata
    checkpoint = torch.load(model_path, map_location=device)
    model_type = checkpoint.get("model_type", "cnn5layer")  # Default to cnn5layer

    # Use dynamic model discovery
    try:
        model_class = get_model_class(model_type)
    except ValueError:
        available = list_available_models()
        raise ValueError(
            f"Unknown model type: {model_type}. Available: {', '.join(sorted(available))}"
        ) from None

    # Extract constructor args from checkpoint (everything except model_state_dict)
    constructor_args = {k: v for k, v in checkpoint.items() if k != "model_state_dict"}

    # Create model with all saved constructor arguments
    model = model_class(**constructor_args)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()
    return model


def run_binary_inference(
    config,
    model_path,
    predictions_csv=None,
    positive_class_name="positive",
    negative_class_name="negative",
    dataset_file=None,
    positive_class_id=8,
    training_set_csv=None,
):
    """
    Run binary classification inference on all dataset files.

    Args:
        config: AudioLoopConfig with dataset and experiment configuration
        model_path: Path to trained binary model
        predictions_csv: Path for output predictions CSV
        positive_class_name: Name for positive class (for logging and output)
        negative_class_name: Name for negative class (for logging and output)
        dataset_file: Path to dataset metadata CSV (auto-detected if None)
        positive_class_id: Audio class ID to treat as positive
        training_set_csv: Path to training set CSV (files to exclude from inference)

    Returns:
        list: Inference results with predictions and confidences
    """
    device = get_device()
    logger.info(f"Using device: {device}")

    # Set default predictions_csv if not provided
    if predictions_csv is None:
        predictions_csv = str(config.output_dir / "predictions.csv")

    # Get dataset config from unified config
    dataset_config = config.get_dataset_config()

    # Auto-detect dataset file if not provided
    if dataset_file is None:
        dataset_file = str(dataset_config.dataset_csv)

    # Load all available metadata and create binary labels inline
    metadata = dataset_config.load_metadata()
    logger.info(f"Found {len(metadata)} total samples in dataset")

    # Load training set filenames to exclude
    training_filenames = load_training_set_filenames(training_set_csv)
    if training_filenames:
        logger.info(f"Excluding {len(training_filenames)} files already in training set")

    # Create dataset entries directly (no intermediate file)
    dataset_entries = []
    filtered_count = 0
    for item in metadata:
        # Use dataset config's path conversion method
        spec_path = dataset_config.get_spectrogram_path(item["filename"], config.specs_dir)
        spec_filename = spec_path.name

        # Skip if already in training set
        if spec_filename in training_filenames:
            filtered_count += 1
            continue

        # Get audio_path for lazy generation
        audio_path = item.get("audio_path") or dataset_config.get_audio_path(item["filename"])

        # Skip if both spec and audio are missing (can't process this file)
        if not spec_path.exists() and not audio_path.exists():
            continue

        # Use dataset config's binary classification method
        is_positive = dataset_config.get_binary_label(item, positive_class_id, positive_class_name)

        spec_path = str(spec_path)

        # Get original class info
        original_class = getattr(item, "classID", getattr(item, "class_id", -1))

        dataset_entries.append(
            {
                "filepath": spec_path,
                "label": int(is_positive),
                "original_class": original_class,
                "filename": spec_filename,
                "audio_path": str(audio_path),
            }
        )

    # Load dataset directly from entries with lazy generation support
    dataset = SpectrogramDataset(
        data=dataset_entries, specs_dir=str(config.specs_dir), dataset_config=dataset_config
    )
    if filtered_count > 0:
        logger.info(f"Filtered out {filtered_count} files already in training set")
    logger.info(f"Running inference on {len(dataset)} files")

    # Binary classification
    num_classes = 2
    logger.info(
        f"Binary classification model (2 classes: {negative_class_name}/0, {positive_class_name}/1)"
    )

    # Load trained model
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")

    model = load_model(model_path, num_classes, device)
    logger.info(f"Loaded model from: {model_path}")

    # Create data loader
    data_loader = DataLoader(
        dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        pin_memory=torch.cuda.is_available(),  # Enable GPU memory pinning
        persistent_workers=config.num_workers > 0,  # Only when using workers
        collate_fn=variable_length_collate_fn,
    )

    # Run inference and collect results
    results = []
    logger.info("Running binary classification inference...")

    with torch.no_grad():
        for batch in tqdm(data_loader):
            true_labels = batch["label"]

            # Standard PyTorch forward pass
            features = batch["data"].to(device)
            logits = model(features)
            probabilities = F.softmax(logits, dim=-1)
            predicted_classes = torch.argmax(probabilities, dim=-1)

            # Calculate confidence (max probability) and entropy
            confidences = torch.max(probabilities, dim=-1)[0]
            entropies = entropy(probabilities)

            # Move tensors to CPU for processing
            probabilities = probabilities.cpu()
            predicted_classes = predicted_classes.cpu()
            confidences = confidences.cpu()
            entropies = entropies.cpu()

            # Process each sample in the batch
            for i in range(len(true_labels)):
                audio_filename = batch["filename"][i]
                true_label = true_labels[i].item()
                original_class = batch.get("original_class", [None] * len(true_labels))[i]
                if original_class is not None:
                    original_class = (
                        original_class if isinstance(original_class, int) else original_class.item()
                    )

                # Extract probability for each class
                prob_negative = probabilities[i][0].item()  # Class 0 probability
                prob_positive = probabilities[i][1].item()  # Class 1 probability

                predicted_class = predicted_classes[i].item()
                prediction_name = (
                    positive_class_name if predicted_class == 1 else negative_class_name
                )

                # Build result dictionary - always include core prediction fields
                result = {
                    "filename": audio_filename,
                    "prediction": (predicted_class == 1),
                    "predicted_class": prediction_name,
                    "target_class": positive_class_name,
                    "confidence": confidences[i].item(),
                    "entropy": entropies[i].item(),
                    "prob_negative": prob_negative,
                    "prob_positive": prob_positive,
                    "original_class": original_class if original_class is not None else -1,
                    "fold": getattr(
                        batch.get("fold", [None] * len(true_labels))[i], "item", lambda: -1
                    )(),
                    "filepath": batch["filepath"][i],
                }

                # Add ground truth fields only if requested
                if config.with_ground_truth:
                    result["ground_truth"] = true_label == 1
                    result["correct"] = true_label == predicted_class
                results.append(result)

    # Ensure outputs directory exists
    config.create_directories()

    # Save results to CSV - conditionally include ground truth columns
    fieldnames = [
        "filename",
        "prediction",
        "predicted_class",
        "target_class",
        "confidence",
        "entropy",
        "prob_negative",
        "prob_positive",
        "original_class",
        "fold",
        "filepath",
    ]

    # Add ground truth columns if requested
    if config.with_ground_truth:
        fieldnames.insert(1, "ground_truth")  # Insert after filename
        fieldnames.insert(-2, "correct")  # Insert before filepath
    with open(predictions_csv, "w", newline="") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        for result in results:
            # Format float values for CSV output
            formatted_result = result.copy()
            formatted_result["confidence"] = f"{result['confidence']:.3f}"
            formatted_result["entropy"] = f"{result['entropy']:.3f}"
            formatted_result["prob_negative"] = f"{result['prob_negative']:.3f}"
            formatted_result["prob_positive"] = f"{result['prob_positive']:.3f}"
            writer.writerow(formatted_result)

    logger.info(f"Predictions saved to: {predictions_csv}")
    return results


def run_active_learning_cycle(
    config,
    positive_class_id,
    positive_class_name,
    negative_class_name,
    run_number,
    model_path=None,
    dataset_file=None,
    training_set_csv=None,
    seed=None,
    log_level=logging.INFO,
):
    """
    Run a complete active learning cycle for binary classification using config-driven parameters.

    Args:
        config: AudioLoopConfig with all active learning settings (total_candidates, selection_mode, etc.)
        positive_class_id: Audio class ID to treat as positive
        positive_class_name: Human-readable name for positive class
        negative_class_name: Human-readable name for negative class
        run_number: Version number for output files (e.g., 1 creates v1 files)
        model_path: Path to trained model (uses config.get_model_path(run_number) if None)
        dataset_file: Path to dataset metadata CSV (auto-detected from config if None)
        training_set_csv: Path to training set CSV (uses config.get_training_set_path(run_number) if None)
        seed: Random seed for reproducibility (overrides config.seed if provided)
        log_level: Logging level (logging.INFO for normal output, logging.WARNING for quiet)

    Returns:
        tuple: (predictions_file, candidates_file)
    """
    # Set up logging for this active learning run
    logger.setLevel(log_level)

    # Ensure output directories exist
    config.create_directories()

    # Use versioned model path if not specified
    if model_path is None:
        model_path = str(config.get_model_path(run_number))

    # Auto-detect training set if not provided
    if training_set_csv is None:
        training_set_csv = str(config.get_training_set_path(run_number))

    logger.info(f"Using model: {model_path}")
    if os.path.exists(training_set_csv):
        logger.info(f"Using training set: {training_set_csv}")
    else:
        logger.info(f"Training set not found: {training_set_csv} (will process all files)")

    # Step 1: Run inference on all files
    logger.info("\nStep 1: Running binary classification inference...")
    predictions_file = str(config.get_predictions_path(run_number))

    _ = run_binary_inference(
        config=config,
        model_path=model_path,
        predictions_csv=predictions_file,
        positive_class_name=positive_class_name,
        negative_class_name=negative_class_name,
        dataset_file=dataset_file,
        positive_class_id=positive_class_id,
        training_set_csv=training_set_csv,
    )

    # Step 2: Select candidates for active learning
    logger.info("\nStep 2: Selecting candidates for human labeling...")
    candidates_file = str(config.get_candidates_path(run_number))

    # Create strategy with parameters from config
    strategy_kwargs = {}
    if config.selection_mode == "basic_transition":
        strategy_kwargs = {
            "f1_threshold": None
            if config.auto_thresholds
            else config.basic_transition_f1_threshold,
            "confidence_threshold": None
            if config.auto_thresholds
            else config.basic_transition_confidence_threshold,
            "variance_threshold": None
            if config.auto_thresholds
            else config.basic_transition_variance_threshold,
            "auto_thresholds": config.auto_thresholds,
            "estimated_positive_pct": config.estimated_positive_pct
            or 0.05,  # Default 5% if not provided
        }

    from .utils.candidate_selection import create_strategy

    strategy = create_strategy(config.selection_mode, **strategy_kwargs)

    logger.info(f"Using strategy: {strategy.get_active_strategy_name()}")
    predictions = load_predictions(predictions_file)
    candidates = strategy.select_candidates(
        predictions=predictions,
        num_candidates=config.total_candidates,
        positive_class_name=positive_class_name,
        negative_class_name=negative_class_name,
        positive_percentage=config.positive_percentage,
        candidate_pool_multiplier=5,
        random_seed=seed,
    )
    save_candidates(candidates, candidates_file)
    print_selection_statistics(
        all_predictions=predictions,
        selected_candidates=candidates,
        strategy_name=strategy.get_active_strategy_name(),
        positive_class_name=positive_class_name,
        negative_class_name=negative_class_name,
        min_confidence=config.min_confidence,
    )

    logger.info("\n🎯 Active Learning Cycle Complete!")
    logger.info("Next steps:")
    logger.info(f"1. Review {candidates_file}")
    logger.info("2. Add human labels to 'needs_human_label' column")
    logger.info("3. Add those labels to your training set")
    logger.info("4. Retrain model and repeat")

    return predictions_file, candidates_file
