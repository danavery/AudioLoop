import csv
import os

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from .candidate_selection import CandidateSelector
from .models.cnn_5layer import SoundCNN
from .utils.data_utils import entropy, get_device, variable_length_collate_fn
from .utils.dataset_utils import get_dataset_processor
from .utils.spectrogram_dataset import SpectrogramDataset


def load_training_set_filenames(training_set_csv):
    """Load filenames from training set CSV."""
    if not training_set_csv or not os.path.exists(training_set_csv):
        return set()

    training_filenames = set()
    with open(training_set_csv) as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row.get("filename"):
                training_filenames.add(row["filename"])

    return training_filenames


def load_model(model_path, num_classes, device):
    """Load a trained model from disk."""
    # Load the state dict first to detect architecture
    state_dict = torch.load(model_path, map_location=device)

    # Handle both direct state dict and checkpoint format
    if "model_state_dict" in state_dict:
        actual_state_dict = state_dict["model_state_dict"]
    else:
        actual_state_dict = state_dict

    # Detect if the model was trained with BatchNorm by checking for bn keys
    has_batchnorm = any("bn" in key for key in actual_state_dict)

    # Create model with appropriate BatchNorm setting
    model = SoundCNN(num_classes=num_classes, kernel_size=(3, 3), use_batchnorm=has_batchnorm)

    model.load_state_dict(actual_state_dict)
    model.to(device)
    model.eval()
    return model


def run_binary_inference(
    model_path,
    predictions_csv="outputs/predictions.csv",
    positive_class_name="positive",
    negative_class_name="negative",
    dataset_name="urbansound8k",
    dataset_file=None,
    positive_class_id=8,
    training_set_csv=None,
    **dataset_kwargs,
):
    """
    Run binary classification inference on all dataset files.

    Args:
        model_path: Path to trained binary model
        predictions_csv: Path for output predictions CSV
        positive_class_name: Name for positive class (for logging and output)
        negative_class_name: Name for negative class (for logging and output)
        dataset_name: Name of the dataset ('urbansound8k' or 'fsd50k')
        dataset_file: Path to dataset metadata CSV (auto-detected if None)
        positive_class_id: Audio class ID to treat as positive
        training_set_csv: Path to training set CSV (files to exclude from inference)
        **dataset_kwargs: Additional dataset-specific configuration

    Returns:
        list: Inference results with predictions and confidences
    """
    device = get_device()
    print(f"Using device: {device}")

    # Get dataset processor and config
    processor, config = get_dataset_processor(dataset_name, **dataset_kwargs)

    # Auto-detect dataset file if not provided
    if dataset_file is None:
        dataset_file = str(config.dataset_csv)

    # Load all available metadata and create binary labels inline
    metadata = processor.load_metadata(split="dev")
    print(f"Found {len(metadata)} total samples in dataset")

    # Load training set filenames to exclude
    training_filenames = load_training_set_filenames(training_set_csv)
    if training_filenames:
        print(f"Excluding {len(training_filenames)} files already in training set")

    # Create dataset entries directly (no intermediate file)
    dataset_entries = []
    filtered_count = 0
    for item in metadata:
        # Use processor's filename conversion method
        spec_filename = processor.get_spectrogram_filename(item)

        # Skip if already in training set
        if spec_filename in training_filenames:
            filtered_count += 1
            continue

        # Use processor's binary classification method
        is_positive = processor.get_binary_label(item, positive_class_id, positive_class_name)

        spec_path = f"data/all_specs/{spec_filename}"

        # Get original class info
        original_class = getattr(item, "classID", getattr(item, "class_id", -1))

        dataset_entries.append(
            {
                "filepath": spec_path,
                "label": int(is_positive),
                "run": 1,
                "original_class": original_class,
                "filename": spec_filename,
            }
        )

    # Load dataset directly from entries
    dataset = SpectrogramDataset(data=dataset_entries, specs_dir="data/all_specs")
    if filtered_count > 0:
        print(f"Filtered out {filtered_count} files already in training set")
    print(f"Running inference on {len(dataset)} files")

    # Binary classification
    num_classes = 2
    print(
        f"Binary classification model (2 classes: {negative_class_name}/0, {positive_class_name}/1)"
    )

    # Load trained model
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")

    model = load_model(model_path, num_classes, device)
    print(f"Loaded model from: {model_path}")

    # Create data loader
    data_loader = DataLoader(
        dataset,
        batch_size=32,
        shuffle=False,
        num_workers=0,
        pin_memory=False,
        collate_fn=variable_length_collate_fn,
    )

    # Run inference and collect results
    results = []
    print("Running binary classification inference...")

    with torch.no_grad():
        for batch in tqdm(data_loader):
            features = batch["data"].to(device)
            true_labels = batch["label"]

            # Add channel dimension if needed
            if features.ndim == 3:
                features = features.unsqueeze(1)

            # Forward pass
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

                result = {
                    "filename": audio_filename,
                    "true_is_positive": (true_label == 1),
                    "predicted_is_positive": (predicted_class == 1),
                    "prediction": prediction_name,
                    "confidence": confidences[i].item(),
                    "entropy": entropies[i].item(),
                    "prob_negative": prob_negative,
                    "prob_positive": prob_positive,
                    "correct": (true_label == predicted_class),
                    "original_class": original_class if original_class is not None else -1,
                    "fold": getattr(
                        batch.get("fold", [None] * len(true_labels))[i], "item", lambda: -1
                    )(),
                    "filepath": batch["filepath"][i],
                }
                results.append(result)

    # Ensure outputs directory exists
    os.makedirs("outputs", exist_ok=True)

    # Save results to CSV
    fieldnames = [
        "filename",
        "true_is_positive",
        "predicted_is_positive",
        "prediction",
        "confidence",
        "entropy",
        "prob_negative",
        "prob_positive",
        "correct",
        "original_class",
        "fold",
        "filepath",
    ]

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

    print(f"Predictions saved to: {predictions_csv}")
    return results


def run_active_learning_cycle(
    positive_class_id=8,
    positive_class_name="siren",
    negative_class_name="not_siren",
    model_path=None,
    dataset_name="urbansound8k",
    dataset_file=None,
    run_number=1,
    training_set_csv=None,
    total_candidates=50,
    positive_percentage=0.75,
    min_confidence=0.8,
    **dataset_kwargs,
):
    """
    Run a complete active learning cycle for binary classification.

    Args:
        positive_class_id: Audio class ID to treat as positive
        positive_class_name: Human-readable name for positive class
        negative_class_name: Human-readable name for negative class
        model_path: Path to trained model (default: outputs/model_v{run_number}.pt)
        dataset_name: Name of the dataset ('urbansound8k' or 'fsd50k')
        dataset_file: Path to dataset metadata CSV (auto-detected if None)
        run_number: Version number for output files (e.g., 1 creates v1 files)
        training_set_csv: Path to training set CSV (auto-detected if None)
        total_candidates: Total number of candidates to select
        positive_percentage: Percentage of candidates that should be positive predictions
        **dataset_kwargs: Additional dataset-specific configuration

    Returns:
        tuple: (predictions_file, candidates_file)
    """

    # Use versioned model path if not specified
    if model_path is None:
        model_path = f"outputs/model_v{run_number}.pt"

    # Auto-detect training set if not provided
    if training_set_csv is None:
        training_set_csv = f"training_sets/training_set_v{run_number}.csv"

    print(f"Using model: {model_path}")
    if os.path.exists(training_set_csv):
        print(f"Using training set: {training_set_csv}")
    else:
        print(f"Training set not found: {training_set_csv} (will process all files)")

    # Step 1: Run inference on all files
    print("\nStep 1: Running binary classification inference...")
    predictions_file = f"outputs/predictions_v{run_number}.csv"

    _ = run_binary_inference(
        model_path=model_path,
        predictions_csv=predictions_file,
        positive_class_name=positive_class_name,
        negative_class_name=negative_class_name,
        dataset_name=dataset_name,
        dataset_file=dataset_file,
        positive_class_id=positive_class_id,
        training_set_csv=training_set_csv,
        **dataset_kwargs,
    )

    # Step 2: Select candidates for active learning
    print("\nStep 2: Selecting candidates for human labeling...")
    candidates_file = f"outputs/labeling_candidates_v{run_number}.csv"

    selector = CandidateSelector(
        total_candidates=total_candidates,
        positive_percentage=positive_percentage,
        min_confidence=min_confidence,
        candidate_pool_multiplier=5,
    )
    _ = selector.select_candidates(
        predictions_file=predictions_file,
        candidates_csv=candidates_file,
        positive_class_name=positive_class_name,
        negative_class_name=negative_class_name,
    )

    print("\n🎯 Active Learning Cycle Complete!")
    print("Next steps:")
    print(f"1. Review {candidates_file}")
    print("2. Add human labels to 'needs_human_label' column")
    print("3. Add those labels to your training set")
    print("4. Retrain model and repeat")

    return predictions_file, candidates_file
