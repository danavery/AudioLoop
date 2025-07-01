import csv
import os
import random
import time

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from .datasets.fsd50k import FSD50KConfig, FSD50KProcessor
from .datasets.urbansound8k import UrbanSound8KConfig, UrbanSound8KProcessor
from .models.cnn_5layer import SoundCNN
from .utils.data_utils import entropy, get_device, variable_length_collate_fn
from .utils.spectrogram_dataset import SpectrogramDataset


def calculate_percentiles(values, percentiles=None):
    """Calculate percentiles for a list of values."""
    if percentiles is None:
        percentiles = [10, 25, 50, 75, 90, 95]
    if not values:
        return dict.fromkeys(percentiles, 0)
    sorted_values = sorted(values)
    n = len(sorted_values)
    result = {}
    for p in percentiles:
        index = int((p / 100) * (n - 1))
        result[p] = sorted_values[index]
    return result


def analyze_confidence_distribution(predictions, class_name):
    """Analyze confidence distribution and return statistics."""
    if not predictions:
        return None

    confidences = [p["confidence"] for p in predictions]
    percentiles = calculate_percentiles(confidences)

    high_conf = sum(1 for c in confidences if c >= 0.8)
    very_high_conf = sum(1 for c in confidences if c >= 0.9)
    ultra_high_conf = sum(1 for c in confidences if c >= 0.95)

    return {
        "class_name": class_name,
        "count": len(predictions),
        "confidences": confidences,
        "percentiles": percentiles,
        "mean": sum(confidences) / len(confidences),
        "min": min(confidences),
        "max": max(confidences),
        "high_conf": high_conf,
        "very_high_conf": very_high_conf,
        "ultra_high_conf": ultra_high_conf,
    }


def print_confidence_distribution(predictions, class_name):
    """Print detailed confidence distribution for a class."""
    stats = analyze_confidence_distribution(predictions, class_name)
    if not stats:
        print(f"No {class_name} predictions to analyze")
        return

    print(
        f"\n{stats['class_name'].title()} Prediction Confidence Distribution ({stats['count']} samples):"
    )
    print(f"  Mean: {stats['mean']:.3f}")
    print(f"  Min:  {stats['min']:.3f}")
    print(f"  10th: {stats['percentiles'][10]:.3f}")
    print(f"  25th: {stats['percentiles'][25]:.3f}")
    print(f"  50th: {stats['percentiles'][50]:.3f}")
    print(f"  75th: {stats['percentiles'][75]:.3f}")
    print(f"  90th: {stats['percentiles'][90]:.3f}")
    print(f"  95th: {stats['percentiles'][95]:.3f}")
    print(f"  Max:  {stats['max']:.3f}")

    print(
        f"  High confidence (≥0.8):  {stats['high_conf']}/{stats['count']} ({stats['high_conf'] / stats['count']:.1%})"
    )
    print(
        f"  Very high conf (≥0.9):   {stats['very_high_conf']}/{stats['count']} ({stats['very_high_conf'] / stats['count']:.1%})"
    )
    print(
        f"  Ultra high conf (≥0.95): {stats['ultra_high_conf']}/{stats['count']} ({stats['ultra_high_conf'] / stats['count']:.1%})"
    )


def save_confidence_stats(
    predictions_file,
    positive_preds,
    negative_preds,
    positive_class_name,
    negative_class_name,
    overall_accuracy,
    positive_accuracy,
    negative_accuracy,
    avg_confidence,
    min_conf,
    max_conf,
    high_conf_percentage,
):
    """Save detailed confidence distribution statistics to a file."""
    import re

    # Extract version/run number from predictions filename for stats filename
    version_match = re.search(r"_v(\d+)", predictions_file)
    version_suffix = f"_v{version_match.group(1)}" if version_match else ""

    stats_file = f"outputs/confidence_stats{version_suffix}.txt"

    with open(stats_file, "w") as f:
        f.write(f"Confidence Distribution Statistics{version_suffix}\n")
        f.write("=" * 50 + "\n\n")

        # Overall model performance
        f.write("Model Performance Summary:\n")
        f.write(f"Overall Accuracy: {overall_accuracy:.3f}\n")
        f.write(f"True {positive_class_name} Accuracy: {positive_accuracy:.3f}\n")
        f.write(f"True {negative_class_name} Accuracy: {negative_accuracy:.3f}\n")
        f.write(
            f"Overall Confidence: avg={avg_confidence:.3f}, range={min_conf:.3f}-{max_conf:.3f}\n"
        )
        f.write(f"High confidence samples (≥0.8): {high_conf_percentage:.1%}\n\n")

        # Predicted positive class distribution
        pos_stats = analyze_confidence_distribution(positive_preds, positive_class_name)
        if pos_stats:
            f.write(
                f"Predicted {pos_stats['class_name'].title()} Distribution ({pos_stats['count']} samples):\n"
            )
            f.write(f"  Mean: {pos_stats['mean']:.3f}\n")
            f.write(f"  Min:  {pos_stats['min']:.3f}\n")
            f.write(f"  10th: {pos_stats['percentiles'][10]:.3f}\n")
            f.write(f"  25th: {pos_stats['percentiles'][25]:.3f}\n")
            f.write(f"  50th: {pos_stats['percentiles'][50]:.3f}\n")
            f.write(f"  75th: {pos_stats['percentiles'][75]:.3f}\n")
            f.write(f"  90th: {pos_stats['percentiles'][90]:.3f}\n")
            f.write(f"  95th: {pos_stats['percentiles'][95]:.3f}\n")
            f.write(f"  Max:  {pos_stats['max']:.3f}\n")

            f.write(
                f"  High confidence (≥0.8):  {pos_stats['high_conf']}/{pos_stats['count']} ({pos_stats['high_conf'] / pos_stats['count']:.1%})\n"
            )
            f.write(
                f"  Very high conf (≥0.9):   {pos_stats['very_high_conf']}/{pos_stats['count']} ({pos_stats['very_high_conf'] / pos_stats['count']:.1%})\n"
            )
            f.write(
                f"  Ultra high conf (≥0.95): {pos_stats['ultra_high_conf']}/{pos_stats['count']} ({pos_stats['ultra_high_conf'] / pos_stats['count']:.1%})\n\n"
            )

        # Predicted negative class distribution
        neg_stats = analyze_confidence_distribution(negative_preds, negative_class_name)
        if neg_stats:
            f.write(
                f"Predicted {neg_stats['class_name'].title()} Distribution ({neg_stats['count']} samples):\n"
            )
            f.write(f"  Mean: {neg_stats['mean']:.3f}\n")
            f.write(f"  Min:  {neg_stats['min']:.3f}\n")
            f.write(f"  10th: {neg_stats['percentiles'][10]:.3f}\n")
            f.write(f"  25th: {neg_stats['percentiles'][25]:.3f}\n")
            f.write(f"  50th: {neg_stats['percentiles'][50]:.3f}\n")
            f.write(f"  75th: {neg_stats['percentiles'][75]:.3f}\n")
            f.write(f"  90th: {neg_stats['percentiles'][90]:.3f}\n")
            f.write(f"  95th: {neg_stats['percentiles'][95]:.3f}\n")
            f.write(f"  Max:  {neg_stats['max']:.3f}\n")

            f.write(
                f"  High confidence (≥0.8):  {neg_stats['high_conf']}/{neg_stats['count']} ({neg_stats['high_conf'] / neg_stats['count']:.1%})\n"
            )
            f.write(
                f"  Very high conf (≥0.9):   {neg_stats['very_high_conf']}/{neg_stats['count']} ({neg_stats['very_high_conf'] / neg_stats['count']:.1%})\n"
            )
            f.write(
                f"  Ultra high conf (≥0.95): {neg_stats['ultra_high_conf']}/{neg_stats['count']} ({neg_stats['ultra_high_conf'] / neg_stats['count']:.1%})\n\n"
            )

        # Prediction counts
        f.write("Prediction Counts:\n")
        f.write(f"Predicted {positive_class_name}: {pos_stats['count'] if pos_stats else 0}\n")
        f.write(f"Predicted {negative_class_name}: {neg_stats['count'] if neg_stats else 0}\n")
        f.write(
            f"Total predictions: {(pos_stats['count'] if pos_stats else 0) + (neg_stats['count'] if neg_stats else 0)}\n"
        )

    print(f"Confidence statistics saved to: {stats_file}")


def load_model(model_path, num_classes, device):
    """Load trained model from checkpoint."""
    # Try to load the model state dict to check its structure
    state_dict = torch.load(model_path, map_location=device)

    # Check if the model has BatchNorm layers by looking for 'bn' keys
    has_batchnorm = any("bn" in key for key in state_dict)

    # Create model with appropriate BatchNorm setting
    model = SoundCNN(num_classes=num_classes, kernel_size=(3, 3), use_batchnorm=has_batchnorm)

    if has_batchnorm:
        print("Loading model WITH BatchNorm")
    else:
        print("Loading model WITHOUT BatchNorm")

    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    return model


def get_dataset_processor(dataset_name: str, **kwargs):
    """Get the appropriate dataset processor and config."""
    if dataset_name == "urbansound8k":
        config = UrbanSound8KConfig()
        processor = UrbanSound8KProcessor(config)
        return processor, config
    if dataset_name == "fsd50k":
        config = FSD50KConfig()
        processor = FSD50KProcessor(config)
        return processor, config
    raise ValueError(f"Unsupported dataset: {dataset_name}")


def create_binary_labels(
    dataset_name="urbansound8k",
    dataset_file=None,
    positive_class_id=8,
    output_csv="outputs/binary_labels.csv",
    positive_class_name="positive",
    negative_class_name="negative",
    **dataset_kwargs,
):
    """
    Create binary labels from audio dataset for any specified class.

    Args:
        dataset_name: Name of the dataset ('urbansound8k' or 'fsd50k')
        dataset_file: Path to dataset metadata CSV file (auto-detected if None)
        positive_class_id: Audio class ID to treat as positive (1)
        output_csv: Path for output binary labels CSV
        positive_class_name: Name for positive class (for logging)
        negative_class_name: Name for negative class (for logging)
        **dataset_kwargs: Additional dataset-specific configuration

    Returns:
        str: Path to created binary labels CSV
    """
    # Get dataset processor and config
    processor, config = get_dataset_processor(dataset_name, **dataset_kwargs)

    # Auto-detect dataset file if not provided
    if dataset_file is None:
        dataset_file = str(config.dataset_csv)

    binary_data = []
    positive_count = 0
    negative_count = 0

    # Load metadata using the processor
    if dataset_name == "urbansound8k":
        metadata = processor.load_metadata()
        for item in metadata:
            filename = item["filename"]
            class_id = item["class_id"]

            # Binary classification: specified class vs everything else
            is_positive = 1 if class_id == positive_class_id else 0

            if is_positive:
                positive_count += 1
            else:
                negative_count += 1

            # Convert audio filename to spectrogram filename
            spec_filename = filename.replace('.wav', '.pt')
            spec_path = f"data/all_specs/{spec_filename}"

            binary_data.append({
                "filepath": spec_path,
                "label": is_positive,
                "run": 1
            })

    elif dataset_name == "fsd50k":
        metadata = processor.load_metadata(split="dev")
        for item in metadata:
            filename = item["filename"]
            labels = item["labels"]

            # For FSD50K, use the positive_class_name directly
            is_positive = 1 if positive_class_name in labels else 0

            if is_positive:
                positive_count += 1
            else:
                negative_count += 1

            # Convert audio filename to spectrogram filename
            spec_filename = f"{filename}.pt"
            spec_path = f"data/all_specs/{spec_filename}"

            binary_data.append({
                "filepath": spec_path,
                "label": is_positive,
                "run": 1
            })
    else:
        raise ValueError(f"Unsupported dataset: {dataset_name}")

    # Ensure outputs directory exists
    os.makedirs("outputs", exist_ok=True)

    # Write binary labels CSV in training set format
    with open(output_csv, "w", newline="") as f:
        fieldnames = ["filepath", "label", "run"]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(binary_data)

    print(f"Created binary labels: {output_csv}")
    print(f"{positive_class_name} samples: {positive_count}")
    print(f"{negative_class_name} samples: {negative_count}")
    print(f"Total samples: {len(binary_data)}")

    return output_csv


def run_binary_inference(
    model_path,
    labels_file="outputs/binary_labels.csv",
    predictions_csv="outputs/predictions.csv",
    positive_class_name="positive",
    negative_class_name="negative",
):
    """
    Run binary classification inference on all dataset files.

    Args:
        model_path: Path to trained binary model
        labels_file: CSV file with binary labels (filename,label,original_class)
        predictions_csv: Path for output predictions CSV
        positive_class_name: Name for positive class (for logging and output)
        negative_class_name: Name for negative class (for logging and output)

    Returns:
        list: Inference results with predictions and confidences
    """
    device = get_device()
    print(f"Using device: {device}")

    # Load dataset
    dataset = SpectrogramDataset(csv_file=labels_file, specs_dir="data/all_specs")
    print(f"Dataset size: {len(dataset)}")

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
                    "true_label": true_label,
                    "predicted_label": predicted_class,
                    "prediction": prediction_name,
                    "confidence": confidences[i].item(),
                    "entropy": entropies[i].item(),
                    "prob_negative": prob_negative,
                    "prob_positive": prob_positive,
                    "correct": (true_label == predicted_class),
                    "original_class": original_class if original_class is not None else -1,
                    "filepath": batch["filepath"][i],
                }
                results.append(result)

    # Calculate summary statistics
    total_samples = len(results)
    correct_predictions = sum(1 for r in results if r["correct"])
    accuracy = correct_predictions / total_samples

    positive_predictions = sum(1 for r in results if r["prediction"] == positive_class_name)
    negative_predictions = total_samples - positive_predictions

    true_positives = sum(1 for r in results if r["true_label"] == 1)
    true_negatives = total_samples - true_positives

    print("\nBinary Classification Results:")
    print(f"Total samples: {total_samples}")
    print(f"Accuracy: {accuracy:.4f} ({correct_predictions}/{total_samples})")
    print(f"True {positive_class_name}s in dataset: {true_positives}")
    print(f"True {negative_class_name}s in dataset: {true_negatives}")
    print(f"Predicted {positive_class_name}s: {positive_predictions}")
    print(f"Predicted {negative_class_name}s: {negative_predictions}")

    # Calculate mean confidence and entropy
    mean_confidence = sum(r["confidence"] for r in results) / total_samples
    mean_entropy = sum(r["entropy"] for r in results) / total_samples
    print(f"Mean confidence: {mean_confidence:.4f}")
    print(f"Mean entropy: {mean_entropy:.4f}")

    # Ensure outputs directory exists
    os.makedirs("outputs", exist_ok=True)

    # Save results to CSV
    fieldnames = [
        "filename",
        "true_label",
        "predicted_label",
        "prediction",
        "confidence",
        "entropy",
        "prob_negative",
        "prob_positive",
        "correct",
        "original_class",
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

    print(f"\nResults saved to: {predictions_csv}")
    return results


def select_candidates_for_labeling(
    predictions_file="outputs/predictions.csv",
    total_candidates=20,
    positive_percentage=0.80,
    min_confidence=0.8,
    candidates_csv="outputs/labeling_candidates.csv",
    positive_class_name="positive",
    negative_class_name="negative",
    candidate_pool_multiplier=5,
):
    """
    Select predictions for human labeling in active learning.
    Uses percentage-based selection to focus on positive predictions for imbalanced datasets.

    Args:
        predictions_file: CSV file with model predictions
        total_candidates: Total number of candidates to select
        positive_percentage: Percentage of candidates that should be positive predictions (0.0-1.0)
        min_confidence: Initial minimum confidence threshold
        candidates_csv: Output file for candidates
        positive_class_name: Name for positive class
        negative_class_name: Name for negative class
        candidate_pool_multiplier: Multiplier for candidate pool size (e.g., 5 means sample from top 50 if selecting 10)

    Returns:
        list: Selected candidates for human labeling
    """

    # Read predictions
    predictions = []
    with open(predictions_file) as f:
        reader = csv.DictReader(f)
        for row in reader:
            row["confidence"] = float(row["confidence"])
            row["entropy"] = float(row["entropy"])
            predictions.append(row)

    # Calculate numbers based on percentage
    num_positive = int(total_candidates * positive_percentage)
    num_negative = total_candidates - num_positive

    # Separate positive and negative predictions
    positive_preds = [p for p in predictions if p["prediction"] == positive_class_name]
    negative_preds = [p for p in predictions if p["prediction"] == negative_class_name]

    # Sort by confidence (highest first)
    positive_preds.sort(key=lambda x: x["confidence"], reverse=True)
    negative_preds.sort(key=lambda x: x["confidence"], reverse=True)

    # Create broader candidate pools for sampling
    positive_pool_size = min(len(positive_preds), num_positive * candidate_pool_multiplier)
    negative_pool_size = min(len(negative_preds), num_negative * candidate_pool_multiplier)

    positive_pool = positive_preds[:positive_pool_size]
    negative_pool = negative_preds[:negative_pool_size]

    # Randomly sample from the pools to improve diversity
    random.seed(int(time.time()))

    positive_candidates = random.sample(positive_pool, min(num_positive, len(positive_pool)))
    negative_candidates = random.sample(negative_pool, min(num_negative, len(negative_pool)))

    # Combine and shuffle final candidates
    all_candidates = positive_candidates + negative_candidates
    random.shuffle(all_candidates)

    # Calculate overall accuracy statistics (use all predictions, not just selected candidates)
    all_predictions = positive_preds + negative_preds
    total_samples = len(all_predictions)

    # Handle both boolean and string representations of 'correct' field
    correct_predictions = 0
    for p in all_predictions:
        if isinstance(p["correct"], str):
            if p["correct"].lower() == "true":
                correct_predictions += 1
        elif p["correct"]:
            correct_predictions += 1

    overall_accuracy = correct_predictions / total_samples if total_samples > 0 else 0

    # Calculate accuracy by true class (not by predicted class)
    true_positive_samples = []
    true_negative_samples = []

    for p in all_predictions:
        true_label = int(p["true_label"]) if isinstance(p["true_label"], str) else p["true_label"]

        if true_label == 1:
            true_positive_samples.append(p)
        else:
            true_negative_samples.append(p)

    positive_correct = 0
    for p in true_positive_samples:
        if isinstance(p["correct"], str):
            if p["correct"].lower() == "true":
                positive_correct += 1
        elif p["correct"]:
            positive_correct += 1

    negative_correct = 0
    for p in true_negative_samples:
        if isinstance(p["correct"], str):
            if p["correct"].lower() == "true":
                negative_correct += 1
        elif p["correct"]:
            negative_correct += 1

    positive_accuracy = (
        positive_correct / len(true_positive_samples) if len(true_positive_samples) > 0 else 0
    )
    negative_accuracy = (
        negative_correct / len(true_negative_samples) if len(true_negative_samples) > 0 else 0
    )

    # Overall statistics
    all_confidences = [p["confidence"] for p in all_predictions]
    high_conf_count = sum(1 for conf in all_confidences if conf >= min_confidence)
    high_conf_percentage = high_conf_count / total_samples if total_samples > 0 else 0

    avg_confidence = sum(all_confidences) / len(all_confidences) if all_confidences else 0
    min_conf = min(all_confidences) if all_confidences else 0
    max_conf = max(all_confidences) if all_confidences else 0

    print("\nModel Performance Summary:")
    print(f"Overall Accuracy: {overall_accuracy:.3f}")
    print(f"True {positive_class_name} Accuracy: {positive_accuracy:.3f}")
    print(f"True {negative_class_name} Accuracy: {negative_accuracy:.3f}")
    print(f"Overall Confidence: avg={avg_confidence:.3f}, range={min_conf:.3f}-{max_conf:.3f}")
    print(
        f"High confidence samples (≥{min_confidence}): {high_conf_count}/{total_samples} ({high_conf_percentage:.1%})"
    )

    # Show confidence distribution for each class
    print_confidence_distribution(positive_preds, positive_class_name)
    print_confidence_distribution(negative_preds, negative_class_name)

    # Save detailed statistics to file
    save_confidence_stats(
        predictions_file,
        positive_preds,
        negative_preds,
        positive_class_name,
        negative_class_name,
        overall_accuracy,
        positive_accuracy,
        negative_accuracy,
        avg_confidence,
        min_conf,
        max_conf,
        high_conf_percentage,
    )

    # Recommendation for selection strategy
    if overall_accuracy > 0.95 and high_conf_percentage > 0.8:
        print("💡 Recommendation: Model is highly accurate and confident")
        print("   Consider implementing entropy-based selection for uncertainty sampling")
        print("   Current confidence-based selection may not find challenging cases")

    print("\nActive Learning Candidate Selection:")
    print(f"Available {positive_class_name} predictions: {len(positive_preds)}")
    print(f"Available {negative_class_name} predictions: {len(negative_preds)}")
    print(
        f"Candidate pool sizes: {positive_pool_size} {positive_class_name}, {negative_pool_size} {negative_class_name}"
    )
    print(f"Selected {len(positive_candidates)} {positive_class_name} candidates")
    print(f"Selected {len(negative_candidates)} {negative_class_name} candidates")
    print(f"Total candidates for labeling: {len(all_candidates)}")

    if len(all_candidates) > 0:
        # Add labeling helper columns
        for candidate in all_candidates:
            candidate["needs_human_label"] = ""  # Empty column for human to fill
            candidate["human_confidence"] = ""  # Human confidence in the label

        # Ensure outputs directory exists
        os.makedirs("outputs", exist_ok=True)

        # Save candidates
        fieldnames = list(all_candidates[0].keys())
        with open(candidates_csv, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(all_candidates)

        print(f"Candidates saved to: {candidates_csv}")
        if positive_candidates:
            conf_values = [p["confidence"] for p in positive_candidates]
            print(
                f"{positive_class_name} confidence range: {min(conf_values):.3f} - {max(conf_values):.3f}"
            )
            print(
                f"  First 3 {positive_class_name} samples: {[p['filename'] for p in positive_candidates[:3]]}"
            )
        if negative_candidates:
            conf_values = [p["confidence"] for p in negative_candidates]
            print(
                f"{negative_class_name} confidence range: {min(conf_values):.3f} - {max(conf_values):.3f}"
            )
            print(
                f"  First 3 {negative_class_name} samples: {[p['filename'] for p in negative_candidates[:3]]}"
            )
    else:
        print("No candidates found.")

    return all_candidates


def run_active_learning_cycle(
    positive_class_id=8,
    positive_class_name="siren",
    negative_class_name="not_siren",
    model_path=None,
    dataset_name="urbansound8k",
    dataset_file=None,
    run_number=1,
    total_candidates=20,
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
        total_candidates: Total number of candidates to select
        positive_percentage: Percentage of candidates that should be positive predictions
        **dataset_kwargs: Additional dataset-specific configuration

    Returns:
        tuple: (predictions_file, candidates_file)
    """

    # Use versioned model path if not specified
    if model_path is None:
        model_path = f"outputs/model_v{run_number}.pt"

    print(f"Using model: {model_path}")

    # Step 1: Create binary labels for the specified class
    print(f"Step 1: Creating binary labels for {positive_class_name} vs {negative_class_name}...")
    binary_labels_file = create_binary_labels(
        dataset_name=dataset_name,
        dataset_file=dataset_file,
        positive_class_id=positive_class_id,
        output_csv=f"outputs/binary_labels_v{run_number}.csv",
        positive_class_name=positive_class_name,
        negative_class_name=negative_class_name,
        **dataset_kwargs,
    )

    # Step 2: Run inference on all files
    print("\nStep 2: Running binary classification inference...")
    predictions_file = f"outputs/predictions_v{run_number}.csv"

    _ = run_binary_inference(
        model_path=model_path,
        labels_file=binary_labels_file,
        predictions_csv=predictions_file,
        positive_class_name=positive_class_name,
        negative_class_name=negative_class_name,
    )

    # Step 3: Select candidates for active learning
    print("\nStep 3: Selecting candidates for human labeling...")
    candidates_file = f"outputs/labeling_candidates_v{run_number}.csv"

    _ = select_candidates_for_labeling(
        predictions_file=predictions_file,
        total_candidates=total_candidates,
        positive_percentage=positive_percentage,
        min_confidence=min_confidence,
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
