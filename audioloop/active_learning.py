import csv
import os

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from .models.simple_cnn import SimpleCNN
from .utils.urbansound_dataset import UrbanSoundDataset


def entropy(probabilities):
    """Calculate entropy of probability distribution."""
    # Avoid log(0) by adding small epsilon
    eps = 1e-10
    probs = probabilities + eps
    return -torch.sum(probs * torch.log(probs), dim=-1)


def load_model(model_path, num_classes, device):
    """Load trained model from checkpoint."""
    model = SimpleCNN(num_classes=num_classes)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    return model


def collate_fn(batch):
    """Custom collate function to handle variable-length spectrograms."""
    # Extract data and other fields
    data_list = [item["data"] for item in batch]
    labels = torch.tensor([item["label"] for item in batch])

    # Handle different channel dimensions and time lengths
    max_time_frames = max(spec.shape[-1] for spec in data_list)

    padded_data = []
    for spec in data_list:
        # Convert to mono by averaging channels if stereo
        if spec.shape[0] > 1:
            spec = spec.mean(dim=0, keepdim=True)  # Average channels, keep dimension

        # Pad the time dimension (last dimension) to max_time_frames
        pad_size = max_time_frames - spec.shape[-1]
        if pad_size > 0:
            # Pad with zeros on the right side of the time dimension
            pad_tuple = (0, pad_size)  # (left_pad, right_pad) for last dimension
            padded_spec = torch.nn.functional.pad(spec, pad_tuple, mode='constant', value=0)
        else:
            padded_spec = spec
        padded_data.append(padded_spec)

    # Stack the padded spectrograms
    data_tensor = torch.stack(padded_data)

    # Return in the same format as the original batch
    return {
        "data": data_tensor,
        "label": labels,
        "filename": [item["filename"] for item in batch],
        "filepath": [item["filepath"] for item in batch],
        "original_class": [item.get("original_class", -1) for item in batch]
    }


def create_binary_labels(urbansound_csv="data/urbansound8k/UrbanSound8K.csv",
                        positive_class_id=8,
                        output_csv="outputs/binary_labels.csv",
                        positive_class_name="positive",
                        negative_class_name="negative"):
    """
    Create binary labels from UrbanSound8K dataset for any specified class.

    Args:
        urbansound_csv: Path to UrbanSound8K.csv metadata file
        positive_class_id: UrbanSound8K class ID to treat as positive (1)
        output_csv: Path for output binary labels CSV
        positive_class_name: Name for positive class (for logging)
        negative_class_name: Name for negative class (for logging)

    Returns:
        str: Path to created binary labels CSV
    """
    binary_data = []
    positive_count = 0
    negative_count = 0

    with open(urbansound_csv, 'r') as f:
        csv_reader = csv.DictReader(f)
        for row in csv_reader:
            filename = row['slice_file_name']
            class_id = int(row['classID'])

            # Binary classification: specified class vs everything else
            is_positive = 1 if class_id == positive_class_id else 0

            if is_positive:
                positive_count += 1
            else:
                negative_count += 1

            binary_data.append({
                'filename': filename,
                'is_positive': is_positive,
                'original_class': class_id
            })

    # Ensure outputs directory exists
    os.makedirs("outputs", exist_ok=True)

    # Write binary labels CSV
    with open(output_csv, 'w', newline='') as f:
        fieldnames = ['filename', 'is_positive', 'original_class']
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(binary_data)

    print(f"Created binary labels: {output_csv}")
    print(f"{positive_class_name} samples: {positive_count}")
    print(f"{negative_class_name} samples: {negative_count}")
    print(f"Total samples: {len(binary_data)}")

    return output_csv


def run_binary_inference(model_path,
                        labels_file="outputs/binary_labels.csv",
                        output_csv="outputs/predictions.csv",
                        positive_class_name="positive",
                        negative_class_name="negative"):
    """
    Run binary classification inference on all dataset files.

    Args:
        model_path: Path to trained binary model
        labels_file: CSV file with binary labels (filename,is_positive,original_class)
        output_csv: Path for output predictions CSV
        positive_class_name: Name for positive class (for logging and output)
        negative_class_name: Name for negative class (for logging and output)

    Returns:
        list: Inference results with predictions and confidences
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load dataset
    dataset = UrbanSoundDataset(csv_file=labels_file)
    print(f"Dataset size: {len(dataset)}")

    # Binary classification
    num_classes = 2
    print(f"Binary classification model (2 classes: {negative_class_name}/0, {positive_class_name}/1)")

    # Load trained model
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")

    model = load_model(model_path, num_classes, device)
    print(f"Loaded model from: {model_path}")

    # Create data loader
    data_loader = DataLoader(
        dataset,
        batch_size=32,
        shuffle=False,  # Keep original order for analysis
        num_workers=0,
        pin_memory=False,
        collate_fn=collate_fn
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
                true_is_positive = true_labels[i].item()
                original_class = batch.get("original_class", [None] * len(true_labels))[i]
                if original_class is not None:
                    original_class = original_class if isinstance(original_class, int) else original_class.item()

                # Extract probability for each class
                prob_negative = probabilities[i][0].item()  # Class 0 probability
                prob_positive = probabilities[i][1].item()  # Class 1 probability

                predicted_class = predicted_classes[i].item()
                prediction_name = positive_class_name if predicted_class == 1 else negative_class_name

                result = {
                    "filename": audio_filename,
                    "true_is_positive": true_is_positive,
                    "predicted_is_positive": predicted_class,
                    "prediction": prediction_name,
                    "confidence": confidences[i].item(),
                    "entropy": entropies[i].item(),
                    "prob_negative": prob_negative,
                    "prob_positive": prob_positive,
                    "correct": (true_is_positive == predicted_class),
                    "original_class": original_class if original_class is not None else -1,
                    "filepath": batch["filepath"][i]
                }
                results.append(result)

    # Calculate summary statistics
    total_samples = len(results)
    correct_predictions = sum(1 for r in results if r["correct"])
    accuracy = correct_predictions / total_samples

    positive_predictions = sum(1 for r in results if r["prediction"] == positive_class_name)
    negative_predictions = total_samples - positive_predictions

    true_positives = sum(1 for r in results if r["true_is_positive"] == 1)
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
        "filename", "true_is_positive", "predicted_is_positive", "prediction",
        "confidence", "entropy", "prob_negative", "prob_positive",
        "correct", "original_class", "filepath"
    ]

    with open(output_csv, 'w', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)

    print(f"\nResults saved to: {output_csv}")
    return results


def select_candidates_for_labeling(predictions_csv="outputs/predictions.csv",
                                 num_positive=10, num_negative=10,
                                 min_confidence=0.8,
                                 output_csv="outputs/labeling_candidates.csv",
                                 positive_class_name="positive",
                                 negative_class_name="negative"):
    """
    Select predictions for human labeling in active learning.
    Uses adaptive thresholds to ensure both positive and negative examples.

    Args:
        predictions_csv: CSV file with model predictions
        num_positive: Number of positive predictions to select
        num_negative: Number of negative predictions to select
        min_confidence: Initial minimum confidence threshold
        output_csv: Output file for candidates
        positive_class_name: Name for positive class
        negative_class_name: Name for negative class

    Returns:
        list: Selected candidates for human labeling
    """

    # Read predictions
    predictions = []
    with open(predictions_csv, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            row['confidence'] = float(row['confidence'])
            row['entropy'] = float(row['entropy'])
            predictions.append(row)

    # Separate positive and negative predictions
    positive_preds = [p for p in predictions if p['prediction'] == positive_class_name]
    negative_preds = [p for p in predictions if p['prediction'] == negative_class_name]

    # Sort by confidence (highest first)
    positive_preds.sort(key=lambda x: x['confidence'], reverse=True)
    negative_preds.sort(key=lambda x: x['confidence'], reverse=True)

    # Select candidates with adaptive thresholds
    positive_candidates = []
    negative_candidates = []

    # For positives: try high confidence first, then fall back to top N
    high_conf_positive = [p for p in positive_preds if p['confidence'] >= min_confidence]
    if len(high_conf_positive) >= num_positive:
        positive_candidates = high_conf_positive[:num_positive]
    else:
        # Take all high-confidence + top remaining ones
        positive_candidates = high_conf_positive + positive_preds[len(high_conf_positive):num_positive]

    # For negatives: try high confidence first, then fall back to top N
    high_conf_negative = [p for p in negative_preds if p['confidence'] >= min_confidence]
    if len(high_conf_negative) >= num_negative:
        negative_candidates = high_conf_negative[:num_negative]
    else:
        # Take all high-confidence + top remaining ones
        negative_candidates = high_conf_negative + negative_preds[len(high_conf_negative):num_negative]

    all_candidates = positive_candidates + negative_candidates

    print("\nActive Learning Candidate Selection:")
    print(f"Available {positive_class_name} predictions: {len(positive_preds)}")
    print(f"Available {negative_class_name} predictions: {len(negative_preds)}")
    print(f"High-confidence {positive_class_name} (>={min_confidence}): {len(high_conf_positive)}")
    print(f"High-confidence {negative_class_name} (>={min_confidence}): {len(high_conf_negative)}")
    print(f"Selected {len(positive_candidates)} {positive_class_name} candidates")
    print(f"Selected {len(negative_candidates)} {negative_class_name} candidates")
    print(f"Total candidates for labeling: {len(all_candidates)}")

    if len(all_candidates) > 0:
        # Add labeling helper columns
        for candidate in all_candidates:
            candidate['needs_human_label'] = ''  # Empty column for human to fill
            candidate['human_confidence'] = ''   # Human confidence in the label

        # Ensure outputs directory exists
        os.makedirs("outputs", exist_ok=True)

        # Save candidates
        fieldnames = list(all_candidates[0].keys())
        with open(output_csv, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(all_candidates)

        print(f"Candidates saved to: {output_csv}")
        if positive_candidates:
            print(f"{positive_class_name} confidence range: {min(p['confidence'] for p in positive_candidates):.3f} - {max(p['confidence'] for p in positive_candidates):.3f}")
        if negative_candidates:
            print(f"{negative_class_name} confidence range: {min(p['confidence'] for p in negative_candidates):.3f} - {max(p['confidence'] for p in negative_candidates):.3f}")
    else:
        print("No candidates found.")

    return all_candidates


def run_active_learning_cycle(positive_class_id=8,
                             positive_class_name="siren",
                             negative_class_name="not_siren",
                             model_path="outputs/model_100pct_seed_42.pt",
                             urbansound_csv="data/urbansound8k/UrbanSound8K.csv",
                             run_number=1):
    """
    Run a complete active learning cycle for binary classification.

    Args:
        positive_class_id: UrbanSound8K class ID to treat as positive
        positive_class_name: Human-readable name for positive class
        negative_class_name: Human-readable name for negative class
        model_path: Path to trained model
        urbansound_csv: Path to UrbanSound8K metadata
        run_number: Version number for output files (e.g., 1 creates v1 files)

    Returns:
        tuple: (predictions_file, candidates_file)
    """

    # Step 1: Create binary labels for the specified class
    print(f"Step 1: Creating binary labels for {positive_class_name} vs {negative_class_name}...")
    binary_labels_file = create_binary_labels(
        urbansound_csv=urbansound_csv,
        positive_class_id=positive_class_id,
        output_csv=f"outputs/binary_labels_v{run_number}.csv",
        positive_class_name=positive_class_name,
        negative_class_name=negative_class_name
    )

    # Step 2: Run inference on all files
    print("\nStep 2: Running binary classification inference...")
    predictions_file = f"outputs/predictions_v{run_number}.csv"

    _ = run_binary_inference(
        model_path=model_path,
        labels_file=binary_labels_file,
        output_csv=predictions_file,
        positive_class_name=positive_class_name,
        negative_class_name=negative_class_name
    )

    # Step 3: Select candidates for active learning
    print("\nStep 3: Selecting candidates for human labeling...")
    candidates_file = f"outputs/labeling_candidates_v{run_number}.csv"

    _ = select_candidates_for_labeling(
        predictions_csv=predictions_file,
        num_positive=10,
        num_negative=10,
        min_confidence=0.8,
        output_csv=candidates_file,
        positive_class_name=positive_class_name,
        negative_class_name=negative_class_name
    )

    print("\n🎯 Active Learning Cycle Complete!")
    print("Next steps:")
    print(f"1. Review {candidates_file}")
    print("2. Add human labels to 'needs_human_label' column")
    print("3. Add those labels to your training set")
    print("4. Retrain model and repeat")

    return predictions_file, candidates_file


if __name__ == "__main__":
    # Example: Run siren detection cycle (maintains backward compatibility)
    predictions_file, candidates_file = run_active_learning_cycle(
        positive_class_id=8,
        positive_class_name="siren",
        negative_class_name="not_siren",
        model_path="outputs/model_100pct_seed_42.pt",
        run_number=1
    )
