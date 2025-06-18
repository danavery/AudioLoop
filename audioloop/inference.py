import csv
import os

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from .models.cnn_5layer import SoundCNN
from .utils.urbansound_dataset import UrbanSoundDataset


def entropy(probabilities):
    """Calculate entropy of probability distribution."""
    # Avoid log(0) by adding small epsilon
    eps = 1e-10
    probs = probabilities + eps
    return -torch.sum(probs * torch.log(probs), dim=-1)


def load_model(model_path, num_classes, device):
    """Load trained model from checkpoint."""
    model = SoundCNN(num_classes=num_classes, kernel_size=(3, 3))
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
        "filepath": [item["filepath"] for item in batch]
    }


def run_inference(model_path, labels_file="outputs/urbansound8k_files.csv", output_csv="outputs/predictions.csv"):
    """Run inference on all UrbanSound8K data and save predictions to CSV."""
    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load dataset
    dataset = UrbanSoundDataset(csv_file=labels_file)
    print(f"Dataset size: {len(dataset)}")

    # For UrbanSound8K, we know there are 10 classes (0-9), but our model is binary
    # So we'll use 2 classes for our trained model
    num_classes = 2
    print("Using binary classification model (2 classes)")

    # Also report the UrbanSound8K class distribution
    us8k_labels = [dataset[i]["label"] for i in range(min(1000, len(dataset)))]  # Sample first 1000 for speed
    us8k_classes = len(set(us8k_labels))
    print(f"UrbanSound8K has {us8k_classes} original classes in sample")

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
    print("Running inference...")

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
                # Get original audio filename (remove .pt extension and path)
                spec_filename = batch["filename"][i]
                audio_filename = spec_filename.replace(".pt", ".wav")

                # Extract probability for each class
                prob_negative = probabilities[i][0].item()  # Class 0 probability
                prob_positive = probabilities[i][1].item()  # Class 1 probability

                result = {
                    "filename": audio_filename,
                    "urbansound_class": true_labels[i].item(),  # Original UrbanSound8K class (0-9)
                    "predicted_label": predicted_classes[i].item(),
                    "prediction": "positive" if predicted_classes[i].item() == 1 else "negative",
                    "confidence": confidences[i].item(),
                    "entropy": entropies[i].item(),
                    "prob_negative": prob_negative,
                    "prob_positive": prob_positive,
                    "filepath": batch["filepath"][i]
                }
                results.append(result)

    # Calculate summary statistics
    total_samples = len(results)
    positive_predictions = sum(1 for r in results if r["prediction"] == "positive")
    negative_predictions = total_samples - positive_predictions

    print("\nInference Results:")
    print(f"Total samples: {total_samples}")
    print(f"Positive predictions: {positive_predictions}")
    print(f"Negative predictions: {negative_predictions}")

    # Show distribution by UrbanSound8K class
    class_counts = {}
    for r in results:
        us_class = r["urbansound_class"]
        if us_class not in class_counts:
            class_counts[us_class] = {"total": 0, "positive": 0}
        class_counts[us_class]["total"] += 1
        if r["prediction"] == "positive":
            class_counts[us_class]["positive"] += 1

    print("\nPredictions by UrbanSound8K class:")
    for class_id in sorted(class_counts.keys()):
        total = class_counts[class_id]["total"]
        pos = class_counts[class_id]["positive"]
        print(f"  Class {class_id}: {pos}/{total} positive ({pos/total:.2%})")

    # Calculate mean confidence and entropy
    mean_confidence = sum(r["confidence"] for r in results) / total_samples
    mean_entropy = sum(r["entropy"] for r in results) / total_samples
    print(f"Mean confidence: {mean_confidence:.4f}")
    print(f"Mean entropy: {mean_entropy:.4f}")

    # Ensure outputs directory exists
    os.makedirs("outputs", exist_ok=True)

    # Save results to CSV
    fieldnames = [
        "filename", "urbansound_class", "predicted_label", "prediction",
        "confidence", "entropy", "prob_negative", "prob_positive",
        "filepath"
    ]

    with open(output_csv, 'w', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)

    print(f"\nResults saved to: {output_csv}")
    return results


if __name__ == "__main__":
    # Run inference with the trained model on full UrbanSound8K dataset
    model_path = "outputs/model_100pct_seed_42.pt"
    output_file = "outputs/urbansound8k_predictions.csv"

    results = run_inference(
        model_path=model_path,
        labels_file="outputs/urbansound8k_files.csv",
        output_csv=output_file
    )

    print(f"\nInference complete! Check {output_file} for detailed results.")
