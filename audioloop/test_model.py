import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from .models.simple_cnn import SimpleCNN
from .simple_train import collate_fn
from .utils.spec_dataset import SpectrogramDataset


def test_model_on_training_data(model_path="outputs/model_100pct_seed_42.pt", labels_file="labels.csv"):
    """Test the trained model on its original training data to verify it works."""

    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load the dataset (same as training)
    dataset = SpectrogramDataset(labels_file=labels_file)
    print(f"Training dataset size: {len(dataset)}")

    # Create data loader (same as training)
    loader = DataLoader(dataset, batch_size=4, shuffle=False, collate_fn=collate_fn)

    # Load the model (using SimpleCNN now)
    model = SimpleCNN(num_classes=2)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()

    print("\nTesting model on training data:")
    print("-" * 60)

    total_correct = 0
    total_samples = 0

    with torch.no_grad():
        for batch_idx, batch in enumerate(loader):
            features = batch["data"].to(device)
            labels = batch["label"].to(device)
            filenames = batch["filename"]

            # Add channel dimension if needed (same as training)
            if features.ndim == 3:
                features = features.unsqueeze(1)

            print(f"Batch {batch_idx + 1} input shape: {features.shape}")

            # Forward pass
            logits = model(features)
            probabilities = F.softmax(logits, dim=-1)
            predicted_classes = torch.argmax(probabilities, dim=-1)

            # Check each sample in batch
            for i in range(len(labels)):
                true_label = labels[i].item()
                pred_label = predicted_classes[i].item()
                confidence = torch.max(probabilities[i]).item()
                prob_0 = probabilities[i][0].item()
                prob_1 = probabilities[i][1].item()

                correct = (true_label == pred_label)
                total_correct += correct
                total_samples += 1

                status = "✓" if correct else "✗"
                print(f"  {status} {filenames[i][:20]:<20} True: {true_label} Pred: {pred_label} "
                      f"Conf: {confidence:.3f} P(0): {prob_0:.3f} P(1): {prob_1:.3f}")

    accuracy = total_correct / total_samples
    print("-" * 60)
    print(f"Training Accuracy: {accuracy:.4f} ({total_correct}/{total_samples})")

    if accuracy < 0.95:
        print("⚠️  WARNING: Model accuracy on training data is below 95%!")
        print("   This suggests the model may not have trained properly.")
    elif accuracy == 1.0:
        print("✅ Model has 100% accuracy on training data - as expected!")
    else:
        print("✅ Model has good accuracy on training data.")

    return accuracy


if __name__ == "__main__":
    accuracy = test_model_on_training_data()
