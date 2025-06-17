import torch
import torch.nn.functional as F
import torch.optim as optim
from torch import nn
from torch.utils.data import DataLoader

from .models.cnn_5layer import SoundCNN
from .utils.spec_dataset import SpectrogramDataset


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
        "run": [item["run"] for item in batch]
    }


def debug_single_epoch():
    """Debug a single training epoch step by step."""
    print("=== DEBUGGING SINGLE EPOCH ===")

    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Device: {device}")

    # Load dataset
    dataset = SpectrogramDataset(labels_file="labels.csv")
    print(f"Dataset size: {len(dataset)}")

    # Create data loader
    train_loader = DataLoader(
        dataset,
        batch_size=4,
        shuffle=False,  # Don't shuffle for debugging
        num_workers=0,
        pin_memory=False,
        collate_fn=collate_fn
    )

    # Create fresh model
    model = SoundCNN(num_classes=2, kernel_size=(3, 3)).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    print("\n=== BEFORE TRAINING ===")
    test_model_predictions(model, train_loader, device)

    print("\n=== TRAINING ONE EPOCH ===")
    model.train()

    for batch_idx, batch in enumerate(train_loader):
        print(f"\nBatch {batch_idx + 1}:")

        features = batch["data"].to(device)
        labels = batch["label"].to(device)

        print(f"  Raw features shape: {features.shape}")
        print(f"  Labels: {labels.tolist()}")

        # Add channel dimension if needed
        if features.ndim == 3:
            features = features.unsqueeze(1)
        print(f"  Final features shape: {features.shape}")

        # Forward pass
        optimizer.zero_grad()
        outputs = model(features)
        loss = criterion(outputs, labels)

        print(f"  Outputs (logits): {outputs.detach().cpu()}")
        print(f"  Loss: {loss.item():.4f}")

        # Check predictions before backward pass
        probabilities = F.softmax(outputs, dim=-1)
        predicted = torch.argmax(probabilities, dim=-1)

        print(f"  Probabilities: {probabilities.detach().cpu()}")
        print(f"  Predictions: {predicted.cpu().tolist()}")

        correct = (predicted == labels).sum().item()
        accuracy = correct / len(labels)
        print(f"  Batch accuracy: {accuracy:.4f} ({correct}/{len(labels)})")

        # Backward pass
        loss.backward()

        # Check gradients
        total_grad_norm = 0
        for name, param in model.named_parameters():
            if param.grad is not None:
                grad_norm = param.grad.data.norm(2).item()
                total_grad_norm += grad_norm ** 2
        total_grad_norm = total_grad_norm ** 0.5
        print(f"  Total gradient norm: {total_grad_norm:.6f}")

        optimizer.step()

        print(f"  After step - outputs: {model(features).detach().cpu()}")

    print("\n=== AFTER ONE EPOCH ===")
    test_model_predictions(model, train_loader, device)

    # Save model
    torch.save(model.state_dict(), "debug_model.pt")
    print("Model saved as debug_model.pt")

    print("\n=== TESTING SAVED MODEL ===")
    # Load fresh model and test
    fresh_model = SoundCNN(num_classes=2, kernel_size=(3, 3)).to(device)
    fresh_model.load_state_dict(torch.load("debug_model.pt"))
    test_model_predictions(fresh_model, train_loader, device)


def test_model_predictions(model, data_loader, device):
    """Test model predictions on data."""
    model.eval()

    all_correct = 0
    all_total = 0

    with torch.no_grad():
        for batch in data_loader:
            features = batch["data"].to(device)
            labels = batch["label"].to(device)

            if features.ndim == 3:
                features = features.unsqueeze(1)

            outputs = model(features)
            probabilities = F.softmax(outputs, dim=-1)
            predicted = torch.argmax(probabilities, dim=-1)

            for i in range(len(labels)):
                true_label = labels[i].item()
                pred_label = predicted[i].item()
                confidence = torch.max(probabilities[i]).item()

                correct = (true_label == pred_label)
                all_correct += correct
                all_total += 1

                status = "✓" if correct else "✗"
                print(f"    {status} True: {true_label} Pred: {pred_label} Conf: {confidence:.3f}")

    accuracy = all_correct / all_total if all_total > 0 else 0
    print(f"  Overall accuracy: {accuracy:.4f} ({all_correct}/{all_total})")
    return accuracy


def compare_preprocessing():
    """Compare preprocessing between training and inference."""
    print("\n=== COMPARING PREPROCESSING ===")

    dataset = SpectrogramDataset(labels_file="labels.csv")

    # Method 1: Direct dataset access
    print("\nMethod 1 - Direct dataset access:")
    sample = dataset[0]
    print(f"  Shape: {sample['data'].shape}")
    print(f"  Label: {sample['label']}")
    print(f"  Data type: {sample['data'].dtype}")
    print(f"  Data range: [{sample['data'].min():.3f}, {sample['data'].max():.3f}]")

    # Method 2: Through DataLoader
    print("\nMethod 2 - Through DataLoader:")
    loader = DataLoader(dataset, batch_size=1, collate_fn=collate_fn)
    batch = next(iter(loader))

    print(f"  Shape: {batch['data'].shape}")
    print(f"  Label: {batch['label']}")
    print(f"  Data type: {batch['data'].dtype}")
    print(f"  Data range: [{batch['data'].min():.3f}, {batch['data'].max():.3f}]")

    # Check if they're the same
    direct_data = sample['data']
    loader_data = batch['data'][0]

    if direct_data.shape[0] > 1:  # Convert stereo to mono
        direct_data = direct_data.mean(dim=0, keepdim=True)

    print(f"\nAre they equal? {torch.allclose(direct_data, loader_data)}")
    if not torch.allclose(direct_data, loader_data):
        print("  PREPROCESSING MISMATCH DETECTED!")
        print(f"  Direct shape: {direct_data.shape}")
        print(f"  Loader shape: {loader_data.shape}")


def check_label_distribution():
    """Check the label distribution in the dataset."""
    print("\n=== CHECKING LABEL DISTRIBUTION ===")

    dataset = SpectrogramDataset(labels_file="labels.csv")

    labels = []
    for i in range(len(dataset)):
        sample = dataset[i]
        labels.append(sample['label'])

    label_0_count = labels.count(0)
    label_1_count = labels.count(1)

    print(f"Label 0 (not siren): {label_0_count}")
    print(f"Label 1 (siren): {label_1_count}")
    print(f"Total: {len(labels)}")
    print(f"Balance: {label_1_count / len(labels):.2%} positive")

    if label_0_count == 0 or label_1_count == 0:
        print("WARNING: Imbalanced dataset detected!")


if __name__ == "__main__":
    print("🔍 COMPREHENSIVE TRAINING DEBUG")
    print("="*50)

    check_label_distribution()
    compare_preprocessing()
    debug_single_epoch()

    print("\n" + "="*50)
    print("🔍 DEBUG COMPLETE")
