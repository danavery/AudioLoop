"""
Shared data utilities for AudioLoop.

This module provides common data processing functions used across
training, inference, and active learning workflows.
"""

import torch


def simple_collate_fn(batch):
    """
    Collate function for fixed-length spectrograms.

    Used throughout the pipeline where spectrograms are expected to be the same size.
    All spectrograms are expected to be mono (single channel).

    Args:
        batch: List of dataset items with keys: data, label, filename, filepath

    Returns:
        dict: Batched data with tensors for data and labels
    """
    # Extract data and other fields
    data_list = [item["data"] for item in batch]
    labels = torch.tensor([item["label"] for item in batch])

    # Stack the spectrograms (they should all be the same size)
    data_tensor = torch.stack(data_list)

    # Return in the same format as the original batch
    return {
        "data": data_tensor,
        "label": labels,
        "filename": [item["filename"] for item in batch],
        "filepath": [item["filepath"] for item in batch],
    }


def variable_length_collate_fn(batch):
    """
    Collate function for variable-length spectrograms.

    Pads spectrograms within each batch to the longest sequence in that batch.
    This enables training with natural spectrogram lengths while maintaining
    consistent batch tensor shapes for PyTorch.

    Filters out None values from failed samples (missing/corrupted files).

    Args:
        batch: List of dataset items with keys: data, label, filename, filepath
               (or None for failed samples)

    Returns:
        dict: Batched data with tensors padded to max length in batch
    """
    # Filter out None values from failed samples (missing/corrupted files)
    batch = [item for item in batch if item is not None]

    # If entire batch failed, return empty batch
    if len(batch) == 0:
        return {
            "data": torch.empty(0),
            "label": torch.empty(0, dtype=torch.long),
            "filename": [],
            "filepath": [],
        }

    # Extract data and other fields
    data_list = [item["data"] for item in batch]
    labels = torch.tensor([item["label"] for item in batch])

    # Find the maximum length in this batch
    max_length = max(spec.shape[-1] for spec in data_list)

    # Pad all spectrograms to the max length
    padded_data = []
    for spec in data_list:
        current_length = spec.shape[-1]
        if current_length < max_length:
            # Pad with zeros on the right
            pad_size = max_length - current_length
            padded_spec = torch.nn.functional.pad(spec, (0, pad_size), mode="constant", value=0)
        else:
            padded_spec = spec
        padded_data.append(padded_spec)

    # Stack the padded spectrograms (now all same size)
    data_tensor = torch.stack(padded_data)

    result = {
        "data": data_tensor,
        "label": labels,
        "filename": [item["filename"] for item in batch],
        "filepath": [item["filepath"] for item in batch],
    }

    # Include optional fields if present in any item
    if any("original_class" in item for item in batch):
        result["original_class"] = [item.get("original_class") for item in batch]

    return result


def get_device():
    """
    Get the best available device for PyTorch operations.

    Returns:
        torch.device: CUDA if available, MPS if on Apple Silicon, otherwise CPU
    """
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def entropy(probabilities):
    """
    Calculate entropy of probability distribution.

    Args:
        probabilities: Tensor of probabilities with shape (batch_size, num_classes)

    Returns:
        torch.Tensor: Entropy values with shape (batch_size,)
    """
    # Avoid log(0) by adding small epsilon
    eps = 1e-10
    probs = probabilities + eps
    return -torch.sum(probs * torch.log(probs), dim=-1)
