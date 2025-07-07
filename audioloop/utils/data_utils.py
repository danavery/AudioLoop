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
