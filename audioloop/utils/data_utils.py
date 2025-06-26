"""
Shared data utilities for AudioLoop.

This module provides common data processing functions used across
training, inference, and active learning workflows.
"""

import torch


def simple_collate_fn(batch):
    """
    Simple collate function for fixed-length spectrograms.

    Used in training where spectrograms are expected to be the same size.
    Converts stereo to mono by averaging channels.

    Args:
        batch: List of dataset items with keys: data, label, filename, filepath, run

    Returns:
        dict: Batched data with tensors for data and labels
    """
    # Extract data and other fields
    data_list = [item["data"] for item in batch]
    labels = torch.tensor([item["label"] for item in batch])

    # Convert to mono by averaging channels if stereo
    mono_data = []
    for spec in data_list:
        if spec.shape[0] > 1:
            spec = spec.mean(dim=0, keepdim=True)  # Average channels, keep dimension
        mono_data.append(spec)

    # Stack the spectrograms (they should all be the same size now)
    data_tensor = torch.stack(mono_data)

    # Return in the same format as the original batch
    return {
        "data": data_tensor,
        "label": labels,
        "filename": [item["filename"] for item in batch],
        "filepath": [item["filepath"] for item in batch],
        "run": [item.get("run", "1") for item in batch],
    }


def variable_length_collate_fn(batch):
    """
    Collate function for variable-length spectrograms.

    Used in inference where spectrograms may have different time lengths.
    Pads all spectrograms to the maximum length in the batch.

    Args:
        batch: List of dataset items with keys: data, label, filename, filepath

    Returns:
        dict: Batched data with padded tensors
    """
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
            padded_spec = torch.nn.functional.pad(spec, pad_tuple, mode="constant", value=0)
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
