"""Tests for LinearProbe (linear classifier over frozen 1D embeddings)."""

import pytest
import torch

from audioloop.models.linearprobe import LinearProbe


def test_sizes_linear_eagerly_from_dataset_shape():
    """The Linear is built from dataset_shape[0], with params materialized at construction.

    Eager (not LazyLinear) is the contract: the training pipeline counts parameters and builds
    the optimizer before any forward pass, so the params must exist immediately.
    """
    probe = LinearProbe(num_classes=3, dataset_shape=(768,))
    assert probe.fc.in_features == 768
    assert probe.fc.out_features == 3
    assert sum(p.numel() for p in probe.parameters()) == 768 * 3 + 3  # weights + bias


def test_in_features_takes_precedence_over_dataset_shape():
    """The reload path supplies in_features directly; it wins over dataset_shape if both given."""
    probe = LinearProbe(num_classes=2, in_features=512, dataset_shape=(768,))
    assert probe.fc.in_features == 512


def test_requires_a_sizing_argument():
    with pytest.raises(ValueError, match=r"in_features.*or.*dataset_shape"):
        LinearProbe(num_classes=2)


def test_forward_maps_batch_of_embeddings_to_logits():
    probe = LinearProbe(num_classes=4, dataset_shape=(768,))
    out = probe(torch.randn(5, 768))
    assert out.shape == (5, 4)


def test_can_handle_shape_accepts_1d_rejects_2d():
    """1D embeddings yes; 2D spectrograms no — the compatibility gate the pipeline checks."""
    probe = LinearProbe(num_classes=2, dataset_shape=(768,))
    assert probe.can_handle_shape((768,)) is True
    assert probe.can_handle_shape((128, -1)) is False


def test_get_model_info_round_trips_for_checkpoint_reload():
    """get_model_info persists in_features so load_model can rebuild an identical Linear.

    Reproduces the reload path: model_class(**{checkpoint minus state_dict}), then load_state_dict.
    """
    probe = LinearProbe(num_classes=3, dataset_shape=(768,))
    info = probe.get_model_info()
    assert info["model_type"] == "linearprobe"
    assert info["in_features"] == 768

    constructor_args = {k: v for k, v in info.items() if k != "num_parameters"}
    rebuilt = LinearProbe(**constructor_args)  # no dataset_shape available at reload
    assert rebuilt.fc.in_features == 768 and rebuilt.fc.out_features == 3
    rebuilt.load_state_dict(probe.state_dict())  # shapes match => succeeds


def test_discovered_by_registry():
    """Selectable via model_type='linearprobe' through the file-based registry."""
    from audioloop.models.model_registry import get_model_class

    assert get_model_class("linearprobe") is LinearProbe
