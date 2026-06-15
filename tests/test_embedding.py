"""Tests for EmbeddingExtractor (frozen pretrained-model embeddings).

The real wav2vec2 backbone is a 360MB download, so these stub it: the static surface
(get_output_shape / cache_subdir / cache_params) needs no weights at all, and the
pooling + tuple-unpack contract is pinned with a fake model. The real wav2vec2 forward
path was smoke-tested separately (Step 0).
"""

from pathlib import Path
from unittest.mock import Mock

import pytest
import torch

from audioloop.feature_extractor import EmbeddingExtractor


def _make(**kwargs):
    """An EmbeddingExtractor over a dummy dataset_config (unused by the paths under test)."""
    return EmbeddingExtractor(Mock(), **kwargs)


def test_output_shape_is_static_768():
    """get_output_shape answers (768,) from the model registry, no weights loaded."""
    fx = _make()
    assert fx.get_output_shape() == (768,)


def test_construction_does_not_load_backbone():
    """The backbone is lazy: building the extractor must not load weights."""
    assert _make()._model is None


def test_cache_subdir_includes_model_name():
    """cache_subdir carries the model name so wav2vec2 and aves caches never collide."""
    assert _make(model_name="wav2vec2").cache_subdir == "embed_wav2vec2"


def test_cache_params_are_model_name_and_sample_rate():
    fx = _make(sample_rate=16000)
    assert fx.cache_params() == {"model_name": "wav2vec2", "sample_rate": 16000}


def test_unknown_model_name_raises():
    with pytest.raises(ValueError, match="Unknown embedding model 'bogus'"):
        _make(model_name="bogus")


def test_extract_one_unpacks_tuple_and_pools_to_single_vector(monkeypatch):
    """extract_one unpacks (feats, lengths), mean-pools feats[-1] over time to a (768,) list."""
    fx = _make()

    # Bypass audio decode and the real backbone.
    monkeypatch.setattr(fx, "_load_audio", lambda p: torch.zeros(1, 16000))
    fake_model = Mock()
    # extract_features returns (List[Tensor], lengths); final layer is (1, T, 768).
    fake_model.extract_features.return_value = ([torch.ones(1, 5, 768)], None)
    monkeypatch.setattr(fx, "_get_model", lambda: fake_model)

    out = fx.extract_one(Path("clip.wav"))

    assert isinstance(out, list) and len(out) == 1  # N=1 contract
    assert out[0].shape == (768,)
    assert torch.allclose(out[0], torch.ones(768))  # mean over time of all-ones is ones
