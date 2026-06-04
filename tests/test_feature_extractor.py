"""Tests for SpectrogramExtractor (the audio->tensor production split out of DatasetConfig).

These cover the behavior that previously lived on DatasetConfig as
create_spectrogram_transform / fix_spectrogram_length / get_output_shape, now owned by the
extractor and parameterized by the dataset config's get_audio_processing_params().
"""

from pathlib import Path

import pytest
import torch

from audioloop.datasets.audioset_config import AudiosetConfig
from audioloop.datasets.fsd50k_config import FSD50KConfig
from audioloop.datasets.urbansound8k_config import UrbanSound8KConfig
from audioloop.feature_extractor import SpectrogramExtractor

ALL_CONFIGS = [FSD50KConfig, UrbanSound8KConfig, AudiosetConfig]


def test_feature_extractor_property_caches():
    """DatasetConfig.feature_extractor builds once and returns the same instance."""
    config = FSD50KConfig()
    fx = config.feature_extractor
    assert isinstance(fx, SpectrogramExtractor)
    assert config.feature_extractor is fx


@pytest.mark.parametrize("config_class", ALL_CONFIGS)
def test_get_output_shape(config_class):
    """Output shape is (n_mels, -1): correct freq dim and variable-time sentinel."""
    config = config_class()
    shape = config.feature_extractor.get_output_shape()

    assert isinstance(shape, tuple)
    assert len(shape) == 2
    assert all(isinstance(dim, int) for dim in shape)
    assert shape[0] == config._n_mels
    assert shape[1] == -1  # variable time dimension


@pytest.mark.parametrize("config_class", [FSD50KConfig, UrbanSound8KConfig])
def test_fix_length_crops_outliers_and_preserves_short(config_class):
    """_fix_length center-crops above max_spectrogram_length, never pads short clips."""
    config = config_class()
    fx = config.feature_extractor
    max_length = config._max_spectrogram_length

    # Short: preserved, no padding.
    assert fx._fix_length(torch.randn(1, 128, 100)).shape[-1] == 100

    # Within limit: preserved.
    medium = torch.randn(1, 128, max_length // 2)
    assert fx._fix_length(medium).shape[-1] == max_length // 2

    # Outlier: cropped to the max.
    outlier = torch.randn(1, 128, max_length * 2)
    assert fx._fix_length(outlier).shape[-1] == max_length


def test_extract_one_composes_load_transform_fix(monkeypatch):
    """extract_one runs load -> transform -> fix, in that order."""
    config = FSD50KConfig()
    fx = config.feature_extractor
    max_length = config._max_spectrogram_length

    seen = {}

    def fake_load(audio_path):
        seen["loaded"] = audio_path
        return torch.zeros(1, 16000)

    def fake_transform():
        def _t(waveform):
            seen["transformed"] = waveform.shape
            return torch.zeros(128, max_length * 2)  # over-length -> fix must crop

        return _t

    monkeypatch.setattr(fx, "_load_audio", fake_load)
    monkeypatch.setattr(fx, "_create_transform", fake_transform)

    out = fx.extract_one(Path("clip.wav"))

    assert seen["loaded"] == Path("clip.wav")  # load ran first, on the given path
    assert seen["transformed"] == (1, 16000)  # transform ran on the loaded waveform
    assert out.shape[-1] == max_length  # fix ran on the transform output (cropped)
