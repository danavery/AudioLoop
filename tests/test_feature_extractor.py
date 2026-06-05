"""Tests for SpectrogramExtractor (the audio->tensor production split out of DatasetConfig).

These cover the behavior that previously lived on DatasetConfig as
create_spectrogram_transform / fix_spectrogram_length / get_output_shape, now owned by the
extractor and parameterized by its own constructor args (default 44.1kHz log-mel).
"""

from pathlib import Path

import pytest
import torch

from audioloop.datasets.audioset_config import AudiosetConfig
from audioloop.datasets.fsd50k_config import FSD50KConfig
from audioloop.datasets.urbansound8k_config import UrbanSound8KConfig
from audioloop.feature_extractor import SpectrogramExtractor

ALL_CONFIGS = [FSD50KConfig, UrbanSound8KConfig, AudiosetConfig]


@pytest.mark.parametrize("config_class", ALL_CONFIGS)
def test_get_output_shape(config_class):
    """Output shape is (n_mels, -1): correct freq dim and variable-time sentinel."""
    config = config_class()
    fx = SpectrogramExtractor(config)
    shape = fx.get_output_shape()

    assert isinstance(shape, tuple)
    assert len(shape) == 2
    assert all(isinstance(dim, int) for dim in shape)
    assert shape[0] == fx.n_mels
    assert shape[1] == -1  # variable time dimension


@pytest.mark.parametrize("config_class", [FSD50KConfig, UrbanSound8KConfig])
def test_fix_length_crops_outliers_and_preserves_short(config_class):
    """_fix_length center-crops above max_spectrogram_length, never pads short clips."""
    config = config_class()
    fx = SpectrogramExtractor(config)
    max_length = fx.max_spectrogram_length

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
    fx = SpectrogramExtractor(config)
    max_length = fx.max_spectrogram_length

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
    # N=1 fallback: SpectrogramExtractor never windows, so the result is a single-element list.
    assert isinstance(out, list)
    assert len(out) == 1
    assert out[0].shape[-1] == max_length  # fix ran on the transform output (cropped)


def test_process_one_builds_and_saves(tmp_path, monkeypatch):
    """process_one extracts and caches the spec under get_cached_feature_path, returns length."""
    config = FSD50KConfig()
    fx = SpectrogramExtractor(config)

    audio = tmp_path / "clip.wav"
    audio.write_bytes(b"ignored")  # must exist; content irrelevant (extract_one is stubbed)
    out_dir = tmp_path / "specs"
    out_dir.mkdir()

    monkeypatch.setattr(fx, "extract_one", lambda p: [torch.zeros(128, 42)])

    ok, length = fx.process_one({"audio_path": audio, "filename": "clip.wav"}, out_dir)

    assert ok is True
    assert length == 42
    assert fx.get_cached_feature_path("clip.wav", out_dir).exists()


def test_process_one_skips_already_built(tmp_path, monkeypatch):
    """Resumable: an existing spec is skipped (True, None) without invoking the extractor."""
    config = FSD50KConfig()
    fx = SpectrogramExtractor(config)

    audio = tmp_path / "clip.wav"
    audio.write_bytes(b"ignored")
    out_dir = tmp_path / "specs"
    out_dir.mkdir()
    # Pre-build the cached spec.
    torch.save(torch.zeros(128, 10), fx.get_cached_feature_path("clip.wav", out_dir))

    calls = []
    monkeypatch.setattr(fx, "extract_one", lambda p: calls.append(p) or [torch.zeros(128, 10)])

    ok, length = fx.process_one({"audio_path": audio, "filename": "clip.wav"}, out_dir)

    assert ok is True
    assert length is None  # skipped: no length recorded
    assert calls == []  # extractor never ran


def test_process_one_refuses_multi_segment(tmp_path, monkeypatch):
    """N=1 guard: a multi-segment extractor is a failure, not a silent drop of windows.

    Multi-segment caching ({stem}__seg{i}.pt) is wired in Arc B2; until then process_one
    must not save only the first window. Surfaces as a quiet (False, None) with no artifact.
    """
    config = FSD50KConfig()
    fx = SpectrogramExtractor(config)

    audio = tmp_path / "clip.wav"
    audio.write_bytes(b"ignored")
    out_dir = tmp_path / "specs"
    out_dir.mkdir()

    monkeypatch.setattr(fx, "extract_one", lambda p: [torch.zeros(128, 42), torch.zeros(128, 42)])

    ok, length = fx.process_one({"audio_path": audio, "filename": "clip.wav"}, out_dir)

    assert ok is False
    assert length is None
    assert not fx.get_cached_feature_path("clip.wav", out_dir).exists()  # nothing written


def test_process_one_missing_audio_is_failure(tmp_path):
    """A missing audio file is a quiet failure (False, None)."""
    config = FSD50KConfig()
    fx = SpectrogramExtractor(config)
    out_dir = tmp_path / "specs"
    out_dir.mkdir()

    ok, length = fx.process_one(
        {"audio_path": tmp_path / "nope.wav", "filename": "nope.wav"}, out_dir
    )

    assert ok is False
    assert length is None
