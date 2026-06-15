"""`SpectrogramExtractor`: the log-mel spectrogram concrete (load -> transform -> fix).

Owns the audio-processing parameters (sample_rate, n_fft, ...) that previously lived on
`DatasetConfig`, and is the unification point for the offline build path (`build_features`)
and the lazy path (`CachedFeatureDataset`). Params are constructor defaults; experiment-level
overrides come from `AudioLoopConfig.feature_extractor_kwargs` (see
`config.get_feature_extractor`), which constructs the extractor once and flows it into both
build paths consistently.
"""

from pathlib import Path

import torch
import torchaudio
from torch import nn

from audioloop.utils.log_normalize import LogNormalize

from .base import FeatureExtractor


class SpectrogramExtractor(FeatureExtractor):
    """Produce log-mel spectrogram tensors from audio files.

    The load -> transform -> fix behavior is identical across all datasets; only the
    parameter *values* differ, so this is a single concrete class parameterized by its
    constructor arguments rather than a per-dataset subclass. These are *feature* params
    (the decode/STFT/mel target), owned by the extractor — not dataset identity.

    The `dataset_config` reference is retained for file-level dataset knowledge used by the
    build step (get_bad_files, min_audio_file_size), not for params.

    Experiment-level overrides of these defaults come from
    `AudioLoopConfig.feature_extractor_kwargs` (see `config.get_feature_extractor`).
    """

    def __init__(
        self,
        dataset_config,
        *,
        sample_rate: int = 44100,
        n_fft: int = 1024,
        hop_length: int = 256,
        n_mels: int = 128,
        top_db: int = 80,
        max_spectrogram_length: int = 2048,
    ):
        super().__init__(dataset_config, sample_rate=sample_rate)
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.n_mels = n_mels
        self.top_db = top_db
        self.max_spectrogram_length = max_spectrogram_length

    @property
    def cache_subdir(self) -> str:
        return "spectrogram"

    def cache_params(self) -> dict:
        """The STFT/mel params that define the produced spectrogram (see base.cache_params)."""
        return {
            "sample_rate": self.sample_rate,
            "n_fft": self.n_fft,
            "hop_length": self.hop_length,
            "n_mels": self.n_mels,
            "top_db": self.top_db,
            "max_spectrogram_length": self.max_spectrogram_length,
        }

    def extract_one(self, audio_path: Path) -> list[torch.Tensor]:
        """Produce the feature tensor(s) for one audio file: load -> transform -> fix.

        Returns a *list* of feature tensors, one per segment. SpectrogramExtractor does not
        window *today*, so it returns a single-element list (the N=1 case); a windowed
        extractor (e.g. Perch's fixed 5s input, or a future spectrogram-windowing mode that
        tiles long clips instead of center-cropping them in _fix_length) returns one tensor
        per window. The list is the contract that lets each extractor own its own windowing
        without the rest of the pipeline knowing the cardinality up front — including, later,
        this same class once it grows a window_length/hop knob.

        This is the pure audio->tensor core; callers retain their own surrounding policy
        (existence/corruption guards, caching, stats, filename derivation).
        """
        waveform = self._load_audio(audio_path)
        spec = self._create_transform()(waveform)
        return [self._fix_length(spec)]

    def _create_transform(self) -> nn.Sequential:
        """Build the mel-spectrogram + log-normalize transform pipeline."""
        return nn.Sequential(
            torchaudio.transforms.MelSpectrogram(
                sample_rate=self.sample_rate,
                n_fft=self.n_fft,
                hop_length=self.hop_length,
                n_mels=self.n_mels,
            ),
            LogNormalize(top_db=self.top_db),
        )

    def _fix_length(self, spec: torch.Tensor) -> torch.Tensor:
        """Crop outliers longer than max_spectrogram_length (center crop); never pad short."""
        current_length = spec.shape[-1]  # Time dimension is last
        if current_length > self.max_spectrogram_length:
            start_idx = (current_length - self.max_spectrogram_length) // 2
            spec = spec[..., start_idx : start_idx + self.max_spectrogram_length]
        return spec

    def get_output_shape(self) -> tuple[int, ...]:
        """Shape of tensors this extractor produces (excluding batch dim)."""
        return (self.n_mels, -1)  # -1 indicates variable time dimension
