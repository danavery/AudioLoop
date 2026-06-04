"""Feature extraction: audio file -> feature tensor.

`SpectrogramExtractor` owns the audio->spectrogram production pipeline (load -> transform
-> fix) AND the audio-processing parameters (sample_rate, n_fft, ...) that previously lived
on `DatasetConfig`. It is the unification point for the offline build path (`create_specs`)
and the lazy path (`SpectrogramDataset`).

Params are currently constructor defaults; experiment-level overrides (sourced from
`AudioLoopConfig`) arrive in A3, when a FeatureSet built from config flows the configured
extractor into both build paths consistently.
"""

import logging
from pathlib import Path

import torch
import torchaudio
from torch import nn
from torchcodec.decoders import AudioDecoder

from audioloop.utils.log_normalize import LogNormalize

logger = logging.getLogger(__name__)


class SpectrogramExtractor:
    """Produce log-mel spectrogram tensors from audio files.

    The load -> transform -> fix behavior is identical across all datasets; only the
    parameter *values* differ, so this is a single concrete class parameterized by its
    constructor arguments rather than a per-dataset subclass. These are *feature* params
    (the decode/STFT/mel target), owned by the extractor — not dataset identity.

    The `dataset_config` reference is retained for file-level dataset knowledge used by the
    build step (get_spectrogram_path, get_bad_files, min_audio_file_size), not for params.

    Experiment-level overrides of these defaults arrive in A3 (FeatureSet, built from
    AudioLoopConfig, flows the configured extractor into both build paths consistently).
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
        self.dataset_config = dataset_config
        self.sample_rate = sample_rate
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.n_mels = n_mels
        self.top_db = top_db
        self.max_spectrogram_length = max_spectrogram_length

    def extract_one(self, audio_path: Path) -> torch.Tensor:
        """Produce the feature tensor for one audio file: load -> transform -> fix.

        This is the pure audio->tensor core; callers retain their own surrounding policy
        (existence/corruption guards, caching, stats, filename derivation).
        """
        waveform = self._load_audio(audio_path)
        spec = self._create_transform()(waveform)
        return self._fix_length(spec)

    def process_one(self, file_info: dict, output_dir: Path) -> tuple[bool, int | None]:
        """Build and cache one file's feature tensor: the offline (create_specs) build step.

        Applies all guards uniformly before extraction — resumable skip, audio existence,
        the dataset's known-bad files, and minimum file size — then runs extract_one and
        caches the result under get_spectrogram_path. Returns (success, feature_length):
        length is None for files skipped because they were already built. Per-file
        skips/failures are counted by the caller, not logged here, to keep the progress
        bar readable; only unexpected exceptions are logged.
        """
        config = self.dataset_config
        try:
            audio_path = file_info["audio_path"]
            filename = file_info["filename"]
            output_path = config.get_spectrogram_path(filename, output_dir)

            # Resumable: skip files already built (forced rebuild via create_specs clear_output).
            if output_path.exists():
                return True, None

            # Skip missing / known-bad / too-small files (counted in stats, not logged per-file).
            if not audio_path.exists():
                return False, None
            if filename in config.get_bad_files():
                return False, None
            min_size = config.min_audio_file_size
            if min_size is not None and audio_path.stat().st_size < min_size:
                return False, None

            spec = self.extract_one(audio_path)
            torch.save(spec, output_path)
            return True, spec.shape[-1]

        except Exception as e:
            logger.error(f"Error processing {file_info['filename']}: {e}")
            return False, None

    def _load_audio(self, audio_path: Path) -> torch.Tensor:
        """Load audio with torchcodec, resampling to the target rate and converting to mono."""
        decoder = AudioDecoder(str(audio_path), sample_rate=self.sample_rate, num_channels=1)
        return decoder.get_all_samples().data

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
