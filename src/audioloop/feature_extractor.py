"""Feature extraction: audio file -> feature tensor.

`SpectrogramExtractor` owns the audio->spectrogram production pipeline (load -> transform
-> fix) that previously lived on `DatasetConfig`. It is the unification point for the
offline build path (`create_specs`) and the lazy path (`SpectrogramDataset`).

(a)-phase note: the extractor currently reads its audio-processing parameters back from a
`DatasetConfig` via `get_audio_processing_params()`. A later step (A2c) relocates those
values onto the extractor itself (constructed from `AudioLoopConfig`), at which point the
dataset config sheds audio processing entirely.
"""

import logging
from pathlib import Path
from typing import Any

import torch
import torchaudio
from torch import nn
from torchcodec.decoders import AudioDecoder

from audioloop.utils.log_normalize import LogNormalize

logger = logging.getLogger(__name__)


class SpectrogramExtractor:
    """Produce log-mel spectrogram tensors from audio files.

    The load -> transform -> fix behavior is identical across all datasets; only the
    parameter *values* differ, so this is a single concrete class parameterized by the
    config's `get_audio_processing_params()` rather than a per-dataset subclass.
    """

    def __init__(self, dataset_config):
        self.dataset_config = dataset_config

    @property
    def _params(self) -> dict[str, Any]:
        return self.dataset_config.get_audio_processing_params()

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
        target_sample_rate = self._params["sample_rate"]
        decoder = AudioDecoder(str(audio_path), sample_rate=target_sample_rate, num_channels=1)
        return decoder.get_all_samples().data

    def _create_transform(self) -> nn.Sequential:
        """Build the mel-spectrogram + log-normalize transform pipeline."""
        p = self._params
        return nn.Sequential(
            torchaudio.transforms.MelSpectrogram(
                sample_rate=p["sample_rate"],
                n_fft=p["n_fft"],
                hop_length=p["hop_length"],
                n_mels=p["n_mels"],
            ),
            LogNormalize(top_db=p["top_db"]),
        )

    def _fix_length(self, spec: torch.Tensor) -> torch.Tensor:
        """Crop outliers longer than max_spectrogram_length (center crop); never pad short."""
        max_length = self._params["max_spectrogram_length"]
        current_length = spec.shape[-1]  # Time dimension is last
        if current_length > max_length:
            start_idx = (current_length - max_length) // 2
            spec = spec[..., start_idx : start_idx + max_length]
        return spec

    def get_output_shape(self) -> tuple[int, ...]:
        """Shape of tensors this extractor produces (excluding batch dim)."""
        return (self._params["n_mels"], -1)  # -1 indicates variable time dimension
