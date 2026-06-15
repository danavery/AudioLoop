"""Feature extraction: audio file -> feature tensor(s).

The `FeatureExtractor` base owns the shared build orchestration; concrete extractors (one
file each) override only the audio->tensor core. Re-exported here so callers import from
`audioloop.feature_extractor` regardless of which file a class lives in. Unlike `models/`
(file-discovery), extractors are selected by an explicit dict in `config.get_feature_extractor`,
so the concretes are re-exported by name (not just the base).
"""

from .base import FeatureExtractor
from .spectrogram import SpectrogramExtractor

__all__ = ["FeatureExtractor", "SpectrogramExtractor"]
