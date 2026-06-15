"""`EmbeddingExtractor`: frozen pretrained-model embeddings (regime A).

Wraps a frozen audio model (wav2vec2 today, AVES next) as a feature extractor: decode the
waveform, run the frozen backbone, mean-pool the final layer to a single (D,) vector per
file. This is regime A — frozen weights + fixed (deterministic) inputs — so the output is
cacheable exactly like a spectrogram. The backbone is loaded lazily on first use; most call
sites only need the static output shape / cache paths and never touch the weights.

Augmentation deliberately does NOT live here: `extract_one` must stay deterministic so the
cached `.pt` is the one true embedding. Input-space augmentation (which would force an
uncached, per-epoch recompute — regime B) belongs to a later `__getitem__` stage, not the
cacheable core.
"""

from pathlib import Path

import torch
import torchaudio
from torchaudio.models import Wav2Vec2Model

from .base import FeatureExtractor

# model_name -> (torchaudio pipelines bundle, embedding dim). The bundle is a lightweight
# descriptor; bundle.get_model() is what downloads weights, so this dict is import-cheap and
# lets get_output_shape answer (D,) without loading anything. The model name is also the cache
# coexistence axis (see cache_subdir), so distinct models MUST be distinct keys.
_MODELS = {
    "wav2vec2": (torchaudio.pipelines.WAV2VEC2_BASE, 768),
}


class EmbeddingExtractor(FeatureExtractor):
    """Produce a single pooled embedding vector per audio file from a frozen backbone."""

    def __init__(self, dataset_config, *, model_name: str = "wav2vec2", sample_rate: int = 16000):
        if model_name not in _MODELS:
            raise ValueError(
                f"Unknown embedding model {model_name!r}; expected one of {sorted(_MODELS)}"
            )
        super().__init__(dataset_config, sample_rate=sample_rate)
        self.model_name = model_name
        self._model: Wav2Vec2Model | None = None  # lazily loaded on first extract_one

    @property
    def cache_subdir(self) -> str:
        # Includes the model name: wav2vec2 vs AVES are the same class with a different param,
        # so a class-level token would let one silently reuse the other's cache.
        return f"embed_{self.model_name}"

    def cache_params(self) -> dict:
        return {"model_name": self.model_name, "sample_rate": self.sample_rate}

    def get_output_shape(self) -> tuple[int, ...]:
        # Static: the embedding dim is known from the model registry, no weights needed. This is
        # what lets training-over-cached-features and the shape-compat check avoid loading 360MB.
        return (_MODELS[self.model_name][1],)

    def _get_model(self) -> Wav2Vec2Model:
        """Lazily load + freeze the backbone (downloads weights on first call)."""
        if self._model is None:
            bundle, _ = _MODELS[self.model_name]
            model = bundle.get_model()
            # bundle.get_model() is stub-typed as base Module; assert the real runtime type
            # (which exposes extract_features) so it's narrowed AND verified, not just trusted.
            assert isinstance(model, Wav2Vec2Model)
            model.eval()
            self._model = model
        return self._model

    def extract_one(self, audio_path: Path) -> list[torch.Tensor]:
        """Decode -> frozen backbone -> mean-pool the final layer to one (D,) vector.

        Returns a single-element list (the N=1 case): the pooled embedding is one feature per
        file, reusing the list[Tensor] contract with no segment machinery. Deterministic by
        design (no augmentation) so the result is cacheable.
        """
        waveform = self._load_audio(audio_path)  # (1, samples) mono @ self.sample_rate
        with torch.inference_mode():
            # NB extract_features returns a TUPLE (List[Tensor], lengths); unpack and drop
            # lengths. feats[-1] is the final transformer layer, shape (1, T, D).
            feats, _ = self._get_model().extract_features(waveform)
        embedding = feats[-1].mean(dim=1).squeeze(0)  # (1, T, D) -> (1, D) -> (D,)
        return [embedding]
