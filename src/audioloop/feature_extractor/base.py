"""The shared `FeatureExtractor` base: build orchestration common to every extractor.

`FeatureExtractor` owns the parts that are identical across extractors: the offline
`process_one` build step and its guards, audio loading (`_load_audio`), cache-path
resolution (`get_cached_feature_path`), and the per-extractor cache subdir + validating
`extractor.json` manifest (`ensure_cache_dir`). Concrete extractors override only
`extract_one` (the audio->tensor core), `get_output_shape`, `cache_subdir`, and
`cache_params`; see `spectrogram.SpectrogramExtractor` for the log-mel concrete.
"""

import json
import logging
from pathlib import Path

import torch
from torchcodec.decoders import AudioDecoder

logger = logging.getLogger(__name__)


class FeatureExtractor:
    """Shared base: the build orchestration common to every extractor.

    Owns `process_one` (the offline create_specs build step) and its guards, `_load_audio`
    (decode/resample/mono to `self.sample_rate`), `get_cached_feature_path` (the .pt cache
    artifact path), and `ensure_cache_dir` (the per-extractor cache subdir + `extractor.json`
    validity stamp) — all verified extractor-generic: they touch only dataset_config
    knowledge, the target sample rate, `cache_subdir`/`cache_params`, and `extract_one`.
    Concrete extractors override `extract_one`, `get_output_shape`, `cache_subdir`, and
    `cache_params`; everything else is inherited.

    This is a plain base, not an ABC — the value is sharing the orchestrator, not enforcing
    a parallel-impl contract, so there are no abstract hooks beyond those override points.
    """

    def __init__(self, dataset_config, *, sample_rate: int):
        self.dataset_config = dataset_config
        self.sample_rate = sample_rate

    @property
    def cache_subdir(self) -> str:
        """Name of this extractor's cache subdir under the shared cache root. **Override point.**

        The subdir is the *coexistence axis*: caches that can exist simultaneously (what you
        A/B in experiments) get distinct subdirs. For extractors that wrap a named pretrained
        model, this MUST include the model name so swapping models doesn't silently reuse the
        wrong cache.
        """
        raise NotImplementedError

    def cache_params(self) -> dict:
        """The identifying constructor params for the cache manifest. **Override point.**

        Returns the params that define this extractor's output, EXCLUDING `dataset_config`
        (never read by `extract_one`) and any lazily-loaded model state. Written into
        `extractor.json` and strict-compared on reuse, so any param that changes the produced
        features belongs here. Explicit (not `self.__dict__`) so transient state can't leak in.
        """
        raise NotImplementedError

    def extract_one(self, audio_path: Path) -> list[torch.Tensor]:
        """Produce the feature tensor(s) for one audio file. **Override point.**

        Returns a *list* of feature tensors, one per segment. Non-windowing extractors
        return a single-element list (the N=1 case); a windowed extractor returns one tensor
        per window. The list is the contract that lets each extractor own its own windowing
        without the rest of the pipeline knowing the cardinality up front. See
        SpectrogramExtractor for the concrete spectrogram rationale.
        """
        del audio_path  # override point; signature documents the contract
        raise NotImplementedError

    def get_output_shape(self) -> tuple[int, ...]:
        """Shape of tensors this extractor produces (excluding batch dim). **Override point.**"""
        raise NotImplementedError

    def get_cached_feature_path(self, filename: str, output_dir: Path) -> Path:
        """Resolve the on-disk cache path for one file's feature tensor.

        The cached artifact is the audio file's stem with a .pt suffix (clip.flac ->
        clip.pt), placed under this extractor's `cache_subdir` within output_dir
        (output_dir is the shared cache root; the subdir keeps extractors from colliding).
        This is a *feature* concern — the cache artifact the extractor produces — which is
        why it lives here rather than on DatasetConfig (dataset identity). The previous
        per-dataset overrides only differed in which extension they stripped; Path.stem
        subsumes them all.
        """
        return Path(output_dir) / self.cache_subdir / (Path(filename).stem + ".pt")

    def ensure_cache_dir(self, output_dir: Path) -> Path:
        """Create-or-verify this extractor's cache subdir and its `extractor.json` manifest.

        Returns the subdir path (`output_dir/cache_subdir`). On a fresh subdir, creates it and
        writes a manifest of {class name, cache_params()}. On an existing subdir, STRICT-compares
        the stored manifest against the current one and raises ValueError on any difference —
        turning a silent train-on-stale-features bug (e.g. a changed n_mels reusing old .pt files)
        into a loud error. The escape hatch is rebuilding with `create_specs --clear`.

        Called ONCE per build/load by the orchestrators (create_specs, SpectrogramDataset), not
        per file — per-file directory creation is handled cheaply by process_one / the lazy path.
        """
        subdir = Path(output_dir) / self.cache_subdir
        manifest_path = subdir / "extractor.json"
        current = {"class": type(self).__name__, "params": self.cache_params()}

        if not subdir.exists():
            subdir.mkdir(parents=True, exist_ok=True)
            manifest_path.write_text(json.dumps(current, indent=2, sort_keys=True))
            return subdir

        if manifest_path.exists():
            stored = json.loads(manifest_path.read_text())
            if stored != current:
                raise ValueError(
                    f"Cache at {subdir} was built with different feature-extractor settings.\n"
                    f"  stored:  {stored}\n"
                    f"  current: {current}\n"
                    f"Rebuild with `create_specs --clear` to regenerate this extractor's cache."
                )
        else:
            # Subdir exists without a manifest (e.g. a pre-manifest cache): stamp one now.
            manifest_path.write_text(json.dumps(current, indent=2, sort_keys=True))
        return subdir

    def process_one(self, file_info: dict, output_dir: Path) -> tuple[bool, int | None]:
        """Build and cache one file's feature tensor: the offline (create_specs) build step.

        Applies all guards uniformly before extraction — resumable skip, audio existence,
        the dataset's known-bad files, and minimum file size — then runs extract_one and
        caches the result under get_cached_feature_path. Returns (success, feature_length):
        length is None for files skipped because they were already built. Per-file
        skips/failures are counted by the caller, not logged here, to keep the progress
        bar readable; only unexpected exceptions are logged.

        This orchestration is extractor-generic: the only extractor-specific step is the
        `extract_one` call, dispatched polymorphically to the concrete subclass.
        """
        config = self.dataset_config
        try:
            audio_path = file_info["audio_path"]
            filename = file_info["filename"]
            output_path = self.get_cached_feature_path(filename, output_dir)

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

            specs = self.extract_one(audio_path)
            # N=1 fallback: a single segment is cached under the plain {stem}.pt path,
            # byte-identical to the pre-Arc-B behavior. A windowing extractor (N>1) needs
            # multi-segment caching ({stem}__seg{i}.pt), which lands with segment
            # enumeration (extractor.num_segments, from audio metadata) in the
            # windowing/Perch arc. Until then the offline build is single-segment; refuse
            # rather than silently dropping windows.
            if len(specs) != 1:
                raise NotImplementedError(
                    f"Offline build caches one segment per file; this extractor produced "
                    f"{len(specs)}. Multi-segment (windowed) extractors aren't supported yet."
                )
            spec = specs[0]
            # Cheap, idempotent: ensures the cache subdir exists (mirrors the lazy path's
            # makedirs). Manifest validity is handled once-per-build by ensure_cache_dir.
            output_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save(spec, output_path)
            return True, spec.shape[-1]

        except Exception as e:
            logger.error(f"Error processing {file_info['filename']}: {e}")
            return False, None

    def _load_audio(self, audio_path: Path) -> torch.Tensor:
        """Load audio with torchcodec, resampling to the target rate and converting to mono."""
        decoder = AudioDecoder(str(audio_path), sample_rate=self.sample_rate, num_channels=1)
        return decoder.get_all_samples().data
