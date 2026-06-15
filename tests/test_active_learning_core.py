"""Smoke tests for run_binary_inference: the producer of every predictions CSV downstream.

The seam is the dataset config's enumeration surface (load_metadata / get_bad_files), stubbed
on a real FSD50KConfig instance. Everything else is the production path: extractor cache-path
resolution, the temp-CSV filter step, SpectrogramDataset, variable-length collate, the real
model forward pass, and the CSV writer whose schema the candidate-selection suite consumes.
"""

import csv
import logging

import pytest
import torch

from audioloop import active_learning_core
from audioloop.active_learning_core import load_training_set_filenames, run_binary_inference
from audioloop.config import AudioLoopConfig
from audioloop.datasets.fsd50k_config import FSD50KConfig
from audioloop.models.simplecnn import SimpleCnn

_CPU = torch.device("cpu")

# Core prediction columns, in the order run_binary_inference writes them (no ground truth).
EXPECTED_FIELDNAMES = [
    "filename",
    "prediction",
    "predicted_class",
    "target_class",
    "confidence",
    "entropy",
    "prob_negative",
    "prob_positive",
    "original_class",
    "audio_path",
    "filepath",
]


@pytest.fixture(autouse=True)
def cpu_device(monkeypatch):
    """Pin inference to CPU so results are identical on CUDA/MPS hosts."""
    monkeypatch.setattr(active_learning_core, "get_device", lambda: _CPU)


def _smoke_config(**overrides):
    defaults = {
        "experiment_name": "smoke-infer",
        "batch_size": 4,
        "num_workers": 0,  # no worker processes in tests
    }
    return AudioLoopConfig(**{**defaults, **overrides})


def _save_checkpoint(tmp_path, num_classes=2):
    """Save an (untrained) SimpleCnn in the production checkpoint format."""
    model = SimpleCnn(num_classes=num_classes)
    checkpoint = {
        "model_state_dict": model.state_dict(),
        "model_type": "simplecnn",
        "num_classes": num_classes,
    }
    path = tmp_path / "model_v1.pt"
    torch.save(checkpoint, path)
    return path


def _stub_dataset_enumeration(monkeypatch, metadata, bad_files: set[str] | None = None):
    """Point AudioLoopConfig.get_dataset_config at a real FSD50KConfig with stubbed enumeration.

    Only load_metadata and get_bad_files are replaced — the real get_binary_label (reads
    item["labels"]) and get_audio_path still run, so the filter loop is exercised as in
    production. The same instance is reused by get_feature_extractor, matching the
    run_binary_inference call pattern.
    """
    dataset_config = FSD50KConfig()
    resolved_bad_files = bad_files or set()

    def fake_load_metadata(split=None):
        del split
        return metadata

    def fake_get_dataset_config(self):
        del self
        return dataset_config

    monkeypatch.setattr(dataset_config, "load_metadata", fake_load_metadata)
    monkeypatch.setattr(dataset_config, "get_bad_files", lambda: resolved_bad_files)
    monkeypatch.setattr(AudioLoopConfig, "get_dataset_config", fake_get_dataset_config)
    return dataset_config


def _write_specs(config, filenames):
    """Write cached spec tensors with *varying* time lengths.

    Different lengths per file make inference batches ragged, so the test also covers the
    variable-length collate padding on the eval path (not just training).
    """
    extractor = config.get_feature_extractor()
    extractor.ensure_cache_dir(config.specs_dir)  # create the extractor's cache subdir
    for i, filename in enumerate(filenames):
        spec = torch.randn(1, extractor.n_mels, 30 + 5 * i)
        torch.save(spec, extractor.get_cached_feature_path(filename, config.specs_dir))


def _metadata_rows(n_files):
    """Metadata as load_metadata returns it: clips 2 and 3 carry the positive label."""
    return [
        {"filename": f"clip{i}.wav", "labels": ["target"] if i in (2, 3) else ["other"]}
        for i in range(n_files)
    ]


class TestLoadTrainingSetFilenames:
    def test_missing_or_none_path_returns_empty_set(self, tmp_path):
        assert load_training_set_filenames(None) == set()
        assert load_training_set_filenames(str(tmp_path / "nope.csv")) == set()

    def test_reads_filename_and_filepath_columns(self, tmp_path):
        """Both CSV dialects resolve to bare filenames (filepath gets basename'd)."""
        path = tmp_path / "training.csv"
        path.write_text("filepath,label\ndata/specs/a.pt,1\nb.pt,0\n")

        assert load_training_set_filenames(str(path)) == {"a.pt", "b.pt"}


class TestRunBinaryInference:
    def test_inference_filters_predicts_and_writes_csv(self, tmp_path, monkeypatch):
        """End to end: bad files and training-set files are excluded, every surviving file
        gets a coherent prediction row, and the CSV matches the documented schema."""
        config = _smoke_config()
        _stub_dataset_enumeration(monkeypatch, _metadata_rows(6), bad_files={"clip0.wav"})
        _write_specs(config, [f"clip{i}.wav" for i in range(6)])

        # clip1's *spec* filename in the training set must exclude it from inference.
        training_csv = tmp_path / "training_set.csv"
        training_csv.write_text("filename,label\nclip1.pt,1\n")

        model_path = _save_checkpoint(tmp_path)
        predictions_csv = tmp_path / "predictions.csv"

        results = run_binary_inference(
            config=config,
            model_path=str(model_path),
            predictions_csv=str(predictions_csv),
            positive_class_name="target",
            negative_class_name="other",
            training_set_csv=str(training_csv),
        )

        # 6 files - 1 bad - 1 already trained on = 4 predictions.
        assert {r["filename"] for r in results} == {"clip2.pt", "clip3.pt", "clip4.pt", "clip5.pt"}

        for r in results:
            assert r["prob_negative"] + r["prob_positive"] == pytest.approx(1.0, abs=1e-5)
            assert r["confidence"] == pytest.approx(max(r["prob_negative"], r["prob_positive"]))
            assert r["entropy"] >= 0.0
            assert r["prediction"] == (r["prob_positive"] > r["prob_negative"])
            assert r["predicted_class"] == ("target" if r["prediction"] else "other")
            assert r["target_class"] == "target"
            # getattr() on the metadata dict never finds classID, so this is always -1.
            # Pinned as current behavior: if original_class ever starts propagating, this
            # test should be updated alongside the consumers of that column.
            assert r["original_class"] == -1
            assert "ground_truth" not in r  # with_ground_truth defaults to False

        with predictions_csv.open() as f:
            rows = list(csv.DictReader(f))
        assert len(rows) == 4
        assert list(rows[0].keys()) == EXPECTED_FIELDNAMES
        for row in rows:
            assert 0.0 <= float(row["prob_positive"]) <= 1.0
            assert 0.0 <= float(row["prob_negative"]) <= 1.0
            assert 0.0 <= float(row["confidence"]) <= 1.0

    def test_ground_truth_mode_adds_columns_from_binary_labels(self, tmp_path, monkeypatch):
        """with_ground_truth=True adds ground_truth/correct, derived from the dataset
        config's real get_binary_label (positive_class_name in item['labels'])."""
        config = _smoke_config(with_ground_truth=True)
        _stub_dataset_enumeration(monkeypatch, _metadata_rows(4))
        _write_specs(config, [f"clip{i}.wav" for i in range(4)])

        model_path = _save_checkpoint(tmp_path)

        # No predictions_csv: exercises the default config.output_dir/"predictions.csv" path.
        results = run_binary_inference(
            config=config,
            model_path=str(model_path),
            positive_class_name="target",
            negative_class_name="other",
        )

        assert len(results) == 4
        by_name = {r["filename"]: r for r in results}
        for name in ("clip2.pt", "clip3.pt"):
            assert by_name[name]["ground_truth"] is True
        for name in ("clip0.pt", "clip1.pt"):
            assert by_name[name]["ground_truth"] is False
        for r in results:
            assert r["correct"] == (r["prediction"] == r["ground_truth"])

        default_csv = config.output_dir / "predictions.csv"
        assert default_csv.exists()
        with default_csv.open() as f:
            header = next(csv.reader(f))
        assert "ground_truth" in header
        assert "correct" in header
        assert set(header) == set(EXPECTED_FIELDNAMES) | {"ground_truth", "correct"}

    def test_missing_model_raises(self, tmp_path, monkeypatch):
        """A missing checkpoint is a loud FileNotFoundError, not an empty result set."""
        config = _smoke_config()
        _stub_dataset_enumeration(monkeypatch, _metadata_rows(2))
        _write_specs(config, ["clip0.wav", "clip1.wav"])

        with pytest.raises(FileNotFoundError, match="Model file not found"):
            run_binary_inference(
                config=config,
                model_path=str(tmp_path / "missing.pt"),
                predictions_csv=str(tmp_path / "predictions.csv"),
                positive_class_name="target",
                negative_class_name="other",
            )


class TestTrainThenInfer:
    def test_training_checkpoint_feeds_inference(self, tmp_path, monkeypatch):
        """The two halves of an active-learning cycle compose: run_training's checkpoint is
        consumed by run_binary_inference without re-saving or format shims."""
        from audioloop import training_core
        from audioloop.training_core import run_training

        monkeypatch.setattr(training_core, "get_device", lambda: _CPU)

        config = _smoke_config(
            model_type="simplecnn", max_epochs=1, use_lr_scheduler=False
        )
        _stub_dataset_enumeration(monkeypatch, _metadata_rows(4))
        filenames = [f"clip{i}.wav" for i in range(4)]
        _write_specs(config, filenames)

        # Train on the same cached specs (alternating labels for two classes).
        training_csv = tmp_path / "training_set.csv"
        with training_csv.open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["filename", "label"])
            writer.writeheader()
            writer.writerows({"filename": n, "label": i % 2} for i, n in enumerate(filenames))

        run_training(config, str(training_csv), version=1, log_level=logging.WARNING)

        results = run_binary_inference(
            config=config,
            model_path=str(config.get_model_path(1)),
            predictions_csv=str(tmp_path / "predictions.csv"),
            positive_class_name="target",
            negative_class_name="other",
        )

        assert len(results) == 4
        assert all(0.0 <= r["prob_positive"] <= 1.0 for r in results)
