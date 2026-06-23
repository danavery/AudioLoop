"""
Tests for training core functionality.

This module tests class weighting, the real training loop (execute_training_loop with
scripted epoch results), and run_training end to end on tiny synthetic spectrograms.
"""

import csv
import logging
from typing import cast

import pytest
import torch
from torch.utils.data import DataLoader

from audioloop import training_core
from audioloop.active_learning_core import load_model
from audioloop.config import AudioLoopConfig
from audioloop.feature_extractor import EmbeddingExtractor
from audioloop.models.linearprobe import LinearProbe
from audioloop.models.simplecnn import SimpleCnn
from audioloop.training_core import execute_training_loop, run_training, setup_loss_criterion
from audioloop.utils.cached_feature_dataset import CachedFeatureDataset
from audioloop.utils.stopping_criteria import AccuracyCriterion, PlateauCriterion


class _FakeTrainDataset:
    """Minimal stand-in for CachedFeatureDataset.

    setup_loss_criterion only reads `.samples` (for the per-sample labels) and `len(...)`
    (for the total), so a list of {"label": ...} dicts is a faithful substitute that keeps
    the test off the disk/audio path.
    """

    def __init__(self, labels):
        self.samples = [{"label": label} for label in labels]

    def __len__(self):
        return len(self.samples)


_CPU = torch.device("cpu")


def _criterion_for(config: AudioLoopConfig, labels: list[int]) -> torch.nn.CrossEntropyLoss:
    """Build the loss criterion for a config + label distribution via the real factory.

    Casts the lightweight fake to CachedFeatureDataset (setup_loss_criterion only touches the
    duck-typed surface) and narrows the nn.Module return to CrossEntropyLoss so callers can
    read `.weight` directly — and so the test pins the factory's return contract.
    """
    dataset = cast(CachedFeatureDataset, _FakeTrainDataset(labels))
    criterion = setup_loss_criterion(config, dataset, _CPU)
    assert isinstance(criterion, torch.nn.CrossEntropyLoss)
    return criterion


class TestClassWeightingConfig:
    """Test class weighting configuration."""

    def test_class_weighting_default(self):
        """Test that class weighting defaults to 0.70."""
        config = AudioLoopConfig()
        assert config.class_weighting == 0.70

    def test_class_weighting_adaptive_mode(self):
        """Test that adaptive class weighting can be enabled."""
        config = AudioLoopConfig(class_weighting="adaptive")
        assert config.class_weighting == "adaptive"

    def test_class_weighting_fixed_mode(self):
        """Test that fixed class weighting can be configured."""
        config = AudioLoopConfig(class_weighting=0.25)
        assert config.class_weighting == 0.25

    def test_class_weighting_validation_rejects_invalid_string(self):
        """Test that invalid string values are rejected."""
        with pytest.raises(ValueError, match="class_weighting must be 'adaptive'"):
            AudioLoopConfig(class_weighting="invalid")

    def test_class_weighting_validation_rejects_out_of_range_float(self):
        """Test that out-of-range float values are rejected."""
        with pytest.raises(ValueError, match=r"between 0\.0 and 1\.0"):
            AudioLoopConfig(class_weighting=1.5)


class TestClassWeightingCalculation:
    """Test the loss criterion built by setup_loss_criterion (the real production path).

    These exercise setup_loss_criterion end-to-end and assert on the weight tensor it
    attaches to the returned CrossEntropyLoss, rather than re-deriving the weighting formula
    in the test (which would pass even if training_core's formula were wrong).
    """

    def test_adaptive_balanced_dataset_gives_equal_weights(self):
        """Adaptive weighting on a 50/50 split yields equal class weights."""
        criterion = _criterion_for(AudioLoopConfig(class_weighting="adaptive"), [0] * 50 + [1] * 50)

        assert criterion.weight is not None
        assert torch.allclose(criterion.weight, torch.tensor([1.0, 1.0]), atol=1e-6)

    def test_adaptive_imbalanced_dataset_upweights_minority(self):
        """Adaptive weighting on a 90/10 split upweights the minority class ~9x."""
        criterion = _criterion_for(AudioLoopConfig(class_weighting="adaptive"), [0] * 90 + [1] * 10)

        # total / (num_classes * count): [100/(2*90), 100/(2*10)] = [0.5556, 5.0]
        expected = torch.tensor([100.0 / (2 * 90), 100.0 / (2 * 10)])
        assert criterion.weight is not None
        assert torch.allclose(criterion.weight, expected, atol=1e-6)
        assert criterion.weight[1] > criterion.weight[0]
        assert abs(criterion.weight[1] / criterion.weight[0] - 9.0) < 0.1

    def test_adaptive_extreme_imbalance(self):
        """Adaptive weighting on a 999/1 split produces a ~999x weight ratio."""
        criterion = _criterion_for(AudioLoopConfig(class_weighting="adaptive"), [0] * 999 + [1] * 1)

        assert criterion.weight is not None
        assert criterion.weight[1] > criterion.weight[0]
        assert abs(criterion.weight[1] / criterion.weight[0] - 999.0) < 1.0

    def test_fixed_float_weighting_targets_positive_ratio(self):
        """A fixed float weighting sets pos weight = (1 - w) / w with neg weight 1.0."""
        criterion = _criterion_for(AudioLoopConfig(class_weighting=0.25), [0] * 90 + [1] * 10)

        # weight_neg=1.0, weight_pos=(1-0.25)/0.25=3.0; independent of the actual counts.
        assert criterion.weight is not None
        assert torch.allclose(criterion.weight, torch.tensor([1.0, 3.0]), atol=1e-6)

    def test_none_weighting_produces_unweighted_loss(self):
        """class_weighting=None yields an unweighted CrossEntropyLoss (weight is None)."""
        criterion = _criterion_for(AudioLoopConfig(class_weighting=None), [0] * 90 + [1] * 10)

        assert criterion.weight is None

    def test_weighted_criterion_is_usable(self):
        """The criterion setup_loss_criterion returns produces a finite, non-negative loss."""
        criterion = _criterion_for(AudioLoopConfig(class_weighting="adaptive"), [0] * 90 + [1] * 10)

        dummy_logits = torch.randn(10, 2)  # batch_size=10, num_classes=2
        dummy_labels = torch.randint(0, 2, (10,))
        loss = criterion(dummy_logits, dummy_labels)

        assert torch.isfinite(loss)
        assert loss.item() >= 0


def _run_scripted_loop(monkeypatch, epoch_results, stopping_criterion, max_epochs):
    """Run the REAL execute_training_loop with train_epoch replaced by a scripted sequence.

    Only the gradient work (train_epoch) is stubbed; the model, optimizer, criterion under
    test, and all loop bookkeeping (best-model updates, break vs for-else completion) are
    the production code. This is the loop-side counterpart of the criterion-only suites in
    test_stopping_criteria.py: those prove the criteria, these prove the loop honors them.

    The script is an iterator on purpose: if the loop runs more epochs than scripted, the
    stub raises StopIteration and the test fails loudly instead of looping on stale values.
    """
    results = iter(epoch_results)

    def scripted_train_epoch(*args: object, **kwargs: object):
        del args, kwargs  # train_epoch's signature, but the script ignores it
        return next(results)

    monkeypatch.setattr(training_core, "train_epoch", scripted_train_epoch)

    model = SimpleCnn(num_classes=2)
    stopping_criterion.reset()

    return model, execute_training_loop(
        model=model,
        train_loader=cast(DataLoader, None),  # only consumed by the stubbed train_epoch
        optimizer=torch.optim.Adam(model.parameters()),
        criterion=torch.nn.CrossEntropyLoss(),
        device=_CPU,
        stopping_criterion=stopping_criterion,
        scheduler=None,
        config=AudioLoopConfig(max_epochs=max_epochs),
        train_dataset=cast(CachedFeatureDataset, None),  # floor is explicit, so never read
    )


class TestExecuteTrainingLoop:
    """Test the real training loop's integration with stopping criteria.

    Criteria are built with an explicit accuracy_floor: the loop always forwards
    train_dataset to should_stop, and a None floor would trigger auto-calculation from
    label balance, coupling stop timing to fixture data instead of the scripted losses.
    """

    def test_plateau_early_stop_breaks_loop_and_tracks_best(self, monkeypatch):
        """The loop stops when plateau patience runs out, keeping the best-epoch snapshot."""
        # (avg_loss, accuracy) per epoch: improves for 3 epochs, then degrades.
        script = [(1.0, 0.5), (0.8, 0.6), (0.7, 0.7), (0.75, 0.7), (0.8, 0.7)]
        criterion = PlateauCriterion(patience=2, min_delta=0.01, accuracy_floor=0.0)

        model, (final_accuracy, best_accuracy, num_epochs) = _run_scripted_loop(
            monkeypatch, script, criterion, max_epochs=10
        )

        assert num_epochs == 5  # stopped early: patience (2) exhausted at epoch 5 of 10
        assert final_accuracy == 0.7
        assert best_accuracy == 0.7  # accuracy at the last loss improvement (epoch 3)

        # The loop must have pushed a real state-dict snapshot into the criterion.
        best_state = criterion.get_best_model_state()
        assert best_state is not None
        assert best_state.keys() == model.state_dict().keys()

    def test_natural_completion_runs_all_epochs(self, monkeypatch):
        """With loss improving every epoch the loop completes max_epochs (for-else path)."""
        script = [(1.0, 0.5), (0.9, 0.6), (0.8, 0.7)]
        criterion = PlateauCriterion(patience=50, accuracy_floor=0.0)

        _, (final_accuracy, best_accuracy, num_epochs) = _run_scripted_loop(
            monkeypatch, script, criterion, max_epochs=3
        )

        assert num_epochs == 3
        assert final_accuracy == 0.7
        assert best_accuracy == 0.7  # best updated every epoch; last improvement wins

    def test_no_best_model_when_criterion_never_saves(self, monkeypatch):
        """AccuracyCriterion without perfect accuracy yields best_accuracy=None.

        This pins the fallback contract save_trained_model relies on: when no best state
        was tracked, callers get the final-epoch accuracy and the final model state.
        """
        script = [(1.0, 0.5), (0.9, 0.6), (0.8, 0.7)]
        criterion = AccuracyCriterion(max_epochs=3)

        _, (final_accuracy, best_accuracy, num_epochs) = _run_scripted_loop(
            monkeypatch, script, criterion, max_epochs=3
        )

        assert num_epochs == 3
        assert final_accuracy == 0.7
        assert best_accuracy is None
        assert criterion.get_best_model_state() is None


def _build_training_set(config, n_files=8, time_frames=40):
    """Materialize a tiny pre-built training set in the fake project root.

    Writes random (1, n_mels, T) tensors at the exact cache paths the extractor resolves
    (so CachedFeatureDataset takes the cached-load path, no audio decoding) plus the labels
    CSV. Alternating labels guarantee both classes are present for num_classes detection.
    """
    extractor = config.get_feature_extractor()
    extractor.ensure_cache_dir(config.feature_cache_dir)  # create the extractor's cache subdir
    rows = []
    for i in range(n_files):
        filename = f"clip{i}.wav"
        spec = torch.randn(1, extractor.n_mels, time_frames)
        torch.save(spec, extractor.get_cached_feature_path(filename, config.feature_cache_dir))
        rows.append({"filename": filename, "label": i % 2})

    csv_path = config.feature_cache_dir.parent / "training_set.csv"
    with csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["filename", "label"])
        writer.writeheader()
        writer.writerows(rows)
    return csv_path


class TestRunTrainingEndToEnd:
    """Smoke tests through the real run_training: dataset -> loop -> checkpoint -> reload.

    get_device is pinned to CPU so the test is deterministic and identical on CUDA/MPS
    hosts; everything else (DataLoader, collate, model, criterion factory, stopping
    criterion, checkpoint save) is the production path.
    """

    @pytest.fixture(autouse=True)
    def cpu_device(self, monkeypatch):
        monkeypatch.setattr(training_core, "get_device", lambda: _CPU)

    def _smoke_config(self, **overrides):
        defaults = {
            "experiment_name": "smoke",
            "model_type": "simplecnn",
            "max_epochs": 3,
            "batch_size": 4,
            "num_workers": 0,  # no worker processes in tests
            "use_lr_scheduler": False,
        }
        return AudioLoopConfig(**{**defaults, **overrides})

    def test_trains_saves_and_reloads_checkpoint(self):
        """run_training completes, saves a checkpoint at the config path, and the
        checkpoint round-trips through load_model into a usable classifier."""
        config = self._smoke_config()
        csv_path = _build_training_set(config)

        accuracy, num_epochs = run_training(
            config, str(csv_path), version=1, log_level=logging.WARNING
        )

        assert 0.0 <= accuracy <= 1.0
        assert 1 <= num_epochs <= config.max_epochs

        model_path = config.get_model_path(1)
        assert model_path.exists()

        loaded = load_model(model_path, _CPU)
        assert isinstance(loaded, SimpleCnn)
        assert loaded.get_model_info()["num_classes"] == 2

        logits = loaded(torch.randn(2, 1, 128, 40))
        assert logits.shape == (2, 2)
        assert torch.isfinite(logits).all()

    def test_trains_linear_probe_over_embeddings(self, monkeypatch):
        """End-to-end: embedding extractor (768,) features + linear probe, no backbone load.

        Pre-cached (768,) tensors and a static get_output_shape mean extract_one (the wav2vec2
        forward) is never invoked — so the whole embedding->probe path trains without
        downloading the backbone. _get_model is patched to fail loudly if that assumption breaks.
        """

        def _no_backbone(self):
            raise AssertionError("backbone must not load: cached features + static shape")

        monkeypatch.setattr(EmbeddingExtractor, "_get_model", _no_backbone)

        config = self._smoke_config(feature_extractor_type="embedding", model_type="linearprobe")
        extractor = config.get_feature_extractor()
        extractor.ensure_cache_dir(config.feature_cache_dir)
        rows = []
        for i in range(8):
            filename = f"clip{i}.wav"
            torch.save(
                torch.randn(768),
                extractor.get_cached_feature_path(filename, config.feature_cache_dir),
            )
            rows.append({"filename": filename, "label": i % 2})
        csv_path = config.feature_cache_dir.parent / "embed_training_set.csv"
        with csv_path.open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["filename", "label"])
            writer.writeheader()
            writer.writerows(rows)

        accuracy, num_epochs = run_training(
            config, str(csv_path), version=1, log_level=logging.WARNING
        )

        assert 0.0 <= accuracy <= 1.0
        assert 1 <= num_epochs <= config.max_epochs

        loaded = load_model(config.get_model_path(1), _CPU)
        assert isinstance(loaded, LinearProbe)
        assert loaded.get_model_info()["in_features"] == 768  # rebuilt from the checkpoint
        logits = loaded(torch.randn(2, 768))
        assert logits.shape == (2, 2)
        assert torch.isfinite(logits).all()

    def test_custom_model_path_overrides_config_path(self, tmp_path):
        """An explicit model_path wins over config.get_model_path(version)."""
        config = self._smoke_config(max_epochs=1)
        csv_path = _build_training_set(config, n_files=4)
        custom_path = tmp_path / "outputs" / "smoke" / "custom_model.pt"
        custom_path.parent.mkdir(parents=True, exist_ok=True)

        run_training(
            config, str(csv_path), version=1, model_path=str(custom_path), log_level=logging.WARNING
        )

        assert custom_path.exists()
        assert not config.get_model_path(1).exists()
        assert isinstance(load_model(custom_path, _CPU), SimpleCnn)
