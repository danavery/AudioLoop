"""
Tests for audioloop.utils.candidate_selection module.

This module contains comprehensive tests for the candidate selection strategies
used in the AudioLoop active learning framework, including the simplified
BasicTransitionStrategy design.
"""

import logging
from unittest.mock import Mock, patch

import pytest

from audioloop.utils.candidate_selection import (
    BasicTransitionStrategy,
    CandidateSelectionStrategy,
    ConfidenceStrategy,
    EntropyStrategy,
    create_strategy,
    load_predictions,
    save_candidates,
)
from audioloop.utils.candidate_selection.mixed_entropy import MixedEntropyStrategy


def get_default_thresholds():
    """Get the actual default thresholds from BasicTransitionStrategy implementation."""
    # Create a strategy to extract the actual defaults
    strategy = BasicTransitionStrategy()
    return strategy.f1_threshold, strategy.confidence_threshold, strategy.variance_threshold


# Test fixtures
@pytest.fixture
def sample_predictions():
    """Create sample predictions for testing."""
    return [
        {
            "filename": "file1.pt",
            "prediction": 1,
            "predicted_class": "positive",
            "confidence": 0.95,
            "entropy": 0.1,
            "filepath": "path/file1.pt",
        },
        {
            "filename": "file2.pt",
            "prediction": 1,
            "predicted_class": "positive",
            "confidence": 0.85,
            "entropy": 0.3,
            "filepath": "path/file2.pt",
        },
        {
            "filename": "file3.pt",
            "prediction": 0,
            "predicted_class": "negative",
            "confidence": 0.92,
            "entropy": 0.2,
            "filepath": "path/file3.pt",
        },
        {
            "filename": "file4.pt",
            "prediction": 0,
            "predicted_class": "negative",
            "confidence": 0.78,
            "entropy": 0.4,
            "filepath": "path/file4.pt",
        },
        {
            "filename": "file5.pt",
            "prediction": 1,
            "predicted_class": "positive",
            "confidence": 0.65,
            "entropy": 0.6,
            "filepath": "path/file5.pt",
        },
        {
            "filename": "file6.pt",
            "prediction": 0,
            "predicted_class": "negative",
            "confidence": 0.55,
            "entropy": 0.8,
            "filepath": "path/file6.pt",
        },
    ]


@pytest.fixture
def empty_predictions():
    """Create empty predictions list for edge case testing."""
    return []


@pytest.fixture
def single_prediction():
    """Create single prediction for edge case testing."""
    return [
        {
            "filename": "single.pt",
            "prediction": 1,
            "predicted_class": "positive",
            "confidence": 0.9,
            "entropy": 0.2,
            "filepath": "path/single.pt",
        }
    ]


@pytest.fixture
def imbalanced_predictions():
    """Create imbalanced predictions (many negative, few positive)."""
    predictions = []

    # Add 2 positive predictions
    for i in range(2):
        predictions.append(
            {
                "filename": f"pos_{i}.pt",
                "prediction": 1,
                "predicted_class": "positive",
                "confidence": 0.8 + i * 0.1,
                "entropy": 0.2 + i * 0.1,
                "filepath": f"path/pos_{i}.pt",
            }
        )

    # Add 10 negative predictions
    for i in range(10):
        predictions.append(
            {
                "filename": f"neg_{i}.pt",
                "prediction": 0,
                "predicted_class": "negative",
                "confidence": 0.6 + i * 0.03,
                "entropy": 0.3 + i * 0.05,
                "filepath": f"path/neg_{i}.pt",
            }
        )

    return predictions


@pytest.fixture
def predictions_with_ground_truth():
    """Create predictions with ground truth labels for metrics calculation."""
    return [
        {
            "filename": "file1.pt",
            "prediction": 1,
            "predicted_class": "positive",
            "confidence": 0.95,
            "entropy": 0.1,
            "ground_truth": True,
            "predicted_is_positive": True,
            "filepath": "path/file1.pt",
        },
        {
            "filename": "file2.pt",
            "prediction": 1,
            "predicted_class": "positive",
            "confidence": 0.85,
            "entropy": 0.3,
            "ground_truth": True,
            "predicted_is_positive": True,
            "filepath": "path/file2.pt",
        },
        {
            "filename": "file3.pt",
            "prediction": 0,
            "predicted_class": "negative",
            "confidence": 0.92,
            "entropy": 0.2,
            "ground_truth": False,
            "predicted_is_positive": False,
            "filepath": "path/file3.pt",
        },
        {
            "filename": "file4.pt",
            "prediction": 0,
            "predicted_class": "negative",
            "confidence": 0.78,
            "entropy": 0.4,
            "ground_truth": False,
            "predicted_is_positive": False,
            "filepath": "path/file4.pt",
        },
    ]


# Base class tests
def test_candidate_selection_strategy_is_abstract():
    """Test that CandidateSelectionStrategy cannot be instantiated directly."""
    with pytest.raises(TypeError):
        CandidateSelectionStrategy()  # type: ignore


def test_candidate_selection_strategy_get_name():
    """Test that get_name returns class name."""

    class TestStrategy(CandidateSelectionStrategy):
        def select_candidates(
            self,
            predictions,
            num_candidates,
            positive_class_name="positive",
            negative_class_name="negative",
            **kwargs,
        ):
            return []

    strategy = TestStrategy()
    assert strategy.get_name() == "TestStrategy"


# ConfidenceStrategy tests
def test_confidence_strategy_basic_selection(sample_predictions):
    """Test basic candidate selection with confidence strategy."""
    strategy = ConfidenceStrategy()

    # Request 4 candidates with 50% positive
    candidates = strategy.select_candidates(
        sample_predictions, num_candidates=4, positive_percentage=0.5, random_seed=42
    )

    assert len(candidates) == 4

    # Check that we have roughly equal positive/negative
    positive_count = sum(1 for c in candidates if c["predicted_class"] == "positive")
    negative_count = sum(1 for c in candidates if c["predicted_class"] == "negative")

    assert positive_count == 2
    assert negative_count == 2


def test_confidence_strategy_sorts_by_confidence(sample_predictions):
    """Test that confidence strategy prioritizes high-confidence predictions."""
    strategy = ConfidenceStrategy()

    # Use multiplier of 1 to get top candidates directly
    candidates = strategy.select_candidates(
        sample_predictions,
        num_candidates=4,
        positive_percentage=0.5,
        candidate_pool_multiplier=1,
        random_seed=42,
    )

    # Separate by prediction type
    positive_candidates = [c for c in candidates if c["predicted_class"] == "positive"]
    negative_candidates = [c for c in candidates if c["predicted_class"] == "negative"]

    # Should get highest confidence positive and negative
    assert len(positive_candidates) == 2
    assert len(negative_candidates) == 2

    # Check that we got the highest confidence samples
    # From sample_predictions: positive confidences are [0.95, 0.85, 0.65]
    # From sample_predictions: negative confidences are [0.92, 0.78, 0.55]
    positive_confidences = [c["confidence"] for c in positive_candidates]
    negative_confidences = [c["confidence"] for c in negative_candidates]

    assert 0.95 in positive_confidences  # Highest positive confidence
    assert 0.85 in positive_confidences  # Second highest positive confidence
    assert 0.92 in negative_confidences  # Highest negative confidence
    assert 0.78 in negative_confidences  # Second highest negative confidence


def test_confidence_strategy_percentage_split(sample_predictions):
    """Test different positive percentages."""
    strategy = ConfidenceStrategy()

    # Test 80% positive
    candidates = strategy.select_candidates(
        sample_predictions, num_candidates=5, positive_percentage=0.8, random_seed=42
    )

    positive_count = sum(1 for c in candidates if c["predicted_class"] == "positive")
    # Should get 3 positives (all available) since we only have 3 positive samples
    assert positive_count == 3

    # Test 20% positive
    candidates = strategy.select_candidates(
        sample_predictions, num_candidates=5, positive_percentage=0.2, random_seed=42
    )

    positive_count = sum(1 for c in candidates if c["predicted_class"] == "positive")
    assert positive_count == 1  # 20% of 5 = 1


def test_confidence_strategy_empty_predictions(empty_predictions):
    """Test confidence strategy with empty predictions."""
    strategy = ConfidenceStrategy()

    candidates = strategy.select_candidates(empty_predictions, num_candidates=5, random_seed=42)

    assert len(candidates) == 0


def test_confidence_strategy_insufficient_candidates(single_prediction):
    """Test confidence strategy when there aren't enough candidates."""
    strategy = ConfidenceStrategy()

    candidates = strategy.select_candidates(single_prediction, num_candidates=5, random_seed=42)

    # Should return what's available
    assert len(candidates) == 1
    assert candidates[0]["filename"] == "single.pt"


def test_confidence_strategy_imbalanced_data(imbalanced_predictions):
    """Test confidence strategy with imbalanced data."""
    strategy = ConfidenceStrategy()

    # Request more positives than available
    candidates = strategy.select_candidates(
        imbalanced_predictions,
        num_candidates=6,
        positive_percentage=0.8,  # Would want 4-5 positives, but only 2 available
        random_seed=42,
    )

    positive_count = sum(1 for c in candidates if c["predicted_class"] == "positive")
    negative_count = sum(1 for c in candidates if c["predicted_class"] == "negative")

    # Should get all 2 positives and 2 negatives (total 4 candidates)
    assert positive_count == 2
    assert negative_count == 2
    assert len(candidates) == 4


def test_confidence_strategy_get_name():
    """Test confidence strategy name."""
    strategy = ConfidenceStrategy()
    assert strategy.get_name() == "Confidence-Based"


# EntropyStrategy tests
def test_entropy_strategy_basic_selection(sample_predictions):
    """Test basic candidate selection with entropy strategy."""
    strategy = EntropyStrategy()

    candidates = strategy.select_candidates(
        sample_predictions, num_candidates=4, positive_percentage=0.5, random_seed=42
    )

    assert len(candidates) == 4

    positive_count = sum(1 for c in candidates if c["predicted_class"] == "positive")
    negative_count = sum(1 for c in candidates if c["predicted_class"] == "negative")

    assert positive_count == 2
    assert negative_count == 2


def test_entropy_strategy_sorts_by_entropy(sample_predictions):
    """Test that entropy strategy prioritizes high-entropy predictions."""
    strategy = EntropyStrategy()

    # Use multiplier of 1 to get top candidates directly
    candidates = strategy.select_candidates(
        sample_predictions,
        num_candidates=4,
        positive_percentage=0.5,
        candidate_pool_multiplier=1,
        random_seed=42,
    )

    # Separate by prediction type
    positive_candidates = [c for c in candidates if c["predicted_class"] == "positive"]
    negative_candidates = [c for c in candidates if c["predicted_class"] == "negative"]

    # Should get highest entropy samples
    # From sample_predictions: positive entropies are [0.1, 0.3, 0.6]
    # From sample_predictions: negative entropies are [0.2, 0.4, 0.8]
    positive_entropies = [c["entropy"] for c in positive_candidates]
    negative_entropies = [c["entropy"] for c in negative_candidates]

    assert 0.6 in positive_entropies  # Highest positive entropy
    assert 0.3 in positive_entropies  # Second highest positive entropy
    assert 0.8 in negative_entropies  # Highest negative entropy
    assert 0.4 in negative_entropies  # Second highest negative entropy


def test_entropy_strategy_get_name():
    """Test entropy strategy name."""
    strategy = EntropyStrategy()
    assert strategy.get_name() == "Entropy-Based (Uncertainty Sampling)"


# BasicTransitionStrategy tests (simplified design)
def test_basic_transition_with_hard_coded_defaults():
    """Test BasicTransitionStrategy uses hard-coded defaults when no thresholds provided."""
    expected_f1, expected_conf, expected_var = get_default_thresholds()
    strategy = BasicTransitionStrategy()

    # Should use implementation defaults
    assert strategy.f1_threshold == expected_f1
    assert strategy.confidence_threshold == expected_conf
    assert strategy.variance_threshold == expected_var

    # Test that thresholds are reasonable (behavioral testing)
    assert strategy.f1_threshold is not None and 0.1 <= strategy.f1_threshold <= 0.5
    assert (
        strategy.confidence_threshold is not None and 0.7 <= strategy.confidence_threshold <= 0.95
    )
    assert strategy.variance_threshold is not None and 0.05 <= strategy.variance_threshold <= 0.25


def test_basic_transition_with_explicit_thresholds():
    """Test BasicTransitionStrategy uses provided thresholds when all are specified."""
    strategy = BasicTransitionStrategy(
        f1_threshold=0.3, confidence_threshold=0.95, variance_threshold=0.1
    )

    # Should use provided values
    assert strategy.f1_threshold == 0.3
    assert strategy.confidence_threshold == 0.95
    assert strategy.variance_threshold == 0.1


@patch("audioloop.utils.adaptive_thresholds.threshold_calculator")
def test_basic_transition_with_auto_thresholds(mock_threshold_calc):
    """Test BasicTransitionStrategy uses auto-calculated thresholds when requested."""
    # Mock the threshold calculator
    mock_threshold_calc.calculate_thresholds.return_value = (0.25, 0.92, 0.08)

    strategy = BasicTransitionStrategy(auto_thresholds=True, estimated_positive_pct=0.1)

    # Should use auto-calculated values
    assert strategy.f1_threshold == 0.25
    assert strategy.confidence_threshold == 0.92
    assert strategy.variance_threshold == 0.08

    # Verify the threshold calculator was called correctly
    mock_threshold_calc.calculate_thresholds.assert_called_once_with(0.1, None)
    mock_threshold_calc.print_analysis.assert_called_once()


def test_basic_transition_no_partial_overrides():
    """Test that partial threshold specification raises ValueError."""
    # Providing only some thresholds should raise an error
    with pytest.raises(ValueError, match="Must specify all three thresholds"):
        BasicTransitionStrategy(f1_threshold=0.4)

    with pytest.raises(ValueError, match="Must specify all three thresholds"):
        BasicTransitionStrategy(confidence_threshold=0.95)

    with pytest.raises(ValueError, match="Must specify all three thresholds"):
        BasicTransitionStrategy(variance_threshold=0.08)

    # Providing two thresholds should also raise an error
    with pytest.raises(ValueError, match="Must specify all three thresholds"):
        BasicTransitionStrategy(f1_threshold=0.4, confidence_threshold=0.95)


def test_basic_transition_auto_thresholds_with_manual_conflict():
    """Test that auto_thresholds=True with manual thresholds raises ValueError."""
    with pytest.raises(
        ValueError, match="Cannot specify both auto_thresholds=True and manual thresholds"
    ):
        BasicTransitionStrategy(auto_thresholds=True, f1_threshold=0.4)

    with pytest.raises(
        ValueError, match="Cannot specify both auto_thresholds=True and manual thresholds"
    ):
        BasicTransitionStrategy(
            auto_thresholds=True,
            f1_threshold=0.4,
            confidence_threshold=0.95,
            variance_threshold=0.08,
        )


def test_basic_transition_behavior_with_defaults():
    """Test transition logic with default thresholds."""
    expected_f1, expected_conf, expected_var = get_default_thresholds()
    strategy = BasicTransitionStrategy()

    # Metrics that should trigger transition with defaults
    good_metrics = {
        "f1_score": (expected_f1 or 0.2) + 0.1,  # Above threshold
        "mean_confidence": (expected_conf or 0.9) + 0.05,  # Above threshold
        "std_confidence": (expected_var or 0.12) - 0.04,  # Below threshold
    }

    # Metrics that should not trigger transition
    poor_metrics = {
        "f1_score": (expected_f1 or 0.2) - 0.1,  # Below threshold
        "mean_confidence": (expected_conf or 0.9) - 0.05,  # Below threshold
        "std_confidence": (expected_var or 0.12) + 0.03,  # Above threshold
    }

    assert strategy._should_transition(good_metrics)
    assert not strategy._should_transition(poor_metrics)


def test_basic_transition_behavior_with_explicit_thresholds():
    """Test transition logic with explicit thresholds."""
    strategy = BasicTransitionStrategy(
        f1_threshold=0.4, confidence_threshold=0.95, variance_threshold=0.1
    )

    # Same metrics as before, but different thresholds
    metrics = {
        "f1_score": 0.3,  # < 0.4 explicit threshold
        "mean_confidence": 0.95,  # = 0.95 explicit threshold
        "std_confidence": 0.08,  # < 0.1 explicit threshold
    }

    # Should not transition because F1 is below the higher threshold
    assert not strategy._should_transition(metrics)


def test_basic_transition_empty_metrics():
    """Test that transition doesn't happen with empty metrics."""
    strategy = BasicTransitionStrategy()

    assert not strategy._should_transition({})
    # Test with empty dictionary instead of None
    assert not strategy._should_transition({})


def test_basic_transition_strategy_delegation(sample_predictions):
    """Test that BasicTransitionStrategy properly delegates to underlying strategies."""
    strategy = BasicTransitionStrategy()

    # Mock the underlying strategies
    strategy.confidence_strategy = Mock()
    strategy.entropy_strategy = Mock()
    strategy.confidence_strategy.select_candidates.return_value = ["confidence_result"]
    strategy.entropy_strategy.select_candidates.return_value = ["entropy_result"]

    # Mock metrics calculation to control transition decision
    with patch(
        "audioloop.utils.candidate_selection.basic_transition.calculate_binary_metrics"
    ) as mock_metrics:
        # Test no transition (use confidence)
        mock_metrics.return_value = {
            "f1_score": 0.1,
            "mean_confidence": 0.8,
            "std_confidence": 0.15,
        }

        result = strategy.select_candidates(sample_predictions, num_candidates=2)

        strategy.confidence_strategy.select_candidates.assert_called_once()
        strategy.entropy_strategy.select_candidates.assert_not_called()
        assert result == ["confidence_result"]
        assert strategy.current_strategy_name == "confidence"

        # Reset mocks
        strategy.confidence_strategy.reset_mock()
        strategy.entropy_strategy.reset_mock()

        # Test transition (use entropy)
        mock_metrics.return_value = {
            "f1_score": 0.3,
            "mean_confidence": 0.95,
            "std_confidence": 0.08,
        }

        result = strategy.select_candidates(sample_predictions, num_candidates=2)

        strategy.confidence_strategy.select_candidates.assert_not_called()
        strategy.entropy_strategy.select_candidates.assert_called_once()
        assert result == ["entropy_result"]
        assert strategy.current_strategy_name == "entropy"


def test_basic_transition_behavioral_properties():
    """Test that thresholds have reasonable properties regardless of how they're set."""
    strategies = [
        BasicTransitionStrategy(),  # Defaults
        BasicTransitionStrategy(
            f1_threshold=0.3, confidence_threshold=0.95, variance_threshold=0.1
        ),  # Explicit
    ]

    # Add auto strategy with mocked calculator
    with patch("audioloop.utils.adaptive_thresholds.threshold_calculator") as mock_calc:
        mock_calc.calculate_thresholds.return_value = (0.25, 0.92, 0.08)
        strategies.append(BasicTransitionStrategy(auto_thresholds=True))

    for strategy in strategies:
        # All thresholds should be reasonable
        assert strategy.f1_threshold is not None and 0 < strategy.f1_threshold < 1, (
            f"F1 threshold {strategy.f1_threshold} not in valid range"
        )
        assert (
            strategy.confidence_threshold is not None and 0.5 < strategy.confidence_threshold < 1
        ), f"Confidence threshold {strategy.confidence_threshold} not in valid range"
        assert strategy.variance_threshold is not None and 0 < strategy.variance_threshold < 1, (
            f"Variance threshold {strategy.variance_threshold} not in valid range"
        )

        # Relationships should make sense
        assert strategy.confidence_threshold is not None and strategy.confidence_threshold > 0.5, (
            "Confidence threshold should be better than random"
        )


def test_basic_transition_strategy_get_name():
    """Test basic transition strategy name."""
    strategy = BasicTransitionStrategy()
    assert strategy.get_name() == "Basic Transition"


def test_basic_transition_strategy_name_changes_with_transition():
    """Test that active strategy name reflects current strategy being used."""
    strategy = BasicTransitionStrategy()

    # get_name() should be static
    assert strategy.get_name() == "Basic Transition"

    # get_active_strategy_name() should reflect current state
    assert strategy.get_active_strategy_name() == "Basic Transition (Confidence-Based)"

    # After using entropy, active strategy name should change
    strategy.current_strategy_name = "entropy"
    assert strategy.get_name() == "Basic Transition"  # Still static
    assert strategy.get_active_strategy_name() == "Basic Transition (Entropy-Based)"


def test_basic_transition_with_real_metrics_calculation(predictions_with_ground_truth):
    """Test BasicTransitionStrategy with real metrics calculation."""
    strategy = BasicTransitionStrategy(
        f1_threshold=0.99,  # Very high threshold that won't be met
        confidence_threshold=0.99,  # Very high threshold that won't be met
        variance_threshold=0.01,  # Very low threshold that won't be met
    )

    candidates = strategy.select_candidates(
        predictions_with_ground_truth, num_candidates=2, random_seed=42
    )

    # Should use confidence strategy since thresholds are very high
    assert strategy.current_strategy_name == "confidence"
    assert len(candidates) <= 2


def test_basic_transition_print_analysis_no_crash(capsys):
    """Test that _print_transition_analysis doesn't crash with various inputs."""
    strategy = BasicTransitionStrategy()

    # Test with valid metrics
    metrics = {
        "f1_score": 0.25,
        "mean_confidence": 0.92,
        "std_confidence": 0.08,
    }
    strategy._print_transition_analysis(metrics, True)

    # Test with empty metrics
    strategy._print_transition_analysis({}, False)

    # Test with None metrics
    strategy._print_transition_analysis({}, False)

    # Capture output to verify it produces some output
    captured = capsys.readouterr()
    assert "Basic Transition" in captured.out


# create_strategy function tests
def test_create_strategy_confidence():
    """Test create_strategy function with confidence mode."""
    strategy = create_strategy("confidence")
    assert isinstance(strategy, ConfidenceStrategy)


def test_create_strategy_entropy():
    """Test create_strategy function with entropy mode."""
    strategy = create_strategy("entropy")
    assert isinstance(strategy, EntropyStrategy)


def test_create_strategy_basic_transition():
    """Test create_strategy function with basic_transition mode."""
    strategy = create_strategy("basic_transition")
    assert isinstance(strategy, BasicTransitionStrategy)


def test_create_strategy_invalid_mode():
    """Test create_strategy function with invalid mode."""
    with pytest.raises(ValueError, match="Unknown selection mode"):
        create_strategy("invalid_mode")


# Parametrized tests
@pytest.mark.parametrize("strategy_class", [ConfidenceStrategy, EntropyStrategy])
def test_strategies_handle_kwargs(strategy_class, sample_predictions):
    """Test that strategies properly handle keyword arguments."""
    strategy = strategy_class()

    # Test with various kwargs
    candidates = strategy.select_candidates(
        sample_predictions,
        num_candidates=2,
        positive_percentage=0.5,
        candidate_pool_multiplier=3,
        random_seed=42,
        extra_param="ignored",  # Should be ignored
    )

    assert len(candidates) <= 2  # May be less if not enough predictions


@pytest.mark.parametrize("strategy_class", [ConfidenceStrategy, EntropyStrategy])
def test_strategies_reproducible_with_seed(strategy_class, sample_predictions):
    """Test that strategies produce reproducible results with same seed."""
    strategy = strategy_class()

    candidates1 = strategy.select_candidates(sample_predictions, num_candidates=3, random_seed=42)

    candidates2 = strategy.select_candidates(sample_predictions, num_candidates=3, random_seed=42)

    # Should get same results with same seed
    assert len(candidates1) == len(candidates2)

    # Check that the same files were selected (order might differ due to shuffling)
    filenames1 = {c["filename"] for c in candidates1}
    filenames2 = {c["filename"] for c in candidates2}
    assert filenames1 == filenames2


@pytest.mark.parametrize("num_candidates", [1, 3, 5, 10])
def test_confidence_strategy_respects_num_candidates(sample_predictions, num_candidates):
    """Test that confidence strategy respects num_candidates parameter."""
    strategy = ConfidenceStrategy()

    candidates = strategy.select_candidates(
        sample_predictions, num_candidates=num_candidates, random_seed=42
    )

    # The strategy may return fewer candidates due to positive/negative balancing
    # and available samples, so just verify we don't exceed the request
    assert len(candidates) <= num_candidates
    assert len(candidates) > 0 or num_candidates == 0


@pytest.mark.parametrize("positive_percentage", [0.1, 0.3, 0.5, 0.7, 0.9])
def test_entropy_strategy_respects_positive_percentage(sample_predictions, positive_percentage):
    """Test that entropy strategy respects positive_percentage parameter."""
    strategy = EntropyStrategy()

    candidates = strategy.select_candidates(
        sample_predictions,
        num_candidates=4,
        positive_percentage=positive_percentage,
        random_seed=42,
    )

    positive_count = sum(1 for c in candidates if c["predicted_class"] == "positive")
    expected_positive = int(4 * positive_percentage)

    # Should match expected count or be limited by available samples
    available_positive = sum(1 for p in sample_predictions if p["predicted_class"] == "positive")
    expected_positive = min(expected_positive, available_positive)

    assert positive_count == expected_positive


# Edge case tests for all strategies
@pytest.mark.parametrize("strategy_class", [ConfidenceStrategy, EntropyStrategy])
def test_strategies_with_zero_candidates(strategy_class, sample_predictions):
    """Test strategies handle zero candidates gracefully."""
    strategy = strategy_class()

    candidates = strategy.select_candidates(sample_predictions, num_candidates=0, random_seed=42)

    assert len(candidates) == 0


@pytest.mark.parametrize("strategy_class", [ConfidenceStrategy, EntropyStrategy])
def test_strategies_with_custom_class_names(strategy_class, sample_predictions):
    """Test strategies work with custom positive/negative class names."""
    strategy = strategy_class()

    # Modify predictions to use custom class names
    custom_predictions = []
    for p in sample_predictions:
        custom_p = p.copy()
        custom_p["predicted_class"] = "dog_bark" if p["predicted_class"] == "positive" else "other"
        custom_predictions.append(custom_p)

    candidates = strategy.select_candidates(
        custom_predictions,
        num_candidates=2,
        positive_class_name="dog_bark",
        negative_class_name="other",
        random_seed=42,
    )

    assert len(candidates) <= 2
    # Verify the custom class names are preserved
    for candidate in candidates:
        assert candidate["predicted_class"] in ["dog_bark", "other"]


def test_basic_transition_edge_cases():
    """Test edge cases specifically for BasicTransitionStrategy."""
    strategy = BasicTransitionStrategy()

    # Mock the metrics calculation to avoid needing ground truth
    with patch(
        "audioloop.utils.candidate_selection.basic_transition.calculate_binary_metrics"
    ) as mock_metrics:
        mock_metrics.return_value = {}  # Empty metrics

        # Test the _should_transition method directly with empty metrics
        assert not strategy._should_transition({})

        # Empty predictions
        candidates = strategy.select_candidates([], num_candidates=5, random_seed=42)
        assert len(candidates) == 0

        # Zero candidates requested
        sample_preds = [
            {
                "filename": "test.pt",
                "prediction": 1,
                "predicted_class": "positive",
                "confidence": 0.9,
            }
        ]
        candidates = strategy.select_candidates(sample_preds, num_candidates=0, random_seed=42)
        assert len(candidates) == 0


# Utility function tests
def test_load_predictions(tmp_path):
    """Test load_predictions function."""
    # Create a temporary CSV file
    csv_file = tmp_path / "test_predictions.csv"
    csv_content = """filename,confidence,entropy,prediction,predicted_class
file1.pt,0.95,0.1,1,positive
file2.pt,0.85,0.3,0,negative
"""
    csv_file.write_text(csv_content)

    # Load predictions
    predictions = load_predictions(str(csv_file))

    assert len(predictions) == 2
    assert predictions[0]["filename"] == "file1.pt"
    assert predictions[0]["confidence"] == 0.95
    assert predictions[0]["entropy"] == 0.1
    assert predictions[0]["prediction"] == "1"
    assert predictions[0]["predicted_class"] == "positive"
    assert predictions[1]["filename"] == "file2.pt"
    assert predictions[1]["confidence"] == 0.85
    assert predictions[1]["entropy"] == 0.3
    assert predictions[1]["prediction"] == "0"
    assert predictions[1]["predicted_class"] == "negative"


def test_save_candidates(tmp_path):
    """Test save_candidates function."""
    # Create test candidates
    candidates = [
        {
            "filename": "file1.pt",
            "prediction": 1,
            "predicted_class": "positive",
            "confidence": 0.95,
        },
        {
            "filename": "file2.pt",
            "prediction": 0,
            "predicted_class": "negative",
            "confidence": 0.85,
        },
    ]

    # Save candidates
    output_file = tmp_path / "test_candidates.csv"
    save_candidates(candidates, str(output_file))

    # Verify file was created and contains expected content
    assert output_file.exists()
    content = output_file.read_text()
    assert (
        "filename,prediction,predicted_class,confidence,needs_human_label,human_confidence"
        in content
    )
    assert "file1.pt,1,positive,0.95,," in content
    assert "file2.pt,0,negative,0.85,," in content


def test_save_candidates_empty_list(tmp_path, caplog):
    """Test save_candidates with empty candidates list."""
    output_file = tmp_path / "empty_candidates.csv"

    with caplog.at_level(logging.INFO):
        save_candidates([], str(output_file))

    # Should not create file and should log message
    assert not output_file.exists()
    assert "No candidates to save" in caplog.text


def test_save_candidates_creates_directory(tmp_path):
    """Test that save_candidates creates output directory if it doesn't exist."""
    candidates = [
        {"filename": "test.pt", "prediction": 1, "predicted_class": "positive", "confidence": 0.9}
    ]

    # Use a nested directory that doesn't exist
    output_file = tmp_path / "nested" / "dir" / "candidates.csv"
    save_candidates(candidates, str(output_file))

    # Verify directory was created and file exists
    assert output_file.exists()
    assert output_file.parent.exists()


def test_basic_transition_threshold_consistency():
    """Test that thresholds are internally consistent regardless of exact values."""
    # Test different ways of creating strategies
    strategy1 = BasicTransitionStrategy()  # Defaults

    # Create another strategy with same defaults
    f1, conf, var = get_default_thresholds()
    strategy2 = BasicTransitionStrategy(
        f1_threshold=f1, confidence_threshold=conf, variance_threshold=var
    )

    # Both should have identical thresholds
    assert strategy1.f1_threshold == strategy2.f1_threshold
    assert strategy1.confidence_threshold == strategy2.confidence_threshold
    assert strategy1.variance_threshold == strategy2.variance_threshold

    # Test metrics that are clearly above/below thresholds
    high_metrics = {
        "f1_score": max((strategy1.f1_threshold or 0.2) * 2, 0.8),
        "mean_confidence": min((strategy1.confidence_threshold or 0.9) + 0.05, 0.98),
        "std_confidence": max((strategy1.variance_threshold or 0.12) * 0.5, 0.02),
    }

    low_metrics = {
        "f1_score": (strategy1.f1_threshold or 0.2) * 0.5,
        "mean_confidence": (strategy1.confidence_threshold or 0.9) * 0.9,
        "std_confidence": (strategy1.variance_threshold or 0.12) * 2,
    }

    # Both strategies should behave the same
    assert strategy1._should_transition(high_metrics) == strategy2._should_transition(high_metrics)
    assert strategy1._should_transition(low_metrics) == strategy2._should_transition(low_metrics)


def test_basic_transition_threshold_properties():
    """Test that thresholds have sensible properties without checking exact values."""
    strategy = BasicTransitionStrategy()

    # Test invariants that should always hold
    assert strategy.f1_threshold is not None and strategy.f1_threshold > 0, (
        "F1 threshold must be positive"
    )
    assert strategy.f1_threshold is not None and strategy.f1_threshold < 1, (
        "F1 threshold must be less than 1"
    )
    assert strategy.confidence_threshold is not None and strategy.confidence_threshold > 0.5, (
        "Confidence threshold should be better than random"
    )
    assert strategy.confidence_threshold is not None and strategy.confidence_threshold < 1, (
        "Confidence threshold must be less than 1"
    )
    assert strategy.variance_threshold is not None and strategy.variance_threshold > 0, (
        "Variance threshold must be positive"
    )
    assert strategy.variance_threshold is not None and strategy.variance_threshold < 0.5, (
        "Variance threshold should be reasonable"
    )

    # Test relationships
    assert (
        strategy.confidence_threshold is not None
        and strategy.f1_threshold is not None
        and strategy.confidence_threshold > strategy.f1_threshold
    ), "Confidence typically higher than F1"

    # Test with extreme cases to verify threshold logic
    definitely_good = {
        "f1_score": 0.9,
        "mean_confidence": 0.98,
        "std_confidence": 0.01,
    }

    definitely_bad = {
        "f1_score": 0.05,
        "mean_confidence": 0.55,
        "std_confidence": 0.4,
    }

    assert strategy._should_transition(definitely_good), "Should transition with excellent metrics"
    assert not strategy._should_transition(definitely_bad), (
        "Should not transition with poor metrics"
    )


def test_basic_transition_relative_behavior():
    """Test BasicTransitionStrategy behavior relative to its own thresholds."""
    strategy = BasicTransitionStrategy()

    # Create metrics relative to the strategy's actual thresholds
    just_above_all = {
        "f1_score": (strategy.f1_threshold or 0.2) + 0.01,
        "mean_confidence": (strategy.confidence_threshold or 0.9) + 0.01,
        "std_confidence": (strategy.variance_threshold or 0.12) - 0.01,
    }

    just_below_f1 = {
        "f1_score": (strategy.f1_threshold or 0.2) - 0.01,
        "mean_confidence": (strategy.confidence_threshold or 0.9) + 0.01,
        "std_confidence": (strategy.variance_threshold or 0.12) - 0.01,
    }

    just_below_conf = {
        "f1_score": (strategy.f1_threshold or 0.2) + 0.01,
        "mean_confidence": (strategy.confidence_threshold or 0.9) - 0.01,
        "std_confidence": (strategy.variance_threshold or 0.12) - 0.01,
    }

    just_above_var = {
        "f1_score": (strategy.f1_threshold or 0.2) + 0.01,
        "mean_confidence": (strategy.confidence_threshold or 0.9) + 0.01,
        "std_confidence": (strategy.variance_threshold or 0.12) + 0.01,
    }

    # Only when all criteria are met should it transition
    assert strategy._should_transition(just_above_all), "Should transition when all criteria met"
    assert not strategy._should_transition(just_below_f1), "Should not transition with low F1"
    assert not strategy._should_transition(just_below_conf), (
        "Should not transition with low confidence"
    )
    assert not strategy._should_transition(just_above_var), (
        "Should not transition with high variance"
    )

# =============================================================================
# Mixed Entropy Strategy Tests
# =============================================================================


class TestMixedEntropyStrategy:
    """Tests for MixedEntropyStrategy."""

    def test_basic_distribution(self):
        """Test that samples are distributed 70/20/10 across entropy levels."""
        strategy = MixedEntropyStrategy()

        # Create predictions with known entropy distribution
        predictions = []
        for i in range(200):
            predictions.append({
                "filename": f"file{i}.pt",
                "prediction": 1 if i % 2 == 0 else 0,
                "predicted_class": "positive" if i % 2 == 0 else "negative",
                "confidence": 0.5 + (i / 400),  # 0.5 to 1.0
                "entropy": 1.0 - (i / 200),  # 1.0 to 0.0 (descending)
                "filepath": f"path/file{i}.pt",
            })

        candidates = strategy.select_candidates(predictions, 50)

        assert len(candidates) == 50, "Should return exactly 50 candidates"

        # Check entropy distribution
        # High: top 20% of 200 = 40 examples (entropy 1.0 to 0.8)
        # Medium: next 40% = 80 examples (entropy 0.8 to 0.4)
        # Low: bottom 40% = 80 examples (entropy 0.4 to 0.0)

        high_count = sum(1 for c in candidates if c["entropy"] >= 0.8)
        medium_count = sum(1 for c in candidates if 0.4 <= c["entropy"] < 0.8)
        low_count = sum(1 for c in candidates if c["entropy"] < 0.4)

        # Allow some tolerance due to pool multiplier sampling
        assert 30 <= high_count <= 40, f"Expected ~35 high-entropy, got {high_count}"
        assert 5 <= medium_count <= 15, f"Expected ~10 medium-entropy, got {medium_count}"
        assert 3 <= low_count <= 12, f"Expected ~5 low-entropy, got {low_count}"

    def test_entropy_ordering(self):
        """Test that high bucket has higher entropy than medium, medium > low."""
        strategy = MixedEntropyStrategy()

        predictions = []
        for i in range(150):
            predictions.append({
                "filename": f"file{i}.pt",
                "prediction": 1,
                "predicted_class": "positive",
                "confidence": 0.9,
                "entropy": 1.0 - (i / 150),  # Descending from 1.0 to 0.0
                "filepath": f"path/file{i}.pt",
            })

        candidates = strategy.select_candidates(predictions, 50)

        # Separate by entropy level based on known distribution
        high = [c for c in candidates if c["entropy"] >= 0.8]
        medium = [c for c in candidates if 0.4 <= c["entropy"] < 0.8]
        low = [c for c in candidates if c["entropy"] < 0.4]

        if high and medium:
            assert min(c["entropy"] for c in high) >= max(c["entropy"] for c in medium), (
                "High bucket should have higher entropy than medium"
            )

        if medium and low:
            assert min(c["entropy"] for c in medium) >= max(c["entropy"] for c in low), (
                "Medium bucket should have higher entropy than low"
            )

    def test_small_dataset_fallback(self):
        """Test fallback to pure high-entropy sampling for datasets < 100 examples."""
        strategy = MixedEntropyStrategy()

        # Small dataset (< 100)
        predictions = []
        for i in range(50):
            predictions.append({
                "filename": f"file{i}.pt",
                "prediction": 1,
                "predicted_class": "positive",
                "confidence": 0.9,
                "entropy": 1.0 - (i / 50),
                "filepath": f"path/file{i}.pt",
            })

        candidates = strategy.select_candidates(predictions, 30)

        assert len(candidates) == 30, "Should return 30 candidates"

        # Should favor high-entropy (fallback mode)
        avg_entropy = sum(c["entropy"] for c in candidates) / len(candidates)
        assert avg_entropy > 0.5, "Should favor high-entropy in fallback mode"

    def test_insufficient_high_entropy(self):
        """Test handling when high-entropy bucket doesn't have enough examples."""
        strategy = MixedEntropyStrategy()

        # Create predictions where only 10 are in top 20% (should target 35)
        predictions = []
        for i in range(100):
            # First 10 are high entropy, rest are low
            entropy = 0.9 if i < 10 else 0.1
            predictions.append({
                "filename": f"file{i}.pt",
                "prediction": 1,
                "predicted_class": "positive",
                "confidence": 0.8,
                "entropy": entropy,
                "filepath": f"path/file{i}.pt",
            })

        candidates = strategy.select_candidates(predictions, 50)

        # Should still return 50 candidates, pulling from available pools
        assert len(candidates) == 50, "Should return 50 candidates despite shortage"

    def test_mutual_exclusivity_with_stratification(self):
        """Test that mixed_entropy rejects positive_percentage parameter."""
        strategy = MixedEntropyStrategy()

        predictions = [
            {
                "filename": "file1.pt",
                "prediction": 1,
                "predicted_class": "positive",
                "confidence": 0.9,
                "entropy": 0.5,
                "filepath": "path/file1.pt",
            }
        ]

        with pytest.raises(ValueError, match="incompatible with positive_percentage"):
            strategy.select_candidates(
                predictions,
                10,
                positive_percentage=0.75  # Should raise error
            )

    def test_pool_multiplier_diversity(self):
        """Test that pool multiplier maintains diversity in sampling."""
        strategy = MixedEntropyStrategy()

        predictions = []
        for i in range(200):
            predictions.append({
                "filename": f"file{i}.pt",
                "prediction": 1,
                "predicted_class": "positive",
                "confidence": 0.8,
                "entropy": 1.0 - (i / 200),
                "filepath": f"path/file{i}.pt",
                "unique_id": i,  # Track which were selected
            })

        # Run multiple times with same seed - should get same results
        candidates1 = strategy.select_candidates(
            predictions, 50, candidate_pool_multiplier=5, random_seed=42
        )
        candidates2 = strategy.select_candidates(
            predictions, 50, candidate_pool_multiplier=5, random_seed=42
        )

        # Same seed should give same results
        ids1 = sorted([c["unique_id"] for c in candidates1])
        ids2 = sorted([c["unique_id"] for c in candidates2])
        assert ids1 == ids2, "Same seed should produce same candidates"

        # Different seed should give different results
        candidates3 = strategy.select_candidates(
            predictions, 50, candidate_pool_multiplier=5, random_seed=99
        )
        ids3 = sorted([c["unique_id"] for c in candidates3])
        assert ids1 != ids3, "Different seed should produce different candidates"

    def test_all_same_entropy(self):
        """Test edge case where all predictions have identical entropy."""
        strategy = MixedEntropyStrategy()

        predictions = []
        for i in range(100):
            predictions.append({
                "filename": f"file{i}.pt",
                "prediction": 1,
                "predicted_class": "positive",
                "confidence": 0.8,
                "entropy": 0.5,  # All the same
                "filepath": f"path/file{i}.pt",
            })

        candidates = strategy.select_candidates(predictions, 50)

        # Should still select 50 candidates
        assert len(candidates) == 50, "Should handle uniform entropy distribution"

    def test_get_name(self):
        """Test that strategy has correct name."""
        strategy = MixedEntropyStrategy()
        name = strategy.get_name()

        assert "Mixed" in name, "Name should mention mixed sampling"
        assert "Entropy" in name or "entropy" in name, "Name should mention entropy"

    def test_shuffling(self):
        """Test that candidates are shuffled (not sorted by entropy)."""
        strategy = MixedEntropyStrategy()

        predictions = []
        for i in range(150):
            predictions.append({
                "filename": f"file{i}.pt",
                "prediction": 1,
                "predicted_class": "positive",
                "confidence": 0.8,
                "entropy": 1.0 - (i / 150),  # Strictly descending
                "filepath": f"path/file{i}.pt",
            })

        candidates = strategy.select_candidates(predictions, 50, random_seed=42)

        # Extract entropy values in order
        entropies = [c["entropy"] for c in candidates]

        # Should NOT be sorted (either ascending or descending)
        is_sorted_desc = all(entropies[i] >= entropies[i+1] for i in range(len(entropies)-1))
        is_sorted_asc = all(entropies[i] <= entropies[i+1] for i in range(len(entropies)-1))

        assert not is_sorted_desc and not is_sorted_asc, (
            "Candidates should be shuffled, not sorted by entropy"
        )


class TestMixedEntropyIntegration:
    """Integration tests for MixedEntropyStrategy."""

    def test_create_strategy_factory(self):
        """Test that mixed_entropy can be created via factory function."""
        strategy = create_strategy("mixed_entropy")

        assert isinstance(strategy, MixedEntropyStrategy), (
            "Factory should create MixedEntropyStrategy"
        )

    def test_integration_with_real_workflow(self):
        """Test integration with active learning workflow."""
        strategy = create_strategy("mixed_entropy")

        # Simulate realistic predictions
        predictions = []
        for i in range(1000):
            predictions.append({
                "filename": f"audio{i}.pt",
                "prediction": 1 if i % 10 < 3 else 0,  # ~30% positive
                "predicted_class": "positive" if i % 10 < 3 else "negative",
                "confidence": 0.5 + (i % 50) / 100,  # Varies 0.5-1.0
                "entropy": abs(0.5 - (i % 100) / 100),  # Varies 0.0-0.5
                "filepath": f"data/audio{i}.pt",
            })

        candidates = strategy.select_candidates(
            predictions,
            50,
            candidate_pool_multiplier=5,
            random_seed=42
        )

        assert len(candidates) == 50, "Should select correct number"
        assert all("filename" in c for c in candidates), "Should preserve prediction structure"
        assert all("entropy" in c for c in candidates), "Should preserve entropy values"

    def test_comparison_with_pure_entropy(self):
        """Test that mixed entropy gives different distribution than pure entropy."""
        predictions = []
        for i in range(200):
            predictions.append({
                "filename": f"file{i}.pt",
                "prediction": 1,
                "predicted_class": "positive",
                "confidence": 0.8,
                "entropy": 1.0 - (i / 200),
                "filepath": f"path/file{i}.pt",
            })

        # Pure entropy (use EntropyStrategy)
        pure_strategy = EntropyStrategy()
        pure_candidates = pure_strategy.select_candidates(
            predictions, 50, random_seed=42
        )

        # Mixed entropy
        mixed_strategy = MixedEntropyStrategy()
        mixed_candidates = mixed_strategy.select_candidates(
            predictions, 50, random_seed=42
        )

        # Calculate average entropy for each
        pure_avg = sum(c["entropy"] for c in pure_candidates) / len(pure_candidates)
        mixed_avg = sum(c["entropy"] for c in mixed_candidates) / len(mixed_candidates)

        # Mixed should have higher average entropy (focuses on top 20% bucket)
        # Pure entropy with pool_multiplier=5 samples from much broader range
        assert mixed_avg > pure_avg, (
            f"Mixed entropy should have higher avg entropy than pure "
            f"(mixed: {mixed_avg:.4f}, pure: {pure_avg:.4f})"
        )
