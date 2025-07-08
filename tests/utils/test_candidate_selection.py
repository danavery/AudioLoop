"""
Tests for audioloop.utils.candidate_selection module.

This module contains comprehensive tests for the candidate selection strategies
used in the AudioLoop active learning framework, including the simplified
BasicTransitionStrategy design.
"""

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
            "prediction": "positive",
            "confidence": 0.95,
            "entropy": 0.1,
            "filepath": "path/file1.pt",
        },
        {
            "filename": "file2.pt",
            "prediction": "positive",
            "confidence": 0.85,
            "entropy": 0.3,
            "filepath": "path/file2.pt",
        },
        {
            "filename": "file3.pt",
            "prediction": "negative",
            "confidence": 0.92,
            "entropy": 0.2,
            "filepath": "path/file3.pt",
        },
        {
            "filename": "file4.pt",
            "prediction": "negative",
            "confidence": 0.78,
            "entropy": 0.4,
            "filepath": "path/file4.pt",
        },
        {
            "filename": "file5.pt",
            "prediction": "positive",
            "confidence": 0.65,
            "entropy": 0.6,
            "filepath": "path/file5.pt",
        },
        {
            "filename": "file6.pt",
            "prediction": "negative",
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
            "prediction": "positive",
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
                "prediction": "positive",
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
                "prediction": "negative",
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
            "prediction": "positive",
            "confidence": 0.95,
            "entropy": 0.1,
            "true_is_positive": True,
            "predicted_is_positive": True,
            "filepath": "path/file1.pt",
        },
        {
            "filename": "file2.pt",
            "prediction": "positive",
            "confidence": 0.85,
            "entropy": 0.3,
            "true_is_positive": True,
            "predicted_is_positive": True,
            "filepath": "path/file2.pt",
        },
        {
            "filename": "file3.pt",
            "prediction": "negative",
            "confidence": 0.92,
            "entropy": 0.2,
            "true_is_positive": False,
            "predicted_is_positive": False,
            "filepath": "path/file3.pt",
        },
        {
            "filename": "file4.pt",
            "prediction": "negative",
            "confidence": 0.78,
            "entropy": 0.4,
            "true_is_positive": False,
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
    positive_count = sum(1 for c in candidates if c["prediction"] == "positive")
    negative_count = sum(1 for c in candidates if c["prediction"] == "negative")

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
    positive_candidates = [c for c in candidates if c["prediction"] == "positive"]
    negative_candidates = [c for c in candidates if c["prediction"] == "negative"]

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

    positive_count = sum(1 for c in candidates if c["prediction"] == "positive")
    # Should get 3 positives (all available) since we only have 3 positive samples
    assert positive_count == 3

    # Test 20% positive
    candidates = strategy.select_candidates(
        sample_predictions, num_candidates=5, positive_percentage=0.2, random_seed=42
    )

    positive_count = sum(1 for c in candidates if c["prediction"] == "positive")
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

    positive_count = sum(1 for c in candidates if c["prediction"] == "positive")
    negative_count = sum(1 for c in candidates if c["prediction"] == "negative")

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

    positive_count = sum(1 for c in candidates if c["prediction"] == "positive")
    negative_count = sum(1 for c in candidates if c["prediction"] == "negative")

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
    positive_candidates = [c for c in candidates if c["prediction"] == "positive"]
    negative_candidates = [c for c in candidates if c["prediction"] == "negative"]

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
    with pytest.raises(ValueError, match="Cannot specify both auto_thresholds=True and manual thresholds"):
        BasicTransitionStrategy(auto_thresholds=True, f1_threshold=0.4)

    with pytest.raises(ValueError, match="Cannot specify both auto_thresholds=True and manual thresholds"):
        BasicTransitionStrategy(auto_thresholds=True, f1_threshold=0.4, confidence_threshold=0.95, variance_threshold=0.08)


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
    with patch("audioloop.utils.candidate_selection.calculate_binary_metrics") as mock_metrics:
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

    positive_count = sum(1 for c in candidates if c["prediction"] == "positive")
    expected_positive = int(4 * positive_percentage)

    # Should match expected count or be limited by available samples
    available_positive = sum(1 for p in sample_predictions if p["prediction"] == "positive")
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
        custom_p["prediction"] = "dog_bark" if p["prediction"] == "positive" else "other"
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
        assert candidate["prediction"] in ["dog_bark", "other"]


def test_basic_transition_edge_cases():
    """Test edge cases specifically for BasicTransitionStrategy."""
    strategy = BasicTransitionStrategy()

    # Mock the metrics calculation to avoid needing ground truth
    with patch("audioloop.utils.candidate_selection.calculate_binary_metrics") as mock_metrics:
        mock_metrics.return_value = {}  # Empty metrics

        # Test the _should_transition method directly with empty metrics
        assert not strategy._should_transition({})

        # Empty predictions
        candidates = strategy.select_candidates([], num_candidates=5, random_seed=42)
        assert len(candidates) == 0

        # Zero candidates requested
        sample_preds = [{"filename": "test.pt", "prediction": "positive", "confidence": 0.9}]
        candidates = strategy.select_candidates(sample_preds, num_candidates=0, random_seed=42)
        assert len(candidates) == 0


# Utility function tests
def test_load_predictions(tmp_path):
    """Test load_predictions function."""
    # Create a temporary CSV file
    csv_file = tmp_path / "test_predictions.csv"
    csv_content = """filename,confidence,entropy,prediction
file1.pt,0.95,0.1,positive
file2.pt,0.85,0.3,negative
"""
    csv_file.write_text(csv_content)

    # Load predictions
    predictions = load_predictions(str(csv_file))

    assert len(predictions) == 2
    assert predictions[0]["filename"] == "file1.pt"
    assert predictions[0]["confidence"] == 0.95
    assert predictions[0]["entropy"] == 0.1
    assert predictions[0]["prediction"] == "positive"
    assert predictions[1]["filename"] == "file2.pt"
    assert predictions[1]["confidence"] == 0.85
    assert predictions[1]["entropy"] == 0.3
    assert predictions[1]["prediction"] == "negative"


def test_save_candidates(tmp_path):
    """Test save_candidates function."""
    # Create test candidates
    candidates = [
        {
            "filename": "file1.pt",
            "prediction": "positive",
            "confidence": 0.95,
        },
        {
            "filename": "file2.pt",
            "prediction": "negative",
            "confidence": 0.85,
        },
    ]

    # Save candidates
    output_file = tmp_path / "test_candidates.csv"
    save_candidates(candidates, str(output_file))

    # Verify file was created and contains expected content
    assert output_file.exists()
    content = output_file.read_text()
    assert "filename,prediction,confidence,needs_human_label,human_confidence" in content
    assert "file1.pt,positive,0.95,," in content
    assert "file2.pt,negative,0.85,," in content


def test_save_candidates_empty_list(tmp_path, capsys):
    """Test save_candidates with empty candidates list."""
    output_file = tmp_path / "empty_candidates.csv"
    save_candidates([], str(output_file))

    # Should not create file and should print message
    assert not output_file.exists()
    captured = capsys.readouterr()
    assert "No candidates to save" in captured.out


def test_save_candidates_creates_directory(tmp_path):
    """Test that save_candidates creates output directory if it doesn't exist."""
    candidates = [{"filename": "test.pt", "prediction": "positive", "confidence": 0.9}]

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
