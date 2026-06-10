"""Tests for StratifiedUncertaintyStrategy: the extreme-imbalance selection strategy.

The whole point of this strategy is the positive quota
(min(available, max(min_positives, ratio * num)) capped at num_candidates): if that
arithmetic drifts, the strategy silently degrades into the all-negative selector it was
built to prevent — no crash, just a starved active-learning loop. These tests pin each
branch of the quota and the no-positives fallback.
"""

from audioloop.config import AudioLoopConfig
from audioloop.utils.candidate_selection import create_strategy
from audioloop.utils.candidate_selection.stratified import StratifiedUncertaintyStrategy


def _pred(name, predicted_class, entropy):
    return {"filename": name, "predicted_class": predicted_class, "entropy": entropy}


def _make_predictions(n_positive, n_negative):
    """Predictions with distinct, deterministic entropies (pos{0} is most uncertain)."""
    positives = [_pred(f"pos{i}", "positive", 0.9 - i * 0.01) for i in range(n_positive)]
    negatives = [_pred(f"neg{i}", "negative", 0.8 - i * 0.01) for i in range(n_negative)]
    return positives + negatives


def _split(candidates):
    pos = [c for c in candidates if c["predicted_class"] == "positive"]
    neg = [c for c in candidates if c["predicted_class"] == "negative"]
    return pos, neg


class TestPositiveQuota:
    def test_min_positives_floor_beats_small_ratio(self):
        """num=10 at ratio 0.4 targets 4, but the floor of 5 wins: 5 pos + 5 neg."""
        strategy = StratifiedUncertaintyStrategy(min_positives=5, target_positive_ratio=0.4)
        selected = strategy.select_candidates(_make_predictions(10, 30), num_candidates=10)

        pos, neg = _split(selected)
        assert len(selected) == 10
        assert len(pos) == 5
        assert len(neg) == 5

    def test_ratio_target_when_above_floor(self):
        """num=20 at ratio 0.4 targets 8 > floor of 5: 8 pos + 12 neg."""
        strategy = StratifiedUncertaintyStrategy(min_positives=5, target_positive_ratio=0.4)
        selected = strategy.select_candidates(_make_predictions(10, 30), num_candidates=20)

        pos, neg = _split(selected)
        assert len(pos) == 8
        assert len(neg) == 12

    def test_scarce_positives_all_selected(self):
        """Fewer positives than the quota: take every one, fill the rest with negatives."""
        strategy = StratifiedUncertaintyStrategy(min_positives=5, target_positive_ratio=0.4)
        selected = strategy.select_candidates(_make_predictions(2, 30), num_candidates=10)

        pos, neg = _split(selected)
        assert {c["filename"] for c in pos} == {"pos0", "pos1"}
        assert len(neg) == 8

    def test_quota_never_exceeds_num_candidates(self):
        """num_candidates below min_positives: the request wins, all slots go to positives."""
        strategy = StratifiedUncertaintyStrategy(min_positives=5, target_positive_ratio=0.4)
        selected = strategy.select_candidates(_make_predictions(10, 30), num_candidates=3)

        pos, neg = _split(selected)
        assert len(selected) == 3
        assert len(pos) == 3
        assert len(neg) == 0


class TestUncertaintyOrdering:
    def test_selects_most_uncertain_within_each_class(self):
        """Within each predicted class, the highest-entropy samples are chosen."""
        strategy = StratifiedUncertaintyStrategy(min_positives=5, target_positive_ratio=0.4)
        selected = strategy.select_candidates(_make_predictions(10, 30), num_candidates=10)

        pos, neg = _split(selected)
        # Entropies descend with index, so the top-5 of each class are indices 0-4.
        assert {c["filename"] for c in pos} == {f"pos{i}" for i in range(5)}
        assert {c["filename"] for c in neg} == {f"neg{i}" for i in range(5)}


class TestNoPositivesFallback:
    def test_falls_back_to_pure_entropy(self):
        """With zero predicted positives, the strategy degrades to global entropy top-k
        instead of returning an empty or short selection."""
        strategy = StratifiedUncertaintyStrategy()
        selected = strategy.select_candidates(_make_predictions(0, 30), num_candidates=10)

        assert len(selected) == 10
        assert {c["filename"] for c in selected} == {f"neg{i}" for i in range(10)}


class TestDeterminism:
    def test_same_seed_same_selection_and_order(self):
        strategy = StratifiedUncertaintyStrategy()
        predictions = _make_predictions(10, 30)

        first = strategy.select_candidates(predictions, num_candidates=10, random_seed=7)
        second = strategy.select_candidates(predictions, num_candidates=10, random_seed=7)

        assert [c["filename"] for c in first] == [c["filename"] for c in second]

    def test_seed_only_affects_order_not_membership(self):
        """The shuffle is presentation-only: membership is determined by entropy + quota."""
        strategy = StratifiedUncertaintyStrategy()
        predictions = _make_predictions(10, 30)

        a = strategy.select_candidates(predictions, num_candidates=10, random_seed=1)
        b = strategy.select_candidates(predictions, num_candidates=10, random_seed=2)

        assert {c["filename"] for c in a} == {c["filename"] for c in b}


def test_factory_constructs_stratified_strategy():
    config = AudioLoopConfig(selection_mode="stratified_uncertainty")
    strategy = create_strategy(config)

    assert isinstance(strategy, StratifiedUncertaintyStrategy)
    assert strategy.get_name() == "Stratified Uncertainty Sampling"
