"""Tests for RandomStrategy: the baseline every smarter strategy is measured against."""

from audioloop.config import AudioLoopConfig
from audioloop.utils.candidate_selection import create_strategy
from audioloop.utils.candidate_selection.random import RandomStrategy

# Deliberately bare rows: no entropy/confidence/predicted_class. The baseline must not
# touch model outputs, and these would KeyError if it ever started to.
PREDICTIONS = [{"filename": f"clip{i}.pt"} for i in range(20)]


def test_respects_num_candidates():
    selected = RandomStrategy().select_candidates(PREDICTIONS, num_candidates=5)

    assert len(selected) == 5
    assert all(c in PREDICTIONS for c in selected)
    assert len({c["filename"] for c in selected}) == 5  # sampling without replacement


def test_over_asking_returns_everything():
    selected = RandomStrategy().select_candidates(PREDICTIONS, num_candidates=100)

    assert len(selected) == len(PREDICTIONS)


def test_seed_reproducibility():
    a = RandomStrategy().select_candidates(PREDICTIONS, num_candidates=5, random_seed=7)
    b = RandomStrategy().select_candidates(PREDICTIONS, num_candidates=5, random_seed=7)
    c = RandomStrategy().select_candidates(PREDICTIONS, num_candidates=5, random_seed=8)

    assert [x["filename"] for x in a] == [x["filename"] for x in b]
    assert {x["filename"] for x in a} != {x["filename"] for x in c}


def test_factory_constructs_random_strategy():
    strategy = create_strategy(AudioLoopConfig(selection_mode="random"))

    assert isinstance(strategy, RandomStrategy)
    assert strategy.get_name() == "Random (Baseline)"
