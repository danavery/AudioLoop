"""Tests for merge_training_sets: the data-integrity step of the active-learning loop.

Every merged CSV becomes the next cycle's training set, so silent corruption here compounds
across cycles. These tests cover both input dialects, the label-validation guards, the
filename-keyed replacement semantics (new label wins, merges are idempotent), version
auto-increment, and the metrics side-channel — and pin one current behavior (relative
filepaths pass through unchanged) so a future change is a conscious one.
"""

import logging

import pytest

from audioloop.config import AudioLoopConfig
from audioloop.merge_labels import compute_and_log_candidate_metrics, merge_training_sets

ORIGINAL_FIELDS = ["filename", "label"]
CANDIDATE_FIELDS = ["filename", "prediction", "confidence", "needs_human_label"]


def _read_rows(path):
    import csv

    with open(path) as f:
        return list(csv.DictReader(f))


class TestMergeTrainingSets:
    def test_merges_original_and_labeled_candidates(self, candidates_csv, tmp_path):
        """Original rows come first, labeled candidates append, output is filename/label."""
        original = candidates_csv(
            [{"filename": "a.pt", "label": 1}, {"filename": "b.pt", "label": 0}],
            ORIGINAL_FIELDS,
            name="training_set_v1.csv",
        )
        new_labels = candidates_csv(
            [
                {"filename": "c.pt", "needs_human_label": "1"},
                {"filename": "d.pt", "needs_human_label": "0"},
            ],
            CANDIDATE_FIELDS,
            name="labeled.csv",
        )
        output = tmp_path / "merged.csv"

        result = merge_training_sets(
            str(original), str(new_labels), str(output), log_level=logging.WARNING
        )

        assert result == str(output)
        rows = _read_rows(output)
        assert [(r["filename"], r["label"]) for r in rows] == [
            ("a.pt", "1"),
            ("b.pt", "0"),
            ("c.pt", "1"),
            ("d.pt", "0"),
        ]

    def test_filepath_dialect_normalizes_absolute_paths_only(self, candidates_csv, tmp_path):
        """The original CSV may use a 'filepath' column; absolute paths are reduced to the
        basename. PINNED CURRENT BEHAVIOR: relative paths pass through *unchanged* (the
        basename step only fires on a leading '/'), so mixed-dialect inputs keep their
        prefixes. If normalization is ever extended to relative paths, update this test
        and check existing training sets for already-merged prefixed rows."""
        original = candidates_csv(
            [
                {"filepath": "/data/specs/abs.pt", "label": 1},
                {"filepath": "data/specs/rel.pt", "label": 0},
            ],
            ["filepath", "label"],
            name="training_set_v1.csv",
        )
        new_labels = candidates_csv([], CANDIDATE_FIELDS, name="labeled.csv")
        output = tmp_path / "merged.csv"

        merge_training_sets(str(original), str(new_labels), str(output), log_level=logging.WARNING)

        rows = _read_rows(output)
        assert rows[0]["filename"] == "abs.pt"
        assert rows[1]["filename"] == "data/specs/rel.pt"

    def test_unlabeled_and_invalid_candidate_labels_are_skipped(self, candidates_csv, tmp_path):
        """Only '0'/'1' (after strip) survive: blanks, out-of-range, and junk are dropped."""
        new_labels = candidates_csv(
            [
                {"filename": "keep1.pt", "needs_human_label": " 1 "},  # whitespace ok
                {"filename": "keep0.pt", "needs_human_label": "0"},
                {"filename": "blank.pt", "needs_human_label": ""},  # unlabeled
                {"filename": "range.pt", "needs_human_label": "2"},  # not binary
                {"filename": "junk.pt", "needs_human_label": "yes"},  # not an int
                {"filename": "", "needs_human_label": "1"},  # no filename
            ],
            CANDIDATE_FIELDS,
            name="labeled.csv",
        )
        output = tmp_path / "merged.csv"

        merge_training_sets(
            str(tmp_path / "missing_original.csv"),
            str(new_labels),
            str(output),
            log_level=logging.WARNING,
        )

        rows = _read_rows(output)
        assert [(r["filename"], r["label"]) for r in rows] == [("keep1.pt", "1"), ("keep0.pt", "0")]

    def test_missing_original_starts_fresh(self, candidates_csv, tmp_path):
        """A nonexistent original is a warning + fresh start, not an error (cycle 1 case)."""
        new_labels = candidates_csv(
            [{"filename": "only.pt", "needs_human_label": "1"}],
            CANDIDATE_FIELDS,
            name="labeled.csv",
        )
        output = tmp_path / "merged.csv"

        merge_training_sets(
            str(tmp_path / "nope.csv"), str(new_labels), str(output), log_level=logging.WARNING
        )

        assert [(r["filename"], r["label"]) for r in _read_rows(output)] == [("only.pt", "1")]

    def test_headerless_original_raises(self, candidates_csv, tmp_path):
        """An original without a recognizable header is rejected loudly: silently treating
        the first data row as a header would drop a sample and corrupt the merge."""
        original = tmp_path / "training_set_v1.csv"
        original.write_text("a.pt,1\nb.pt,0\n")
        new_labels = candidates_csv([], CANDIDATE_FIELDS, name="labeled.csv")

        with pytest.raises(ValueError, match="must have headers"):
            merge_training_sets(
                str(original), str(new_labels), str(tmp_path / "out.csv"),
                log_level=logging.WARNING,
            )

    def test_candidates_without_required_columns_contribute_nothing(
        self, candidates_csv, tmp_path
    ):
        """A candidates file missing needs_human_label adds no rows (logged, not raised)."""
        original = candidates_csv(
            [{"filename": "a.pt", "label": 1}], ORIGINAL_FIELDS, name="training_set_v1.csv"
        )
        bad_candidates = candidates_csv(
            [{"filename": "c.pt", "label": 1}], ORIGINAL_FIELDS, name="wrong_format.csv"
        )
        output = tmp_path / "merged.csv"

        merge_training_sets(
            str(original), str(bad_candidates), str(output), log_level=logging.WARNING
        )

        assert [(r["filename"], r["label"]) for r in _read_rows(output)] == [("a.pt", "1")]

    def test_new_label_replaces_existing_entry(self, candidates_csv, tmp_path):
        """A re-labeled candidate replaces the original row (same position, no duplicate):
        the human's latest label is the freshest truth."""
        original = candidates_csv(
            [{"filename": "a.pt", "label": 1}, {"filename": "b.pt", "label": 0}],
            ORIGINAL_FIELDS,
            name="training_set_v1.csv",
        )
        new_labels = candidates_csv(
            [
                {"filename": "a.pt", "needs_human_label": "0"},  # contradicts original -> wins
                {"filename": "c.pt", "needs_human_label": "1"},  # genuinely new -> appended
            ],
            CANDIDATE_FIELDS,
            name="labeled.csv",
        )
        output = tmp_path / "merged.csv"

        merge_training_sets(str(original), str(new_labels), str(output), log_level=logging.WARNING)

        rows = _read_rows(output)
        assert [(r["filename"], r["label"]) for r in rows] == [
            ("a.pt", "0"),  # replaced in place
            ("b.pt", "0"),
            ("c.pt", "1"),
        ]

    def test_remerging_same_candidates_is_idempotent(self, candidates_csv, tmp_path):
        """Running the same merge twice yields identical output — re-merges can't inflate
        the training set anymore."""
        original = candidates_csv(
            [{"filename": "a.pt", "label": 1}], ORIGINAL_FIELDS, name="training_set_v1.csv"
        )
        new_labels = candidates_csv(
            [{"filename": "b.pt", "needs_human_label": "1"}],
            CANDIDATE_FIELDS,
            name="labeled.csv",
        )
        first = tmp_path / "merged_v2.csv"
        second = tmp_path / "merged_v3.csv"

        merge_training_sets(str(original), str(new_labels), str(first), log_level=logging.WARNING)
        merge_training_sets(str(first), str(new_labels), str(second), log_level=logging.WARNING)

        assert _read_rows(first) == _read_rows(second)

    def test_duplicates_within_inputs_collapse_to_last(self, candidates_csv, tmp_path):
        """Duplicates inside either input collapse, last occurrence wins — so a merge also
        cleans up training sets inflated by pre-fix merges."""
        original = candidates_csv(
            [{"filename": "a.pt", "label": 1}, {"filename": "a.pt", "label": 0}],
            ORIGINAL_FIELDS,
            name="training_set_v1.csv",
        )
        new_labels = candidates_csv(
            [
                {"filename": "b.pt", "needs_human_label": "0"},
                {"filename": "b.pt", "needs_human_label": "1"},
            ],
            CANDIDATE_FIELDS,
            name="labeled.csv",
        )
        output = tmp_path / "merged.csv"

        merge_training_sets(str(original), str(new_labels), str(output), log_level=logging.WARNING)

        rows = _read_rows(output)
        assert [(r["filename"], r["label"]) for r in rows] == [("a.pt", "0"), ("b.pt", "1")]

    def test_auto_output_increments_version_from_original(self, candidates_csv):
        """With no output_csv, the version is parsed from the original filename and bumped:
        training_set_v3 -> config.get_training_set_path(4)."""
        config = AudioLoopConfig(experiment_name="merge-test")
        original = candidates_csv(
            [{"filename": "a.pt", "label": 1}], ORIGINAL_FIELDS, name="training_set_v3.csv"
        )
        new_labels = candidates_csv([], CANDIDATE_FIELDS, name="labeled.csv")

        result = merge_training_sets(
            str(original), str(new_labels), config=config, log_level=logging.WARNING
        )

        assert result == str(config.get_training_set_path(4))
        assert config.get_training_set_path(4).exists()

    def test_auto_output_defaults_to_v2_for_unversioned_original(self, candidates_csv):
        """An original without a version suffix is treated as v1, so output is v2."""
        config = AudioLoopConfig(experiment_name="merge-test")
        original = candidates_csv(
            [{"filename": "a.pt", "label": 1}], ORIGINAL_FIELDS, name="bootstrap.csv"
        )
        new_labels = candidates_csv([], CANDIDATE_FIELDS, name="labeled.csv")

        result = merge_training_sets(
            str(original), str(new_labels), config=config, log_level=logging.WARNING
        )

        assert result == str(config.get_training_set_path(2))

    def test_no_output_and_no_config_raises(self, candidates_csv):
        new_labels = candidates_csv([], CANDIDATE_FIELDS, name="labeled.csv")

        with pytest.raises(ValueError, match="config is required"):
            merge_training_sets("training_set_v1.csv", str(new_labels))


class TestComputeAndLogCandidateMetrics:
    def test_no_config_returns_empty(self, candidates_csv):
        path = candidates_csv([], CANDIDATE_FIELDS, name="labeling_candidates_v1.csv")
        assert compute_and_log_candidate_metrics(str(path), config=None) == {}

    def test_unversioned_filename_skips_metrics(self, candidates_csv):
        """No cycle number in the filename -> metrics are skipped, not guessed."""
        config = AudioLoopConfig(experiment_name="merge-test")
        path = candidates_csv(
            [{"filename": "a.pt", "prediction": "True", "needs_human_label": "1"}],
            CANDIDATE_FIELDS,
            name="labeled.csv",
        )

        assert compute_and_log_candidate_metrics(str(path), config) == {}

    def test_versioned_candidates_compute_and_persist_metrics(self, candidates_csv):
        """A labeling_candidates_v{N} file yields model-vs-human metrics and appends them
        to candidate_metrics_history.json (the cycle-stopping criteria's data source)."""
        config = AudioLoopConfig(experiment_name="merge-test")
        config.create_directories()
        path = candidates_csv(
            [
                # prediction True / human 1 -> TP; True / 0 -> FP; False / 1 -> FN.
                {"filename": "tp.pt", "prediction": "True", "needs_human_label": "1"},
                {"filename": "fp.pt", "prediction": "True", "needs_human_label": "0"},
                {"filename": "fn.pt", "prediction": "False", "needs_human_label": "1"},
                {"filename": "skip.pt", "prediction": "True", "needs_human_label": ""},
            ],
            CANDIDATE_FIELDS,
            name="labeling_candidates_v2.csv",
        )

        metrics = compute_and_log_candidate_metrics(str(path), config, log_level=logging.WARNING)

        assert metrics["num_candidates"] == 3  # the unlabeled row is excluded
        assert metrics["precision"] == pytest.approx(0.5)  # 1 TP / (1 TP + 1 FP)
        assert metrics["recall"] == pytest.approx(0.5)  # 1 TP / (1 TP + 1 FN)
        assert metrics["f1_score"] == pytest.approx(0.5)
        assert (config.output_dir / "candidate_metrics_history.json").exists()
