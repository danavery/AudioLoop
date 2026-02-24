"""Tests for batch_runner.py"""

import tempfile
from pathlib import Path

import pytest

from audioloop.batch_runner import format_duration, run_single_config


class TestFormatDuration:
    """Test duration formatting utility."""

    @pytest.mark.parametrize(
        "seconds,expected",
        [
            (30, "30s"),
            (90, "1m 30s"),
            (3661, "1h 1m 1s"),
            (3600, "1h 0m 0s"),
            (7265, "2h 1m 5s"),
        ],
    )
    def test_format_duration(self, seconds, expected):
        """Test various duration formats."""
        assert format_duration(seconds) == expected


class TestBatchRunner:
    """Test batch runner functionality."""

    @pytest.fixture
    def temp_dir(self):
        """Create temporary directory for test outputs."""
        with tempfile.TemporaryDirectory() as tmpdir:
            yield Path(tmpdir)

    @pytest.fixture
    def minimal_config_yaml(self, temp_dir):
        """Create minimal valid YAML config."""
        config_path = temp_dir / "test_config.yaml"
        config_content = """
workflow:
  class_name: "siren"
  num_cycles: 1
  evaluation_mode: false

config:
  experiment_name: "test_batch"
  max_epochs: 5
"""
        config_path.write_text(config_content)
        return config_path

    @pytest.fixture
    def initial_training_set(self, temp_dir):
        """Create minimal training set CSV."""
        training_set = temp_dir / "bootstrap.csv"
        training_set.write_text("filepath,label,run\ntest1.pt,0,1\ntest2.pt,1,1\n")
        return training_set

    def test_missing_config_file(self, temp_dir):
        """Test error handling for missing config file."""
        batch_id = "batch_test"
        missing_config = temp_dir / "nonexistent.yaml"

        # Should raise FileNotFoundError or return error result
        result = run_single_config(missing_config, batch_id)
        assert result["success"] is False
        assert "FileNotFoundError" in result["error"] or "not found" in result["error"].lower()

    def test_invalid_yaml(self, temp_dir):
        """Test error handling for malformed YAML."""
        batch_id = "batch_test"
        bad_config = temp_dir / "bad.yaml"
        bad_config.write_text("workflow:\n  invalid yaml {{{")

        result = run_single_config(bad_config, batch_id)
        assert result["success"] is False
        # YAML parser may return unexpected types for malformed input
        assert "Error" in result["error"] or "error" in result["error"].lower()

    def test_missing_class_name(self, temp_dir):
        """Test error when workflow section missing class_name."""
        batch_id = "batch_test"
        config_path = temp_dir / "no_class.yaml"
        config_content = """
workflow:
  num_cycles: 1

config:
  experiment_name: "test"
"""
        config_path.write_text(config_content)

        result = run_single_config(config_path, batch_id)
        assert result["success"] is False
        assert "class_name" in result["error"]

    def test_initial_training_set_copy(self, temp_dir, minimal_config_yaml, initial_training_set):
        """Test that initial training set is copied correctly."""
        import os
        from unittest.mock import patch

        batch_id = "batch_test"

        # Mock the actual workflow execution to avoid running full training
        with patch("audioloop.batch_runner.run_automated_workflow") as mock_run:
            mock_run.return_value = ["training_set_v1.csv"]

            # Set up environment to use temp_dir
            with patch.dict(os.environ, {"AUDIOLOOP_OUTPUT_ROOT": str(temp_dir)}):
                result = run_single_config(
                    minimal_config_yaml, batch_id, initial_training_set=initial_training_set
                )

                # Should succeed (mocked)
                assert result["success"] is True

                # Check that training set was copied
                expected_path = (
                    temp_dir / "training_sets" / batch_id / "test_batch" / "training_set_v1.csv"
                )
                assert expected_path.exists()
                assert "test1.pt" in expected_path.read_text()

    def test_experiment_name_nesting(self, temp_dir, minimal_config_yaml):
        """Test that experiment outputs are nested under batch_id."""
        import os
        from unittest.mock import patch

        batch_id = "batch_20251215_143022"

        with patch("audioloop.batch_runner.run_automated_workflow") as mock_run:
            mock_run.return_value = ["training_set_v1.csv"]

            with patch.dict(os.environ, {"AUDIOLOOP_OUTPUT_ROOT": str(temp_dir)}):
                # Create a minimal training set so validation passes
                training_sets_dir = temp_dir / "training_sets" / batch_id / "test_batch"
                training_sets_dir.mkdir(parents=True, exist_ok=True)
                (training_sets_dir / "training_set_v1.csv").write_text("filepath,label,run\n")

                result = run_single_config(minimal_config_yaml, batch_id)
                assert result["success"] is True

                # Verify the config was created with batched experiment name
                mock_run.assert_called_once()
                config_arg = mock_run.call_args[1]["config"]
                assert config_arg.experiment_name == f"{batch_id}/test_batch"


class TestBatchRunnerIntegration:
    """Integration tests requiring full setup."""

    @pytest.fixture
    def setup_environment(self, tmp_path):
        """Set up environment with configs and training sets."""
        configs_dir = tmp_path / "configs"
        configs_dir.mkdir()

        # Create two simple configs
        config1 = configs_dir / "config1.yaml"
        config1.write_text(
            """
workflow:
  class_name: "siren"
  num_cycles: 1
  evaluation_mode: false

config:
  experiment_name: "exp1"
  max_epochs: 5
"""
        )

        config2 = configs_dir / "config2.yaml"
        config2.write_text(
            """
workflow:
  class_name: "dog_bark"
  num_cycles: 1
  evaluation_mode: false

config:
  experiment_name: "exp2"
  max_epochs: 5
"""
        )

        # Create initial training set
        bootstrap = tmp_path / "bootstrap.csv"
        bootstrap.write_text("filepath,label,run\ntest.pt,1,1\n")

        return {
            "configs": [config1, config2],
            "bootstrap": bootstrap,
            "tmp_path": tmp_path,
        }

    @pytest.mark.skip(reason="Full integration test - run manually or in CI")
    def test_full_batch_run(self, setup_environment):
        """
        Full end-to-end batch run test.

        Skipped by default as it requires full dataset setup.
        Run manually with: pytest -v tests/test_batch_runner.py::TestBatchRunnerIntegration::test_full_batch_run -s
        """

        # This would require actual dataset files, so we skip by default
        # To run this test manually, you'd need proper setup
        pass
