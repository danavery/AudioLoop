"""Tests for YAML configuration file loading."""

import tempfile
from pathlib import Path

import pytest
import yaml

from audioloop.config import AudioLoopConfig


def test_from_yaml_basic():
    """Test loading basic YAML config."""
    yaml_content = """
config:
  experiment_name: "test_exp"
  dataset: "fsd50k"
  max_epochs: 500
"""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
        f.write(yaml_content)
        f.flush()
        config = AudioLoopConfig.from_yaml(f.name)

    assert config.experiment_name == "test_exp"
    assert config.dataset == "fsd50k"
    assert config.max_epochs == 500
    assert config.batch_size == 32  # Default value


def test_from_yaml_with_overrides():
    """Test YAML loading with kwargs overrides."""
    yaml_content = """
config:
  max_epochs: 500
  batch_size: 32
"""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
        f.write(yaml_content)
        f.flush()
        config = AudioLoopConfig.from_yaml(f.name, max_epochs=1000, learning_rate=0.01)

    assert config.max_epochs == 1000  # Overridden
    assert config.batch_size == 32  # From YAML
    assert config.learning_rate == 0.01  # Overridden


def test_from_yaml_flat_format():
    """Test loading flat YAML format (no sections)."""
    yaml_content = """
experiment_name: "flat_test"
dataset: "urbansound8k"
max_epochs: 300
"""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
        f.write(yaml_content)
        f.flush()
        config = AudioLoopConfig.from_yaml(f.name)

    assert config.experiment_name == "flat_test"
    assert config.dataset == "urbansound8k"
    assert config.max_epochs == 300


def test_from_yaml_two_section_format():
    """Test loading two-section format (workflow + config)."""
    yaml_content = """
workflow:
  class_name: "Dog"
  num_cycles: 10

config:
  experiment_name: "test"
  dataset: "fsd50k"
  max_epochs: 500
"""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
        f.write(yaml_content)
        f.flush()
        config = AudioLoopConfig.from_yaml(f.name)

    # Config section loaded
    assert config.experiment_name == "test"
    assert config.dataset == "fsd50k"
    assert config.max_epochs == 500

    # Workflow section ignored by from_yaml (handled by load_workflow_params)
    # Just verify it doesn't cause errors


def test_from_yaml_invalid_field():
    """Test that invalid fields raise clear errors."""
    yaml_content = """
config:
  invalid_field_name: "test"
  max_epochs: 500
"""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
        f.write(yaml_content)
        f.flush()

        with pytest.raises(ValueError, match="Invalid config fields"):
            AudioLoopConfig.from_yaml(f.name)


def test_from_yaml_file_not_found():
    """Test proper error for missing file."""
    with pytest.raises(FileNotFoundError):
        AudioLoopConfig.from_yaml("nonexistent_config.yaml")


def test_from_yaml_class_weighting_adaptive():
    """Test class_weighting handles 'adaptive' string."""
    yaml_content = """
config:
  class_weighting: "adaptive"
"""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
        f.write(yaml_content)
        f.flush()
        config = AudioLoopConfig.from_yaml(f.name)

    assert config.class_weighting == "adaptive"


def test_from_yaml_class_weighting_float():
    """Test class_weighting handles float values."""
    yaml_content = """
config:
  class_weighting: 0.25
"""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
        f.write(yaml_content)
        f.flush()
        config = AudioLoopConfig.from_yaml(f.name)

    assert config.class_weighting == 0.25


def test_from_yaml_class_weighting_null():
    """Test class_weighting handles null/None."""
    yaml_content = """
config:
  class_weighting: null
"""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
        f.write(yaml_content)
        f.flush()
        config = AudioLoopConfig.from_yaml(f.name)

    assert config.class_weighting is None


def test_from_yaml_subset_csv_path_conversion():
    """Test that subset_csv strings are converted to Path objects."""
    yaml_content = """
config:
  subset_csv: "subsets/test.csv"
"""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
        f.write(yaml_content)
        f.flush()
        config = AudioLoopConfig.from_yaml(f.name)

    assert isinstance(config.subset_csv, Path)
    assert str(config.subset_csv) == "subsets/test.csv"


def test_from_yaml_empty_file():
    """Test handling of empty YAML file."""
    yaml_content = ""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
        f.write(yaml_content)
        f.flush()
        config = AudioLoopConfig.from_yaml(f.name)

    # Should create config with all defaults
    assert config.dataset == "fsd50k"  # Default
    assert config.max_epochs == 1000  # Default


def test_from_yaml_null_config_section():
    """Test handling of null config section."""
    yaml_content = """
config: null
"""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
        f.write(yaml_content)
        f.flush()
        config = AudioLoopConfig.from_yaml(f.name)

    # Should create config with all defaults
    assert config.dataset == "fsd50k"  # Default


def test_from_yaml_malformed_yaml():
    """Test error handling for malformed YAML."""
    yaml_content = """
config:
  experiment_name: "test
  invalid yaml
"""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
        f.write(yaml_content)
        f.flush()

        with pytest.raises(yaml.YAMLError):
            AudioLoopConfig.from_yaml(f.name)


def test_from_yaml_override_none_filtered():
    """Test that None values in overrides are filtered."""
    yaml_content = """
config:
  max_epochs: 500
  batch_size: 64
"""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
        f.write(yaml_content)
        f.flush()
        # Pass None override - should be ignored, YAML value used
        config = AudioLoopConfig.from_yaml(f.name, max_epochs=None, learning_rate=0.01)

    assert config.max_epochs == 500  # YAML value (None override filtered)
    assert config.batch_size == 64  # YAML value
    assert config.learning_rate == 0.01  # Override applied


def test_from_yaml_boolean_flags():
    """Test boolean flag handling."""
    yaml_content = """
config:
  auto_thresholds: true
  with_ground_truth: false
  use_lr_scheduler: true
"""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
        f.write(yaml_content)
        f.flush()
        config = AudioLoopConfig.from_yaml(f.name)

    assert config.auto_thresholds is True
    assert config.with_ground_truth is False
    assert config.use_lr_scheduler is True


def test_from_yaml_all_training_params():
    """Test loading all training-related parameters."""
    yaml_content = """
config:
  max_epochs: 800
  batch_size: 64
  learning_rate: 0.0005
  weight_decay: 0.00001
  seed: 12345
  num_workers: 4
  use_lr_scheduler: true
  lr_scheduler_factor: 0.3
  lr_scheduler_patience: 10
  lr_scheduler_min_lr: 0.000001
"""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
        f.write(yaml_content)
        f.flush()
        config = AudioLoopConfig.from_yaml(f.name)

    assert config.max_epochs == 800
    assert config.batch_size == 64
    assert config.learning_rate == 0.0005
    assert config.weight_decay == 0.00001
    assert config.seed == 12345
    assert config.num_workers == 4
    assert config.use_lr_scheduler is True
    assert config.lr_scheduler_factor == 0.3
    assert config.lr_scheduler_patience == 10
    assert config.lr_scheduler_min_lr == 0.000001


def test_from_yaml_all_active_learning_params():
    """Test loading all active learning parameters."""
    yaml_content = """
config:
  total_candidates: 100
  positive_percentage: 0.3
  min_confidence: 0.75
  selection_mode: "entropy"
  candidate_pool_multiplier: 10
  basic_transition_f1_threshold: 0.15
  basic_transition_confidence_threshold: 0.85
  basic_transition_variance_threshold: 0.1
  auto_thresholds: true
  estimated_positive_pct: 0.05
"""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
        f.write(yaml_content)
        f.flush()
        config = AudioLoopConfig.from_yaml(f.name)

    assert config.total_candidates == 100
    assert config.positive_percentage == 0.3
    assert config.min_confidence == 0.75
    assert config.selection_mode == "entropy"
    assert config.candidate_pool_multiplier == 10
    assert config.basic_transition_f1_threshold == 0.15
    assert config.basic_transition_confidence_threshold == 0.85
    assert config.basic_transition_variance_threshold == 0.1
    assert config.auto_thresholds is True
    assert config.estimated_positive_pct == 0.05


def test_from_yaml_validation_still_works():
    """Test that __post_init__ validation still runs on YAML configs."""
    yaml_content = """
config:
  max_epochs: -100
"""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
        f.write(yaml_content)
        f.flush()

        with pytest.raises(ValueError, match="max_epochs must be positive"):
            AudioLoopConfig.from_yaml(f.name)


def test_from_yaml_real_example_minimal():
    """Test loading the actual minimal.yaml example."""
    yaml_content = """
workflow:
  class_name: "siren"
  num_cycles: 3
  evaluation_mode: false

config:
  experiment_name: "quick_test"
"""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
        f.write(yaml_content)
        f.flush()
        config = AudioLoopConfig.from_yaml(f.name)

    assert config.experiment_name == "quick_test"
    assert config.dataset == "fsd50k"  # Default


def test_from_yaml_real_example_search_mode():
    """Test loading a search mode configuration."""
    yaml_content = """
workflow:
  class_name: "dog_bark"
  num_cycles: 15
  auto_label: true
  evaluation_mode: true

config:
  experiment_name: "dog_search"
  dataset: "urbansound8k"
  max_epochs: 1000
  batch_size: 32
  total_candidates: 50
  selection_mode: "entropy"
  cycle_stopping_strategy: "search"
  cycle_patience: 5
  precision_floor: "auto"
  with_ground_truth: true
"""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
        f.write(yaml_content)
        f.flush()
        config = AudioLoopConfig.from_yaml(f.name)

    assert config.experiment_name == "dog_search"
    assert config.dataset == "urbansound8k"
    assert config.max_epochs == 1000
    assert config.selection_mode == "entropy"
    assert config.cycle_stopping_strategy == "search"
    assert config.precision_floor == "auto"
    assert config.with_ground_truth is True
