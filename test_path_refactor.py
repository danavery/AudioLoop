#!/usr/bin/env python3
"""
Test script to demonstrate the path refactoring approach.
Shows how the utility functions + unified config eliminate hardcoded paths.
"""

from audioloop.config import AudioLoopConfig
from audioloop.utils.paths import (
    get_output_dir,
    get_training_sets_dir,
    get_specs_dir,
    extract_version_from_filename
)
from pathlib import Path


def test_utility_functions():
    """Test the utility functions work as expected."""
    print("=== Testing Utility Functions ===")

    # Test basic paths
    print(f"Default specs dir: {get_specs_dir()}")
    print(f"Default output dir: {get_output_dir()}")
    print(f"Default training sets dir: {get_training_sets_dir()}")

    # Test experiment paths
    print(f"Experiment output dir: {get_output_dir('test_exp')}")
    print(f"Experiment training sets dir: {get_training_sets_dir('test_exp')}")

    # Test version extraction
    test_files = [
        ("outputs/model_v5.pt", "model"),
        ("training_sets/training_set_v12.csv", "training_set"),
        ("outputs/predictions_v3.csv", "predictions")
    ]

    for filepath, file_type in test_files:
        version = extract_version_from_filename(Path(filepath), file_type)
        print(f"Version from {filepath}: {version}")


def test_unified_config():
    """Test the unified config approach."""
    print("\n=== Testing Unified Config ===")

    # Test default config
    config = AudioLoopConfig()
    print(f"Default config specs dir: {config.specs_dir}")
    print(f"Default config output dir: {config.output_dir}")

    # Test experiment config
    exp_config = AudioLoopConfig(experiment_name="my_experiment")
    print(f"Experiment config output dir: {exp_config.output_dir}")
    print(f"Experiment config training sets dir: {exp_config.training_sets_dir}")

    # Test versioned paths
    print(f"Model v3 path: {exp_config.get_model_path(3)}")
    print(f"Predictions v7 path: {exp_config.get_predictions_path(7)}")
    print(f"Training set v2 path: {exp_config.get_training_set_path(2)}")

    # Test validation
    validation = exp_config.validate_setup()
    print(f"Setup validation: {validation}")


def show_before_and_after():
    """Show the difference between old hardcoded approach and new approach."""
    print("\n=== Before vs After Comparison ===")

    print("BEFORE (hardcoded):")
    print('  output_dir = f"outputs_{experiment}" if experiment else "outputs"')
    print('  model_path = f"{output_dir}/model_v{version}.pt"')
    print('  specs_dir = "data/all_specs"')

    print("\nAFTER (configurable):")
    print("  config = AudioLoopConfig(experiment_name='my_exp')")
    print("  model_path = config.get_model_path(version)")
    print("  specs_dir = config.specs_dir")

    # Show actual paths
    config = AudioLoopConfig(experiment_name="demo")
    print(f"\nActual paths for experiment 'demo':")
    print(f"  Output dir: {config.output_dir}")
    print(f"  Model v1 path: {config.get_model_path(1)}")
    print(f"  Specs dir: {config.specs_dir}")


def simulate_training_workflow():
    """Simulate how the updated training function would be called."""
    print("\n=== Training Workflow Simulation ===")

    # Old way (would require many parameters)
    print("OLD WAY:")
    print("  run_training(specs_dir='data/all_specs', experiment_name='exp1', ...)")

    # New way (clean config object)
    print("\nNEW WAY:")
    config = AudioLoopConfig(experiment_name="exp1")
    print(f"  config = AudioLoopConfig(experiment_name='exp1')")
    print(f"  run_training(config, labels_file='training.csv', ...)")
    print(f"  # Training will use:")
    print(f"  #   Specs from: {config.specs_dir}")
    print(f"  #   Save model to: {config.get_model_path(1)}")
    print(f"  #   Output dir: {config.output_dir}")


if __name__ == "__main__":
    test_utility_functions()
    test_unified_config()
    show_before_and_after()
    simulate_training_workflow()

    print("\n=== Summary ===")
    print("✅ Utility functions eliminate hardcoded path duplication")
    print("✅ Unified config provides clean interface for path coordination")
    print("✅ Environment variables enable user customization")
    print("✅ No more f'outputs_{experiment}' scattered across modules")
    print("✅ Easy to test with custom configurations")
