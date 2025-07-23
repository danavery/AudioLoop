"""
Simple dataset discovery for AudioLoop.

Convention:
- Filename determines dataset name: my_dataset_config.py -> "my_dataset"  
- Class name follows pattern: my_dataset_config.py -> MyDatasetConfig
- Direct import approach - no complex discovery needed
"""

from pathlib import Path

from .dataset_config import DatasetConfig


def get_dataset_config_class(name: str) -> type[DatasetConfig]:
    """Get dataset config by importing the module directly."""
    try:
        # Direct import: "fsd50k" -> import audioloop.datasets.fsd50k_config
        module_name = f"{name}_config"
        module = __import__(f"audioloop.datasets.{module_name}", fromlist=[module_name])
        
        # Convention: look for class named {Name}Config
        # fsd50k -> FSD50KConfig, urbansound8k -> UrbanSound8KConfig
        if name == "fsd50k":
            class_name = "FSD50KConfig"
        elif name == "urbansound8k":
            class_name = "UrbanSound8KConfig"
        else:
            # For new datasets, use title case: my_dataset -> MyDatasetConfig
            class_name = f"{''.join(word.capitalize() for word in name.split('_'))}Config"
        
        if hasattr(module, class_name):
            return getattr(module, class_name)
        raise ValueError(f"Expected class '{class_name}' not found in {module_name}.py")
            
    except ImportError:
        # Show available options by scanning directory
        available = list_available_datasets()
        raise ValueError(
            f"Dataset '{name}' not found. Available: {', '.join(sorted(available))}\n"
            f"To add '{name}': Create audioloop/datasets/{name}_config.py with class {''.join(word.capitalize() for word in name.split('_'))}Config"
        ) from None


def list_available_datasets() -> list[str]:
    """List available datasets by scanning the datasets directory."""
    datasets_dir = Path(__file__).parent
    available = []
    
    for py_file in datasets_dir.glob("*_config.py"):
        # Skip the base DatasetConfig file
        if py_file.name == "dataset_config.py":
            continue
            
        # Extract dataset name by removing _config suffix
        dataset_name = py_file.stem.replace("_config", "")
        available.append(dataset_name)
    
    return available