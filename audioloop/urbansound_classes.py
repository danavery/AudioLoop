"""UrbanSound8K class definitions and utilities.

This module provides convenient access to UrbanSound8K class information
for setting up binary classification tasks.
"""

# UrbanSound8K class mapping
URBANSOUND8K_CLASSES = {
    0: "air_conditioner",
    1: "car_horn",
    2: "children_playing",
    3: "dog_bark",
    4: "drilling",
    5: "engine_idling",
    6: "gun_shot",
    7: "jackhammer",
    8: "siren",
    9: "street_music"
}

# Reverse mapping for convenience
CLASS_NAME_TO_ID = {name: class_id for class_id, name in URBANSOUND8K_CLASSES.items()}


def get_class_name(class_id):
    """Get human-readable class name from UrbanSound8K class ID.

    Args:
        class_id (int): UrbanSound8K class ID (0-9)

    Returns:
        str: Human-readable class name

    Raises:
        ValueError: If class_id is not valid
    """
    if class_id not in URBANSOUND8K_CLASSES:
        raise ValueError(f"Invalid class ID: {class_id}. Must be 0-9.")
    return URBANSOUND8K_CLASSES[class_id]


def get_class_id(class_name):
    """Get UrbanSound8K class ID from human-readable class name.

    Args:
        class_name (str): Human-readable class name

    Returns:
        int: UrbanSound8K class ID (0-9)

    Raises:
        ValueError: If class_name is not valid
    """
    if class_name not in CLASS_NAME_TO_ID:
        valid_names = list(CLASS_NAME_TO_ID.keys())
        raise ValueError(f"Invalid class name: '{class_name}'. Valid names: {valid_names}")
    return CLASS_NAME_TO_ID[class_name]


def list_classes():
    """Print all available UrbanSound8K classes."""
    print("UrbanSound8K Classes:")
    print("=" * 30)
    for class_id, name in URBANSOUND8K_CLASSES.items():
        print(f"{class_id}: {name}")


def create_negative_class_name(positive_class_name):
    """Create a sensible negative class name for binary classification.

    Args:
        positive_class_name (str): Name of the positive class

    Returns:
        str: Suggested negative class name
    """
    return f"not_{positive_class_name}"


# Common binary classification presets
BINARY_PRESETS = {
    "siren_detection": {
        "positive_class_id": 8,
        "positive_class_name": "siren",
        "negative_class_name": "not_siren"
    },
    "dog_detection": {
        "positive_class_id": 3,
        "positive_class_name": "dog_bark",
        "negative_class_name": "not_dog_bark"
    },
    "gunshot_detection": {
        "positive_class_id": 6,
        "positive_class_name": "gun_shot",
        "negative_class_name": "not_gun_shot"
    },
    "horn_detection": {
        "positive_class_id": 1,
        "positive_class_name": "car_horn",
        "negative_class_name": "not_car_horn"
    },
    "drilling_detection": {
        "positive_class_id": 4,
        "positive_class_name": "drilling",
        "negative_class_name": "not_drilling"
    }
}


def get_binary_preset(preset_name):
    """Get predefined binary classification settings.

    Args:
        preset_name (str): Name of the preset (e.g., "siren_detection")

    Returns:
        dict: Dictionary with positive_class_id, positive_class_name, negative_class_name

    Raises:
        ValueError: If preset_name is not valid
    """
    if preset_name not in BINARY_PRESETS:
        valid_presets = list(BINARY_PRESETS.keys())
        raise ValueError(f"Invalid preset: '{preset_name}'. Valid presets: {valid_presets}")
    return BINARY_PRESETS[preset_name].copy()


def list_binary_presets():
    """Print all available binary classification presets."""
    print("Available Binary Classification Presets:")
    print("=" * 45)
    for preset_name, config in BINARY_PRESETS.items():
        pos_name = config['positive_class_name']
        neg_name = config['negative_class_name']
        class_id = config['positive_class_id']
        print(f"{preset_name:<20} | Class {class_id}: {pos_name} vs {neg_name}")


if __name__ == "__main__":
    # Demo the module
    print("UrbanSound8K Classes Demo")
    print("=" * 40)

    list_classes()
    print()

    list_binary_presets()
    print()

    # Example usage
    print("Example Usage:")
    print("-" * 20)
    siren_config = get_binary_preset("siren_detection")
    print(f"Siren detection config: {siren_config}")

    class_name = get_class_name(8)
    print(f"Class 8 is: {class_name}")

    class_id = get_class_id("dog_bark")
    print(f"'dog_bark' is class: {class_id}")
