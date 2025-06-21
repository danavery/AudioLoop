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







if __name__ == "__main__":
    # Demo the module
    print("UrbanSound8K Classes Demo")
    print("=" * 40)

    list_classes()
    print()

    # Example usage
    print("Example Usage:")
    print("-" * 20)
    class_name = get_class_name(8)
    print(f"Class 8 is: {class_name}")

    class_id = get_class_id("dog_bark")
    print(f"'dog_bark' is class: {class_id}")

    negative_name = f"not_siren"
    print(f"Negative class for 'siren': {negative_name}")
