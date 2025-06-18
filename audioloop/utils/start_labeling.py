import os
import random

from .filter_labels import get_matching_paths_from_csv


def write_starting_labels(n=10, classname="siren"):
    """Create initial training set for any class.

    Args:
        n: Number of positive and negative samples each
        classname: UrbanSound8K class name for positive class

    Returns:
        (positives, negatives): Lists of file paths
    """
    positives = random.sample(get_matching_paths_from_csv(classnames=[classname]), k=n)
    negatives = random.sample(get_matching_paths_from_csv(classnames=[classname], invert=True), k=n)
    return positives, negatives


def create_training_set(n=10, classname="siren", output_path="training_sets/training_set_v1.csv", run=1):
    """Create training set CSV file for any class.

    Args:
        n: Number of positive and negative samples each
        classname: UrbanSound8K class name for positive class
        output_path: Path to save training set CSV
        run: Run number for training set versioning
    """
    positives, negatives = write_starting_labels(n, classname)

    # Format entries
    positive_entries = [f"{p},1,{run}" for p in positives]
    negative_entries = [f"{n},0,{run}" for n in negatives]
    all_entries = positive_entries + negative_entries

    # Shuffle and write
    random.shuffle(all_entries)

    # Create output directory if needed
    output_dir = os.path.dirname(output_path)
    if output_dir:  # Only create directory if path has a directory component
        os.makedirs(output_dir, exist_ok=True)

    with open(output_path, "w") as f:
        f.write("filepath,label,run\n")
        f.write("\n".join(all_entries))

    print(f"Created training set: {output_path}")
    print(f"  {len(positives)} {classname} samples, {len(negatives)} non-{classname} samples")

    return positives, negatives

if __name__ == "__main__":
    # Default behavior - create siren training set
    create_training_set()
