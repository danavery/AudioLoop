import random

from .filter_labels import get_matching_paths_from_csv


def write_starting_labels(n=10):
    positives = random.sample(get_matching_paths_from_csv(classnames=["siren"]), k=n)
    negatives = random.sample(get_matching_paths_from_csv(classnames=["siren"], invert=True), k=n)
    return positives, negatives

if __name__ == "__main__":
    run = 1
    positives, negatives = write_starting_labels()
    positives = [f"{p},1,{run}" for p in positives]
    negatives = [f"{n},0,{run}" for n in negatives]
    with open("labels.csv", "w") as f:
        f.write("\n".join(random.sample(positives + negatives, k=len(positives + negatives))))
