import csv
import os
import random
import time
from typing import Any


class CandidateSelector:
    """
    Handles candidate selection for active learning.

    This class wraps the existing candidate selection logic to make it more
    modular and easier to extend with different selection strategies in the future.
    """

    def __init__(
        self,
        total_candidates: int = 50,
        positive_percentage: float = 0.80,
        min_confidence: float = 0.8,
        candidate_pool_multiplier: int = 5,
    ):
        """
        Initialize the candidate selector.

        Args:
            total_candidates: Total number of candidates to select
            positive_percentage: Percentage of candidates that should be positive predictions (0.0-1.0)
            min_confidence: Minimum confidence threshold for candidate selection
            candidate_pool_multiplier: Multiplier for candidate pool size (e.g., 5 means sample from top 50 if selecting 10)
        """
        self.total_candidates = total_candidates
        self.positive_percentage = positive_percentage
        self.min_confidence = min_confidence
        self.candidate_pool_multiplier = candidate_pool_multiplier

    def select_candidates(
        self,
        predictions_file: str,
        candidates_csv: str,
        positive_class_name: str = "positive",
        negative_class_name: str = "negative",
    ) -> list[dict[str, Any]]:
        """
        Select predictions for human labeling in active learning.
        Uses percentage-based selection to focus on positive predictions for imbalanced datasets.

        Args:
            predictions_file: CSV file with model predictions
            candidates_csv: Output file for candidates
            positive_class_name: Name for positive class
            negative_class_name: Name for negative class

        Returns:
            list: Selected candidates for human labeling
        """
        # Read predictions
        predictions = self._load_predictions(predictions_file)

        # Calculate numbers based on percentage
        num_positive = int(self.total_candidates * self.positive_percentage)
        num_negative = self.total_candidates - num_positive

        # Separate positive and negative predictions
        positive_preds = [p for p in predictions if p["prediction"] == positive_class_name]
        negative_preds = [p for p in predictions if p["prediction"] == negative_class_name]

        # Sort by confidence (highest first)
        positive_preds.sort(key=lambda x: x["confidence"], reverse=True)
        negative_preds.sort(key=lambda x: x["confidence"], reverse=True)

        # Create broader candidate pools for sampling
        positive_pool_size = min(len(positive_preds), num_positive * self.candidate_pool_multiplier)
        negative_pool_size = min(len(negative_preds), num_negative * self.candidate_pool_multiplier)

        positive_pool = positive_preds[:positive_pool_size]
        negative_pool = negative_preds[:negative_pool_size]

        # Randomly sample from the pools to improve diversity
        random.seed(int(time.time()))

        positive_candidates = random.sample(positive_pool, min(num_positive, len(positive_pool)))
        negative_candidates = random.sample(negative_pool, min(num_negative, len(negative_pool)))

        # Combine and shuffle final candidates
        all_candidates = positive_candidates + negative_candidates
        random.shuffle(all_candidates)

        # Print statistics and recommendations
        self._print_statistics(
            predictions,
            positive_preds,
            negative_preds,
            positive_candidates,
            negative_candidates,
            positive_class_name,
            negative_class_name,
            positive_pool_size,
            negative_pool_size,
        )

        # Save candidates to file
        if all_candidates:
            self._save_candidates(all_candidates, candidates_csv)
            self._print_candidate_summary(
                positive_candidates, negative_candidates, positive_class_name, negative_class_name
            )
        else:
            print("No candidates found.")

        return all_candidates

    def _load_predictions(self, predictions_file: str) -> list[dict[str, Any]]:
        """Load predictions from CSV file."""
        predictions = []
        with open(predictions_file) as f:
            reader = csv.DictReader(f)
            for row in reader:
                row["confidence"] = float(row["confidence"])
                row["entropy"] = float(row["entropy"])
                predictions.append(row)
        return predictions

    def _print_statistics(
        self,
        all_predictions: list[dict[str, Any]],
        positive_preds: list[dict[str, Any]],
        negative_preds: list[dict[str, Any]],
        positive_candidates: list[dict[str, Any]],
        negative_candidates: list[dict[str, Any]],
        positive_class_name: str,
        negative_class_name: str,
        positive_pool_size: int,
        negative_pool_size: int,
    ) -> None:
        """Print comprehensive statistics about the selection process."""
        total_samples = len(all_predictions)

        # Calculate overall accuracy statistics
        correct_predictions = 0
        for p in all_predictions:
            if isinstance(p["correct"], str):
                if p["correct"].lower() == "true":
                    correct_predictions += 1
            elif p["correct"]:
                correct_predictions += 1

        overall_accuracy = correct_predictions / total_samples if total_samples > 0 else 0

        # Calculate accuracy by true class
        true_positive_samples = []
        true_negative_samples = []

        for p in all_predictions:
            true_is_positive = p["true_is_positive"]
            # Handle string/boolean values
            if isinstance(true_is_positive, str):
                true_is_positive = true_is_positive.lower() == "true"

            if true_is_positive:
                true_positive_samples.append(p)
            else:
                true_negative_samples.append(p)

        positive_correct = sum(
            1
            for p in true_positive_samples
            if (isinstance(p["correct"], str) and p["correct"].lower() == "true") or p["correct"]
        )
        negative_correct = sum(
            1
            for p in true_negative_samples
            if (isinstance(p["correct"], str) and p["correct"].lower() == "true") or p["correct"]
        )

        positive_accuracy = (
            positive_correct / len(true_positive_samples) if true_positive_samples else 0
        )
        negative_accuracy = (
            negative_correct / len(true_negative_samples) if true_negative_samples else 0
        )

        # Overall confidence statistics
        all_confidences = [p["confidence"] for p in all_predictions]
        high_conf_count = sum(1 for conf in all_confidences if conf >= self.min_confidence)
        high_conf_percentage = high_conf_count / total_samples if total_samples > 0 else 0

        avg_confidence = sum(all_confidences) / len(all_confidences) if all_confidences else 0
        min_conf = min(all_confidences) if all_confidences else 0
        max_conf = max(all_confidences) if all_confidences else 0

        print("\nModel Performance Summary:")
        print(f"Overall Accuracy: {overall_accuracy:.3f}")
        print(f"True {positive_class_name} Accuracy: {positive_accuracy:.3f}")
        print(f"True {negative_class_name} Accuracy: {negative_accuracy:.3f}")
        print(f"Overall Confidence: avg={avg_confidence:.3f}, range={min_conf:.3f}-{max_conf:.3f}")
        print(
            f"High confidence samples (≥{self.min_confidence}): {high_conf_count}/{total_samples} ({high_conf_percentage:.1%})"
        )

        # Show confidence distribution for each class
        self._print_confidence_distribution(positive_preds, positive_class_name)
        self._print_confidence_distribution(negative_preds, negative_class_name)

        # Recommendation for selection strategy
        if overall_accuracy > 0.95 and high_conf_percentage > 0.8:
            print("💡 Recommendation: Model is highly accurate and confident")
            print("   Consider implementing entropy-based selection for uncertainty sampling")
            print("   Current confidence-based selection may not find challenging cases")

        print("\nActive Learning Candidate Selection:")
        print(f"Available {positive_class_name} predictions: {len(positive_preds)}")
        print(f"Available {negative_class_name} predictions: {len(negative_preds)}")
        print(
            f"Candidate pool sizes: {positive_pool_size} {positive_class_name}, {negative_pool_size} {negative_class_name}"
        )
        print(f"Selected {len(positive_candidates)} {positive_class_name} candidates")
        print(f"Selected {len(negative_candidates)} {negative_class_name} candidates")
        print(
            f"Total candidates for labeling: {len(positive_candidates) + len(negative_candidates)}"
        )

    def _print_confidence_distribution(
        self, predictions: list[dict[str, Any]], class_name: str
    ) -> None:
        """Print confidence distribution for a class."""
        if not predictions:
            print(f"{class_name}: No predictions available")
            return

        confidences = [p["confidence"] for p in predictions]
        print(f"{class_name} confidence distribution:")
        print(f"  Count: {len(confidences)}")
        print(f"  Range: {min(confidences):.3f} - {max(confidences):.3f}")
        print(f"  Mean: {sum(confidences) / len(confidences):.3f}")

    def _save_candidates(self, candidates: list[dict[str, Any]], candidates_csv: str) -> None:
        """Save candidates to CSV file."""
        # Add labeling helper columns
        for candidate in candidates:
            candidate["needs_human_label"] = ""  # Empty column for human to fill
            candidate["human_confidence"] = ""  # Human confidence in the label

        # Ensure outputs directory exists
        os.makedirs(
            os.path.dirname(candidates_csv) if os.path.dirname(candidates_csv) else "outputs",
            exist_ok=True,
        )

        # Save candidates
        fieldnames = list(candidates[0].keys())
        with open(candidates_csv, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(candidates)

        print(f"Candidates saved to: {candidates_csv}")

    def _print_candidate_summary(
        self,
        positive_candidates: list[dict[str, Any]],
        negative_candidates: list[dict[str, Any]],
        positive_class_name: str,
        negative_class_name: str,
    ) -> None:
        """Print summary of selected candidates."""
        if positive_candidates:
            conf_values = [p["confidence"] for p in positive_candidates]
            print(
                f"{positive_class_name} confidence range: {min(conf_values):.3f} - {max(conf_values):.3f}"
            )
            print(
                f"  First 3 {positive_class_name} samples: {[p['filename'] for p in positive_candidates[:3]]}"
            )
        if negative_candidates:
            conf_values = [p["confidence"] for p in negative_candidates]
            print(
                f"{negative_class_name} confidence range: {min(conf_values):.3f} - {max(conf_values):.3f}"
            )
            print(
                f"  First 3 {negative_class_name} samples: {[p['filename'] for p in negative_candidates[:3]]}"
            )
