#!/usr/bin/env python3
"""
Tracks key performance metrics for an AudioLoop active learning session.

This script focuses on a "starter pack" of essential metrics to evaluate
the performance of a binary classifier on imbalanced data across multiple
active learning cycles.
"""

import argparse
import csv
import glob
import os
import re
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter

from .config import AudioLoopConfig
from .utils.metrics_utils import (
    calculate_binary_metrics,
    calculate_class_balance_metrics,
    calculate_confidence_percentiles,
    calculate_entropy_metrics,
)


def extract_version_number(filename: str) -> int:
    """Extracts the version number from a prediction filename."""
    match = re.search(r"predictions_v(\d+)\.csv", filename)
    if not match:
        raise ValueError(f"Could not extract version number from {filename}")
    return int(match.group(1))


def calculate_core_metrics(predictions_file: str) -> dict:
    """
    Calculates metrics from a predictions file, handling both production and evaluation modes.

    Args:
        predictions_file: Path to a predictions_v*.csv file.

    Returns:
        A dictionary containing available metrics based on file contents.
    """
    if not os.path.exists(predictions_file):
        raise FileNotFoundError(f"Predictions file not found: {predictions_file}")

    # Load predictions and detect available columns
    predictions = []
    fieldnames = []

    with open(predictions_file, newline="") as csvfile:
        reader = csv.DictReader(csvfile)
        fieldnames = reader.fieldnames or []

        for row in reader:
            # Always load required fields
            pred_data = {
                "prediction": row["prediction"],
                "confidence": float(row["confidence"]),
            }

            # Add optional fields if available
            if "ground_truth" in row and row["ground_truth"] is not None:
                pred_data["ground_truth"] = row["ground_truth"]

            if "entropy" in row and row.get("entropy"):
                pred_data["entropy"] = float(row["entropy"])

            predictions.append(pred_data)

    if not predictions:
        raise ValueError(f"No samples found in {predictions_file}")

    # Check what metrics we can calculate

    has_entropy = "entropy" in fieldnames

    # Calculate available metrics using imported functions
    binary_metrics = calculate_binary_metrics(predictions)
    balance_metrics = calculate_class_balance_metrics(predictions)
    percentile_metrics = calculate_confidence_percentiles(predictions, percentiles=[5, 95])

    # Only calculate entropy metrics if entropy data is available
    entropy_metrics = {}
    if has_entropy:
        entropy_metrics = calculate_entropy_metrics(predictions)

    # Combine all metrics
    all_metrics = {**binary_metrics, **entropy_metrics, **balance_metrics, **percentile_metrics}

    return all_metrics


def track_metrics_across_versions(output_dir: str) -> dict[int, dict]:
    """
    Finds all prediction files and calculates core metrics for each version.

    Args:
        output_dir: The directory containing the prediction files.

    Returns:
        A dictionary mapping version numbers to their calculated metrics.
    """
    pattern = os.path.join(output_dir, "predictions_v*.csv")
    prediction_files = sorted(glob.glob(pattern), key=extract_version_number)

    if not prediction_files:
        raise FileNotFoundError(f"No 'predictions_v*.csv' files found in '{output_dir}'")

    # Detect if any files have ground truth to determine table format
    has_any_ground_truth = False
    for f in prediction_files:
        try:
            with open(f, newline="") as csvfile:
                reader = csv.DictReader(csvfile)
                if "ground_truth" in (reader.fieldnames or []):
                    has_any_ground_truth = True
                    break
        except Exception:
            continue

    # Prepare output lines for both console and file
    output_lines = []
    output_lines.append("Tracking Key Metrics Across Versions:")
    if has_any_ground_truth:
        output_lines.append("Mode: Evaluation (with ground truth)")
        output_lines.append("-" * 120)
        header = f"{'Version':<10} {'F1-Score':<12} {'Precision':<12} {'Recall':<10} {'Neg Acc':<10} {'Mean Conf':<12} {'Std Conf':<11} {'p05-p95 Conf':<15} {'Pred/Actual+':<15}"
    else:
        output_lines.append("Mode: Production (no ground truth - prediction metrics only)")
        output_lines.append("-" * 80)
        header = f"{'Version':<10} {'Mean Conf':<12} {'Std Conf':<11} {'p05-p95 Conf':<15} {'Pred+':<10} {'Total':<10}"

    output_lines.append(header)
    output_lines.append("-" * (120 if has_any_ground_truth else 80))

    # Print to console
    for line in output_lines:
        print(line)

    all_metrics = {}
    for f in prediction_files:
        try:
            version = extract_version_number(f)
            metrics = calculate_core_metrics(f)
            all_metrics[version] = metrics

            p05_p95_str = f"{metrics['p05_confidence']:.2f}-{metrics['p95_confidence']:.2f}"

            if has_any_ground_truth and metrics.get("has_ground_truth", False):
                # Full evaluation metrics format
                metrics_line = (
                    f"v{version:<8} {metrics['f1_score']:<12.3f} "
                    f"{metrics['precision']:<12.3f} {metrics['recall']:<10.3f} "
                    f"{metrics['negative_class_accuracy']:<10.3f} "
                    f"{metrics['mean_confidence']:<12.3f} {metrics['std_confidence']:<11.3f} "
                    f"{p05_p95_str:<15} "
                    f"{metrics['predicted_positive_ratio']:.1%}/{metrics['actual_positive_ratio']:.1%}"
                )
            elif has_any_ground_truth and not metrics.get("has_ground_truth", False):
                # Mixed mode: some files have ground truth, this one doesn't
                metrics_line = (
                    f"v{version:<8} {'N/A':<12} {'N/A':<12} {'N/A':<10} {'N/A':<10} "
                    f"{metrics['mean_confidence']:<12.3f} {metrics['std_confidence']:<11.3f} "
                    f"{p05_p95_str:<15} "
                    f"{metrics['predicted_positive_ratio']:.1%}/N/A"
                )
            else:
                # Production mode: no ground truth anywhere
                metrics_line = (
                    f"v{version:<8} {metrics['mean_confidence']:<12.3f} "
                    f"{metrics['std_confidence']:<11.3f} {p05_p95_str:<15} "
                    f"{metrics['predicted_positive_ratio']:.1%} "
                    f"{metrics['total_samples']:<10}"
                )

            print(metrics_line)
            output_lines.append(metrics_line)
        except Exception as e:
            error_line = f"Could not process file {os.path.basename(f)}: {e}"
            print(error_line)
            output_lines.append(error_line)

    separator = "-" * (120 if has_any_ground_truth else 80)
    print(separator)
    output_lines.append(separator)

    # Save to file
    Path(output_dir).mkdir(exist_ok=True)
    metrics_file = os.path.join(output_dir, "metrics_summary.txt")
    with open(metrics_file, "w") as f:
        f.write("\n".join(output_lines))
    print(f"\nMetrics summary saved to: {metrics_file}")

    return all_metrics


def analyze_trends(results: dict[int, dict], output_dir: str):
    """Prints a summary of the overall trends for key metrics and saves to file."""
    if len(results) < 2:
        return

    versions = sorted(results.keys())
    start_v, end_v = versions[0], versions[-1]
    start_m, end_m = results[start_v], results[end_v]

    # Check if we have ground truth data
    has_ground_truth = start_m.get("has_ground_truth", False) and end_m.get(
        "has_ground_truth", False
    )

    # Prepare output lines for both console and file
    trend_lines = []
    trend_lines.append("\nOverall Performance Summary:")
    if has_ground_truth:
        trend_lines.append("Mode: Evaluation (with ground truth)")
    else:
        trend_lines.append("Mode: Production (prediction metrics only)")
    trend_lines.append("=" * 35)

    def format_trend(metric_name: str, start_val: float, end_val: float, higher_is_better: bool):
        change = end_val - start_val
        arrow = "↑" if (change > 0) == higher_is_better else "↓"
        return f"{arrow} {metric_name:<20}: {start_val:.3f} -> {end_val:.3f} ({change:+.3f})"

    # Always show confidence metrics
    trend_lines.append(
        format_trend("Mean Confidence", start_m["mean_confidence"], end_m["mean_confidence"], True)
    )
    trend_lines.append(
        format_trend("Std Confidence", start_m["std_confidence"], end_m["std_confidence"], True)
    )
    trend_lines.append(
        format_trend("Confidence p05", start_m["p05_confidence"], end_m["p05_confidence"], True)
    )  # Rising floor is good

    # Show evaluation metrics only if ground truth is available
    if has_ground_truth:
        trend_lines.append(
            format_trend("F1-Score", start_m["f1_score"], end_m["f1_score"], higher_is_better=True)
        )
        trend_lines.append(
            format_trend(
                "Negative Class Acc",
                start_m["negative_class_accuracy"],
                end_m["negative_class_accuracy"],
                True,
            )
        )

    # Show entropy if available
    if "mean_entropy" in start_m and "mean_entropy" in end_m:
        trend_lines.append(
            format_trend("Mean Entropy", start_m["mean_entropy"], end_m["mean_entropy"], False)
        )

    # Show prediction ratio trends
    pred_ratio_change = end_m["predicted_positive_ratio"] - start_m["predicted_positive_ratio"]
    if abs(pred_ratio_change) > 0.01:
        if has_ground_truth:
            actual_ratio = end_m["actual_positive_ratio"]
            arrow = (
                "→"
                if abs(end_m["predicted_positive_ratio"] - actual_ratio)
                < abs(start_m["predicted_positive_ratio"] - actual_ratio)
                else "❌"
            )
            trend_lines.append(
                f"{arrow} Predicted+ Ratio     : {start_m['predicted_positive_ratio']:.1%} -> {end_m['predicted_positive_ratio']:.1%} (target: {actual_ratio:.1%})"
            )
        else:
            trend_lines.append(
                f"→ Predicted+ Ratio     : {start_m['predicted_positive_ratio']:.1%} -> {end_m['predicted_positive_ratio']:.1%}"
            )
    trend_lines.append("=" * 35)

    # Print to console
    for line in trend_lines:
        print(line)

    # Append to metrics file
    metrics_file = os.path.join(output_dir, "metrics_summary.txt")
    with open(metrics_file, "a") as f:
        f.write("\n".join(trend_lines))


def plot_core_trends(results: dict[int, dict], save_path: str | None = None):
    """
    Generates and optionally saves plots for the core metrics.
    """
    if not results:
        print("No data available to plot.")
        return

    versions = sorted(results.keys())
    metrics_by_name = defaultdict(list)
    for v in versions:
        for key, value in results[v].items():
            metrics_by_name[key].append(value)

    # Check if we have ground truth data
    has_ground_truth = any(results[v].get("has_ground_truth", False) for v in versions)

    if has_ground_truth:
        fig, axes = plt.subplots(2, 2, figsize=(16, 10))
        fig.suptitle("Core Active Learning Metrics (Evaluation Mode)", fontsize=16)
    else:
        fig, axes = plt.subplots(2, 2, figsize=(16, 10))
        fig.suptitle("Core Active Learning Metrics (Production Mode)", fontsize=16)

    # Plot 1: Performance Metrics (conditional on ground truth)
    if has_ground_truth and "f1_score" in metrics_by_name:
        axes[0, 0].plot(versions, metrics_by_name["f1_score"], "o-", label="F1-Score")
        axes[0, 0].plot(versions, metrics_by_name["precision"], "o--", label="Precision", alpha=0.7)
        axes[0, 0].plot(versions, metrics_by_name["recall"], "o--", label="Recall", alpha=0.7)
        axes[0, 0].plot(
            versions,
            metrics_by_name["negative_class_accuracy"],
            "o:",
            label="Negative Class Acc",
            alpha=0.7,
        )
        axes[0, 0].set_title("Primary Performance (F1, Precision, Recall, Neg Acc)")
        axes[0, 0].set_ylabel("Score")
        axes[0, 0].legend()
        axes[0, 0].set_ylim(0, 1.05)
    else:
        # Production mode: show prediction distribution
        axes[0, 0].plot(
            versions,
            metrics_by_name["predicted_positive_ratio"],
            "o-",
            label="Predicted Positive %",
        )
        axes[0, 0].set_title("Prediction Distribution")
        axes[0, 0].set_ylabel("Predicted Positive Ratio")
        axes[0, 0].yaxis.set_major_formatter(FuncFormatter(lambda y, _: f"{y:.1%}"))
        axes[0, 0].legend()

    # Plot 2: Model Internal State
    mean_conf = metrics_by_name["mean_confidence"]
    p05_conf = metrics_by_name["p05_confidence"]
    p95_conf = metrics_by_name["p95_confidence"]
    axes[0, 1].plot(versions, mean_conf, "o-", color="tab:blue", label="Mean Confidence")
    axes[0, 1].fill_between(
        versions, p05_conf, p95_conf, color="tab:blue", alpha=0.2, label="5th-95th Percentile"
    )

    # Only add entropy if available
    if metrics_by_name.get("mean_entropy"):
        ax2_entropy = axes[0, 1].twinx()
        ax2_entropy.plot(
            versions, metrics_by_name["mean_entropy"], "o--", color="tab:red", label="Mean Entropy"
        )
        ax2_entropy.set_ylabel("Entropy (Uncertainty)", color="tab:red")

    axes[0, 1].set_title("Model Confidence and Uncertainty")
    axes[0, 1].set_ylabel("Confidence", color="tab:blue")
    axes[0, 1].set_ylim(0, 1.05)
    axes[0, 1].legend(loc="upper left")

    # Plot 3: Class Balance Ratios
    axes[1, 0].plot(
        versions,
        metrics_by_name["predicted_positive_ratio"],
        "o--",
        label="Predicted Positive Ratio",
    )

    if has_ground_truth and "actual_positive_ratio" in metrics_by_name:
        axes[1, 0].plot(
            versions,
            metrics_by_name["actual_positive_ratio"],
            "o-",
            label="Actual Positive Ratio",
        )
        axes[1, 0].set_title("Diagnostic: Actual vs. Predicted Class Balance")
    else:
        axes[1, 0].set_title("Predicted Class Balance")

    axes[1, 0].set_ylabel("Positive Class Ratio")
    axes[1, 0].yaxis.set_major_formatter(FuncFormatter(lambda y, _: f"{y:.1%}"))
    axes[1, 0].legend()

    # Plot 4: Confusion Matrix Components or Sample Counts
    if has_ground_truth and "true_positives" in metrics_by_name:
        axes[1, 1].plot(
            versions, metrics_by_name["true_positives"], "o-", label="True Positives (TP)"
        )
        axes[1, 1].plot(
            versions, metrics_by_name["false_positives"], "o-", label="False Positives (FP)"
        )
        axes[1, 1].plot(
            versions, metrics_by_name["false_negatives"], "o-", label="False Negatives (FN)"
        )
        axes[1, 1].set_title("Confusion Matrix Components (Counts)")
        axes[1, 1].set_ylabel("Number of Samples")
        axes[1, 1].legend()
    else:
        # Production mode: show total samples and predicted positives
        axes[1, 1].plot(versions, metrics_by_name["total_samples"], "o-", label="Total Samples")
        axes[1, 1].plot(
            versions, metrics_by_name["predicted_positives"], "o-", label="Predicted Positives"
        )
        axes[1, 1].set_title("Sample Counts")
        axes[1, 1].set_ylabel("Number of Samples")
        axes[1, 1].legend()

    for ax in axes.flat:
        ax.set_xlabel("Version")
        ax.grid(True, alpha=0.4)
        ax.set_xticks(versions)

    plt.tight_layout(rect=(0, 0, 1, 0.96))

    if save_path:
        plt.savefig(save_path, dpi=300)
        print(f"\nPlot saved to '{save_path}'")

    plt.show()


def main():
    """Main entry point for the metrics tracking script."""
    parser = argparse.ArgumentParser(
        description="Track core performance metrics for an AudioLoop active learning session.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Example Usage:
  # Analyze metrics and display plots
  python -m audioloop.track_metrics --plot

  # Analyze metrics and save the plot to a file
  python -m audioloop.track_metrics --save-plot learning_curves.png

  # Track metrics from experiment directory
  python -m audioloop.track_metrics --experiment myexp --plot

Notes:
  • Automatically detects evaluation mode (with ground truth) vs production mode
  • Evaluation mode: Shows F1, precision, recall, accuracy when ground truth available
  • Production mode: Shows prediction and confidence metrics only
  • Mixed mode: Handles files with and without ground truth gracefully
        """,
    )
    parser.add_argument(
        "--experiment",
        type=str,
        help="Experiment name to customize output directory (default: outputs, with experiment: outputs/experiment)",
    )
    parser.add_argument("--plot", action="store_true", help="Display plots of the metrics.")
    parser.add_argument(
        "--save-plot",
        type=str,
        default=None,
        help="Save the metrics plots to the specified file path.",
    )

    args = parser.parse_args()

    # Use config to determine output directory
    config = AudioLoopConfig(experiment_name=args.experiment)
    output_dir = str(config.output_dir)

    try:
        metrics_results = track_metrics_across_versions(output_dir)
        analyze_trends(metrics_results, output_dir)
        if args.plot or args.save_plot:
            plot_core_trends(metrics_results, args.save_plot)
    except FileNotFoundError as e:
        print(f"Error: {e}")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")


if __name__ == "__main__":
    main()
