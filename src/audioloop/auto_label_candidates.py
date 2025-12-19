#!/usr/bin/env python3
"""
Standalone auto-labeling tool for candidates with ground truth.

This tool automatically labels candidate files using ground truth data from
the 'ground_truth' column. It's intended for evaluation/research workflows
where ground truth is available.

Usage:
  # Auto-label candidates file
  python -m audioloop.auto_label_candidates outputs/labeling_candidates_v1.csv

  # Auto-label with verbose output
  python -m audioloop.auto_label_candidates outputs/labeling_candidates_v1.csv --verbose

Requirements:
  - Candidates CSV must have 'ground_truth' column (from evaluation mode)
  - This suggests the CSV was generated with --evaluation-mode flag
  - For production workflows without ground truth, use label_audio.py instead

The tool will:
  1. Read candidates CSV file
  2. Use 'ground_truth' column to set 'needs_human_label' field
  3. Save results back to the same file
  4. Print summary statistics

This is equivalent to running automated_workflow.py with --evaluation-mode --auto-label
but can be used standalone for testing or manual evaluation workflows.
"""

import argparse
import sys
from pathlib import Path


def main():
    """Main entry point for auto-labeling CLI tool."""
    parser = argparse.ArgumentParser(
        description="Auto-label candidates using ground truth data",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Auto-label candidates from active learning
  python -m audioloop.auto_label_candidates outputs/labeling_candidates_v1.csv

  # Auto-label with verbose output
  python -m audioloop.auto_label_candidates outputs/labeling_candidates_v1.csv --verbose

  # Auto-label candidates from specific experiment
  python -m audioloop.auto_label_candidates outputs/myexp/labeling_candidates_v2.csv

Note: This tool requires ground truth data (ground_truth column) in the CSV.
For production workflows without ground truth, use 'python -m audioloop.label_audio' instead.
        """,
    )

    parser.add_argument("candidates_csv", help="Path to candidates CSV file with ground truth data")

    parser.add_argument(
        "--verbose", "-v", action="store_true", help="Show detailed progress information"
    )

    args = parser.parse_args()

    # Validate file exists
    candidates_path = Path(args.candidates_csv)
    if not candidates_path.exists():
        print(f"Error: Candidates file not found: {args.candidates_csv}")
        sys.exit(1)

    # Import and run auto-labeling
    try:
        from audioloop.utils.auto_labeling import auto_label_from_ground_truth

        print(f"Auto-labeling candidates: {args.candidates_csv}")
        print("=" * 60)

        if not args.verbose:
            # Suppress detailed progress for cleaner output
            import contextlib
            import io

            # Capture progress output but let summary through
            with contextlib.redirect_stdout(io.StringIO()):
                results = auto_label_from_ground_truth(args.candidates_csv)

            # Show summary
            print("Auto-labeling complete:")
            print(f"  Positive samples: {results['positive_count']}")
            print(f"  Negative samples: {results['negative_count']}")
            print(f"  Total samples: {results['total']}")
            print(f"  Results saved to: {args.candidates_csv}")
        else:
            # Show all output including progress
            results = auto_label_from_ground_truth(args.candidates_csv)

        print("=" * 60)
        print("✅ Auto-labeling completed successfully")

        # Show some statistics
        if results["total"] > 0:
            pos_pct = (results["positive_count"] / results["total"]) * 100
            print(f"📊 Dataset balance: {pos_pct:.1f}% positive, {100 - pos_pct:.1f}% negative")

        return 0

    except ValueError as e:
        print(f"❌ Error: {e}")
        print("\nThis usually means:")
        print("  • The CSV was generated in production mode (no ground truth)")
        print("  • Use --evaluation-mode flag when generating candidates")
        print("  • For manual labeling, use 'python -m audioloop.label_audio' instead")
        return 1

    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
