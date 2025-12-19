"""
Adaptive threshold calculation for BasicTransition strategy.

This module automatically calculates optimal BasicTransition thresholds based on
dataset characteristics like class imbalance, complexity, and training set size.
"""


class AdaptiveThresholdCalculator:
    """Calculate BasicTransition thresholds based on class imbalance and training set size."""

    def __init__(self):
        # Base thresholds for balanced binary classification
        self.base_f1 = 0.35  # Much higher F1 threshold - model must be mature before entropy
        self.base_confidence = 0.85  # Lower confidence threshold
        self.base_variance = 0.18  # More tolerant variance

    def calculate_thresholds(
        self, estimated_positive_pct: float = 0.05, training_set_size: int | None = None
    ) -> tuple[float, float, float]:
        """
        Calculate adaptive thresholds for BasicTransition strategy.

        Args:
            estimated_positive_pct: Estimated percentage of positive samples (0.0-1.0)
            training_set_size: Size of current training set (optional)

        Returns:
            Tuple of (f1_threshold, confidence_threshold, variance_threshold)
        """
        # F1 threshold based on class imbalance
        # More imbalanced → slightly lower F1 threshold, but still keep it high
        baseline_ratio = 0.1  # Assume 10% is "balanced" for binary classification
        imbalance_factor = max(
            0.7, (estimated_positive_pct / baseline_ratio) ** 0.5
        )  # Less aggressive reduction
        f1_threshold = self.base_f1 * imbalance_factor

        # Confidence threshold based on imbalance (very imbalanced = less reliable)
        if estimated_positive_pct < 0.02:
            confidence_adj = -0.15  # Very aggressive for very rare classes
        elif estimated_positive_pct < 0.05:
            confidence_adj = -0.12  # More aggressive for rare classes
        elif estimated_positive_pct < 0.1:
            confidence_adj = -0.08  # More aggressive for imbalanced classes
        else:
            confidence_adj = -0.02  # Slightly more aggressive for balanced classes

        confidence_threshold = self.base_confidence + confidence_adj

        # Training set size adjustment
        if training_set_size:
            if training_set_size < 50:
                confidence_threshold -= 0.12  # Very aggressive for tiny sets
            elif training_set_size < 100:
                confidence_threshold -= 0.08  # More aggressive for small sets
            elif training_set_size > 500:
                confidence_threshold += 0.01  # Less conservative for large sets

        # Variance threshold - more tolerant for imbalanced classes
        if estimated_positive_pct < 0.05:
            variance_threshold = self.base_variance + 0.08  # Much more tolerant for rare classes
        else:
            variance_threshold = self.base_variance + 0.03  # Slightly more tolerant overall

        # Apply safety bounds - keep F1 threshold high to prevent premature entropy switching
        f1_threshold = max(0.25, min(0.5, f1_threshold))  # Much higher minimum F1
        confidence_threshold = max(0.65, min(0.92, confidence_threshold))
        variance_threshold = max(0.08, min(0.3, variance_threshold))

        return f1_threshold, confidence_threshold, variance_threshold

    def print_analysis(
        self,
        estimated_positive_pct: float,
        f1_threshold: float,
        confidence_threshold: float,
        variance_threshold: float,
        training_set_size: int | None = None,
    ) -> None:
        """Print detailed analysis of threshold calculation."""

        print("\n🎯 Adaptive Threshold Analysis:")
        print("=" * 50)
        print(f"Estimated positive class prevalence: {estimated_positive_pct:.1%}")
        if training_set_size:
            print(f"Training set size: {training_set_size}")

        print("\n📊 Calculated Thresholds:")
        print(f"F1 threshold: {f1_threshold:.3f}")
        print(f"Confidence threshold: {confidence_threshold:.3f}")
        print(f"Variance threshold: {variance_threshold:.3f}")

        print("\n💡 Rationale:")
        if estimated_positive_pct < 0.02:
            print(
                "• Very rare class (< 2%) → High F1 threshold to ensure model maturity before entropy"
            )
        elif estimated_positive_pct < 0.05:
            print("• Rare class (< 5%) → High F1 threshold, aggressive confidence")
        elif estimated_positive_pct < 0.1:
            print("• Imbalanced class (< 10%) → Moderate F1 threshold, moderate confidence")
        else:
            print("• Relatively balanced class (≥ 10%) → Standard high thresholds")

        if training_set_size and training_set_size < 50:
            print("• Very small training set → Very aggressive transition")
        elif training_set_size and training_set_size < 100:
            print("• Small training set → Aggressive transition")
        elif training_set_size and training_set_size > 500:
            print("• Large training set → More conservative transition")

        print("=" * 50)


# Global instance
threshold_calculator = AdaptiveThresholdCalculator()
