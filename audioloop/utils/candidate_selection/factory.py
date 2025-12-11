"""
Factory function for creating candidate selection strategies.

This module provides a config-driven factory for instantiating the appropriate
candidate selection strategy based on configuration parameters.
"""


def create_strategy(config):
    """
    Create a selection strategy from configuration.

    This factory encapsulates the logic for instantiating different strategy
    types and their specific configuration parameters.

    Args:
        config: AudioLoopConfig with selection strategy settings

    Returns:
        CandidateSelectionStrategy instance

    Raises:
        ValueError: If selection_mode is not recognized

    Example:
        >>> from audioloop.config import AudioLoopConfig
        >>> config = AudioLoopConfig(selection_mode="entropy")
        >>> strategy = create_strategy(config)
        >>> candidates = strategy.select_candidates(predictions, num_candidates=50)
    """
    from .basic_transition import BasicTransitionStrategy
    from .confidence import ConfidenceStrategy
    from .entropy import EntropyStrategy
    from .mixed_entropy import MixedEntropyStrategy
    from .random import RandomStrategy
    from .stratified import StratifiedUncertaintyStrategy

    # Build strategy-specific kwargs from config
    strategy_kwargs = {}

    if config.selection_mode == "basic_transition":
        strategy_kwargs = {
            "f1_threshold": None
            if config.auto_thresholds
            else config.basic_transition_f1_threshold,
            "confidence_threshold": None
            if config.auto_thresholds
            else config.basic_transition_confidence_threshold,
            "variance_threshold": None
            if config.auto_thresholds
            else config.basic_transition_variance_threshold,
            "auto_thresholds": config.auto_thresholds,
            "estimated_positive_pct": config.estimated_positive_pct
            or 0.05,  # Default 5% if not provided
        }
    # Other strategies use default constructors (no special kwargs needed)
    # Future strategy-specific configs can be added here as elif blocks

    # Map selection mode to strategy class
    strategies = {
        "confidence": ConfidenceStrategy,
        "entropy": EntropyStrategy,
        "mixed_entropy": MixedEntropyStrategy,
        "basic_transition": BasicTransitionStrategy,
        "stratified_uncertainty": StratifiedUncertaintyStrategy,
        "random": RandomStrategy,
    }

    if config.selection_mode not in strategies:
        # Should never happen due to config validation, but defensive check
        available = ", ".join(strategies.keys())
        raise ValueError(
            f"Unknown selection mode: '{config.selection_mode}'. Available: {available}"
        )

    return strategies[config.selection_mode](**strategy_kwargs)
