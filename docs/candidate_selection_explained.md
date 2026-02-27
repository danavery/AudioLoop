# Candidate Selection

After each training cycle, AudioLoop selects a set of candidates from the unlabeled dataset for human review. The selection strategy determines *which* clips are chosen, and will significantly affect how quickly the model improves.

## How Selection Works

1. The trained model runs inference on all unlabeled audio, producing per-file predictions, confidence scores, and entropy values
2. A pool of the top-scoring candidates is created (pool size = `total_candidates` * `candidate_pool_multiplier`)
3. The final candidates are randomly sampled from that pool

The pool-then-sample approach adds diversity: when many clips have similar scores, random sampling prevents always picking the same files.

**Key concepts:**
- **Confidence**: The model's certainty in its prediction (0.0-1.0). High confidence = the model is sure.
- **Entropy**: A measure of uncertainty. High entropy = near the decision boundary (the model can't decide). For binary classification, entropy ranges from 0 (certain) to ~0.69 (completely uncertain).

## Strategies

### Entropy (`entropy`) — Default

Selects the most uncertain clips — those closest to the model's decision boundary. This is the standard active learning approach: the clips the model struggles with are the most informative for training.

### Confidence (`confidence`)

Selects the clips the model is most confident about. Useful early on to verify the model is learning correctly, but can lead to redundant selections once the model becomes overconfident. For that reason it tends not to work well after the first few cycles.

### Basic Transition (`basic_transition`)

Automatically switches from confidence to entropy based on model performance. Starts with confidence-based selection, then transitions to entropy when all three conditions are met:
- F1 score exceeds a threshold (model has learned basic patterns)
- Mean confidence exceeds a threshold (model is generally confident)
- Confidence std dev falls below a threshold (overconfidence risk)

Thresholds can be set manually or auto-calculated from dataset characteristics (`--auto-thresholds`). Auto-thresholds adjust for class imbalance — rare classes get more aggressive transition criteria.

### Mixed Entropy (`mixed_entropy`) — Experimental

Samples from three entropy tiers: 70% high-entropy (near boundary), 20% medium, 10% low. The idea is that including some confident predictions alongside boundary cases may produce more stable candidate metrics, though at the cost of lower peak performance. Incompatible with `positive_percentage` stratification.

### Stratified Uncertainty (`stratified_uncertainty`)

Entropy-based selection that guarantees a target ratio of predicted-positive to predicted-negative candidates. Useful when you want uncertainty sampling but need to ensure both classes are represented.

### Random (`random`)

Pure random baseline. Ignores model outputs entirely. Useful for measuring whether the active selection strategies are actually providing value.

## Configuration

Set in `audioloop.yaml`:

```yaml
selection_mode: entropy              # Strategy (see above)
total_candidates: 50                 # How many candidates per cycle
candidate_pool_multiplier: 5         # Pool = candidates * multiplier
positive_percentage: null            # null = no stratification, or 0.0-1.0
min_confidence: 0.8                  # Confidence threshold (used by confidence strategy)
```

### Stratification

By default (`positive_percentage: null`), candidates are selected purely by the strategy's scoring metric without regard to predicted class. Setting `positive_percentage` forces a specific ratio. For example, `positive_percentage: 0.6` means 60% of candidates will be predicted-positive.

This is useful for very imbalanced datasets where pure entropy sampling might return almost no predicted positives. However, it biases candidate metrics (see [cycle_stopping_criteria.md](cycle_stopping_criteria.md#stratification-effects)), so leave it null if you're using cycle stopping.

## See Also
- [User Manual: Candidate Selection](user_manual.md#candidate-selection) — overview and key parameters
- [Extending AudioLoop](extending.md) — adding custom selection strategies
