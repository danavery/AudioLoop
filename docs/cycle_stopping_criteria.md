# Cycle Stopping Criteria

Cycle stopping criteria automatically detect when additional active learning cycles are unlikely to improve model quality, saving labeling effort and computation.

**Key distinction:** [Training stopping criteria](stopping_criteria_guide.md) control when to stop training epochs *within a single cycle*. Cycle stopping criteria control when to stop the *entire active learning process* across multiple cycles.

## Important Caveats

Candidate metrics are based on a small number of examples per cycle (default: 50 candidates selected by entropy/uncertainty). This means:

- Metrics are volatile — single-cycle F1 can swing from 0.15 to 0.55 to 0.30
- Candidate performance doesn't always track corpus performance (see [Boundary Sampling Bias](#boundary-sampling-bias) below)
- Automated stopping is best used as a guideline, not a hard rule

**Strong recommendation: Start with fixed cycle counts (`--cycle-stopping-strategy none`) until you understand the convergence patterns in your data, then come back to this document.**

## Strategies

### None (Default)

No automatic stopping. Runs for the number of cycles you specify.

```bash
python -m audioloop.automated_workflow --class-name siren --cycles 25 --cycle-stopping-strategy none
```

### Label Mode

Optimizes **F1 score** on labeled candidates. Designed for auto-labeling workflows where you need balanced precision and recall.

How it works:
1. Calculates F1 score on labeled candidates each cycle (model predictions vs human labels)
2. Tracks rolling average F1 over a window
3. Stops when all conditions are met:
   - Minimum cycles reached
   - F1 rolling average is stable (std dev < `cycle_std_threshold`)
   - No improvement for `cycle_patience` cycles
4. Saves the best model (highest rolling F1) as `model_best.pt`

```bash
python -m audioloop.automated_workflow --class-name siren --cycle-stopping-strategy label
```

### Search Mode

Optimizes **recall** while maintaining a **precision floor**. Designed for finding rare positives where you're willing to accept some false positives.

How it works:
1. Calculates recall and precision on labeled candidates each cycle
2. Tracks rolling average recall over a window
3. Establishes precision floor: `max(0.30, cycle_1_precision - 0.1)` (or a fixed value you set)
4. Stops when all conditions are met:
   - Minimum cycles reached
   - Recall rolling average is stable (std dev < 0.10)
   - Precision >= floor
   - No improvement for `cycle_patience` cycles
5. Saves the best model (highest rolling recall) as `model_best.pt`

```bash
python -m audioloop.automated_workflow --class-name siren --cycle-stopping-strategy search
```

### Churn Mode

**Label-free.** Instead of candidate metrics, churn mode watches how much the model's *predictions on the unlabeled pool* change between cycles. When the model stops changing its mind about the corpus, it has converged. This sidesteps the [Boundary Sampling Bias](#boundary-sampling-bias) that makes candidate F1 an unreliable convergence signal — pool churn is a whole-model property, not a measurement on the deliberately-hard candidate batch.

How it works:
1. Each cycle, compares the current predictions (`predictions_v{N}.csv`) to the previous cycle's on the shared pool, and computes **churn** = the fraction of clips whose predicted label flipped.
2. Tracks a rolling average of churn and its running peak.
3. Stops when the rolling churn has fallen to `<= churn_peak_frac` of its running peak for `churn_patience` consecutive cycles, after `cycle_min_cycles`. (Peak-*relative* rather than an absolute threshold, because the churn floor depends on pool size and class balance.)
4. Saves the model from the cycle where churn flattened as `model_best.pt`.

```bash
python -m audioloop.automated_workflow --class-name siren --cycle-stopping-strategy churn
```

**Tradeoff to know:** churn is a *leading* indicator — predictions stop flipping slightly *before* corpus quality fully peaks (the last gains come from confidence/margin shifts on clips that no longer flip label). So churn mode tends to stop a touch early, which is the safe direction. It is, however, far more reliable than candidate-metric stopping and cannot be fooled into a catastrophic early stop the way candidate F1 can on a weak model. Tune `churn_peak_frac` to trade reliability against how close to the plateau you stop: **lower** (e.g. 0.03) stops later/closer to the peak but may never trigger within budget; **higher** (e.g. 0.10–0.15) stops earlier. The default 0.05 is a reasonable middle.

## Configuration

Set these in `audioloop.yaml`. The strategy itself can also be set via `--cycle-stopping-strategy` on the CLI.

```yaml
# Strategy: "none", "label", "search", or "churn"
cycle_stopping_strategy: label

# Common parameters
cycle_patience: 5           # Cycles without improvement before stopping
cycle_min_delta: 0.02       # Minimum improvement to reset patience
cycle_min_cycles: 10        # Minimum cycles before stopping is allowed
cycle_window: 3             # Rolling average window size (also used by churn mode)
cycle_std_threshold: 0.08   # Max std dev for "stable" (label mode)

# Search mode only
precision_floor: auto       # "auto" or a fixed float (0.0-1.0)

# Churn mode only (label-free)
churn_peak_frac: 0.05       # Stop when rolling churn <= this fraction of its running peak
churn_patience: 2           # Consecutive cycles below threshold before stopping
```

Churn mode reuses `cycle_window` and `cycle_min_cycles`; it ignores `cycle_patience`/`cycle_min_delta`/`cycle_std_threshold` (those govern the candidate-metric strategies) and uses `churn_patience` instead.

## How Candidate Metrics Work

Each cycle, active learning selects candidates for labeling. After labeling, metrics are calculated by comparing model predictions to human labels. These metrics are stored in `outputs/<experiment_name>/candidate_metrics_history.json`.

Rolling averages over a configurable window (`cycle_window`) smooth out the per-cycle noise. The stopping criterion waits for a full window of data before checking, and skips any missing cycles.

Stopping uses a patience-based approach rather than fixed thresholds. This is dataset-agnostic — there's no need to know whether 0.6 or 0.8 is a "good" F1 for your problem. It just watches for when improvement stops.

## Known Limitations

### Boundary Sampling Bias

Candidates are selected via uncertainty sampling — they're deliberately the *hardest* examples. As the model improves, it finds even harder boundary cases. Candidate performance can plateau or decline even while corpus performance improves:

```
Cycle | Candidate F1 | Corpus F1
------|--------------|----------
  13  |    0.542     |   0.681
  19  |    0.545     |   0.761   <- Candidate F1 flat, corpus improving
  29  |    0.537     |   0.772   <- Best corpus F1
```

[Churn Mode](#churn-mode) avoids this bias entirely: it measures prediction stability over the whole pool rather than accuracy on the hard candidate batch, so it isn't misled when candidate F1 goes flat while the corpus is still improving.

### Stratification Effects

If using `positive_percentage` (stratified candidate sampling), candidates are forced to have a specific ratio of predicted-positives to predicted-negatives, which biases the metrics. Use `positive_percentage: null` (the default) for pure entropy sampling when relying on cycle stopping.

## Troubleshooting

**Stopping too early?**
- Increase `cycle_patience` (e.g., 10 instead of 5)
- Increase `cycle_min_cycles` (e.g., 15 instead of 10)
- Decrease `cycle_min_delta` (e.g., 0.01 instead of 0.02)

**Never stopping?**
- Check if candidate metrics are improving at all (review `candidate_metrics_history.json`)
- Verify `cycle_min_cycles` is reasonable
- Check if `precision_floor` is too high (search mode)

**Metrics very noisy?**
- Increase `cycle_window` (e.g., 5 instead of 3)
- Increase candidates per cycle (`total_candidates: 100` instead of 50)
- Ensure `positive_percentage` is null (default)

**"Best" model not actually best?**
- This is a known limitation of candidate-based metrics — validate with held-out data if available
- Compare models from several cycles near the stopping point

## Output Files

```
outputs/<experiment_name>/
    candidate_metrics_history.json   # Per-cycle metrics
    model_best.pt                    # Best model (when stopping triggered)
    model_v1.pt, model_v2.pt, ...   # All cycle models
```

## See Also
- [Training Stopping Criteria](stopping_criteria_guide.md) — within-cycle training stopping
- [Candidate Selection](candidate_selection_explained.md) — how candidates are chosen
