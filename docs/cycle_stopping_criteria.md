# Cycle Stopping Criteria Guide

This guide explains the cross-cycle stopping criteria system for automatically determining when to stop active learning cycles based on candidate performance metrics.

## Overview

Active learning cycles iteratively train models and select new candidates for labeling. Cycle stopping criteria automatically detect when additional cycles are unlikely to improve model quality, saving labeling effort and computational resources.

**Key distinction:**
- **Training stopping criteria** (see `stopping_criteria_guide.md`): Control when to stop training epochs within a single model
- **Cycle stopping criteria** (this guide): Control when to stop the entire active learning process across multiple cycles

## When to Use Cycle Stopping

### ✅ Good Use Cases:
- Auto-labeling workflows where you want to find the best labeling model
- Iterative model improvement with limited labeling budget
- Experimentation to understand convergence patterns

### ⚠️ Limitations:
- **Candidate metrics are volatile** - Based on only ~50 high-entropy examples per cycle
- **May not correlate with corpus performance** - Boundary examples don't always represent overall quality
- **Best used as a guideline** - Validate with held-out test data when possible
- **Experimental feature** - Still under active development

## Built-in Strategies

### None (Default)
```bash
python -m audioloop.automated_workflow --cycle-stopping-strategy none
```

No automatic stopping. Runs for the number of cycles you specify. This is the safest option.

**When to use:**
- When you don't trust automated stopping
- When you have a fixed labeling budget
- When experimenting with different cycle counts

### Label Mode
```bash
python -m audioloop.automated_workflow --cycle-stopping-strategy label
```

Optimizes **F1 score** on labeled candidates. Designed for auto-labeling workflows where you need balanced precision and recall.

**How it works:**
1. Calculates F1 score on labeled candidates each cycle (comparing model predictions to human labels)
2. Tracks rolling average F1 over a window (default: 3 cycles)
3. Monitors for improvement using patience mechanism (default: 5 cycles without improvement)
4. Stops when:
   - Minimum cycles reached (default: 10)
   - AND F1 rolling average stable (std dev < 0.08)
   - AND no improvement for `patience` cycles
5. Saves the best model (highest rolling F1) as `model_best.pt`

**Configuration parameters:**
```python
# In audioloop/config.py or via CLI
cycle_stopping_strategy = "label"
cycle_patience = 5            # Cycles without improvement before stopping
cycle_min_delta = 0.02        # Minimum improvement threshold
cycle_min_cycles = 10         # Minimum cycles before considering stopping
cycle_window = 3              # Rolling average window size
cycle_std_threshold = 0.08    # Stability threshold (std dev)
```

**Example output:**
```
📈 Stopping (label mode):
   F1 (current): 0.545
   F1 (rolling avg): 0.520 (window=3)
   F1 (stability): std=0.045 (threshold=0.08)
   Patience: 3/5
   Best cycle: 14 (rolling avg=0.567)

   Confusion Matrix (50 labeled candidates):
                  Predicted
               Positive  Negative
    Actual Pos      15         3     (recall: 0.833)
           Neg      10        22
                -------  -------
    Precision   0.600
```

### Search Mode
```bash
python -m audioloop.automated_workflow --cycle-stopping-strategy search
```

Optimizes **recall** while maintaining a **precision floor**. Designed for finding rare positives where you're willing to accept some false positives.

**How it works:**
1. Calculates recall and precision on labeled candidates each cycle
2. Tracks rolling average recall over a window (default: 3 cycles)
3. Establishes precision floor: `max(0.30, cycle_1_precision - 0.1)`
4. Stops when:
   - Minimum cycles reached (default: 10)
   - AND recall rolling average stable (std dev < 0.10)
   - AND precision >= floor
   - AND no improvement for `patience` cycles
5. Saves the best model (highest rolling recall) as `model_best.pt`

**Configuration parameters:**
```python
# In audioloop/config.py or via CLI
cycle_stopping_strategy = "search"
cycle_patience = 5            # Cycles without improvement before stopping
cycle_min_delta = 0.02        # Minimum improvement threshold
cycle_min_cycles = 10         # Minimum cycles before considering stopping
cycle_window = 3              # Rolling average window size
precision_floor = "auto"      # Auto-calculated or fixed value (0.0-1.0)
```

**Example output:**
```
📈 Stopping (search mode):
   Recall (current): 0.706
   Recall (rolling avg): 0.680 (window=3)
   Precision (rolling avg): 0.444 (floor=0.300)
   Recall (stability): std=0.078 (threshold=0.10)
   Patience: 2/5
   Best cycle: 19 (rolling recall=0.705)

   Confusion Matrix (50 labeled candidates):
                  Predicted
               Positive  Negative
    Actual Pos      12         5     (recall: 0.706)
           Neg       8        25
                -------  -------
    Precision   0.600
```

## Configuration Options

### Config File (`audioloop/config.py`)

```python
from audioloop.config import AudioLoopConfig

config = AudioLoopConfig(
    # Strategy selection
    cycle_stopping_strategy="label",  # "none", "label", or "search"

    # Stopping parameters
    cycle_patience=5,          # Cycles to wait without improvement
    cycle_min_delta=0.02,      # Minimum improvement threshold (2%)
    cycle_min_cycles=10,       # Minimum cycles before stopping allowed
    cycle_window=3,            # Rolling average window
    cycle_std_threshold=0.08,  # Stability threshold for label mode

    # Search mode specific
    precision_floor="auto",    # "auto" or float (0.0-1.0)
)
```

### CLI Arguments

```bash
python -m audioloop.automated_workflow \
    --cycle-stopping-strategy label \
    --dataset fsd50k \
    --positive-class "Alarm" \
    --experiment-name my_experiment
```

**Available CLI flags:**
- `--cycle-stopping-strategy`: Choose "none", "label", or "search"
- Other cycle parameters use config defaults (modify `config.py` to change)

## How Candidate Metrics Work

### Candidate Selection
Each cycle, active learning selects ~50 "candidates" - examples the model is most uncertain about. These are labeled (by humans or auto-labeling) and added to the training set.

### Metric Calculation
After labeling, we calculate metrics by comparing:
- **Model predictions** (what the model thought before labeling)
- **Human labels** (ground truth for these specific examples)

**Important:** We never use full corpus ground truth for stopping decisions (it wouldn't be available in production).

### Metrics Tracked

For each cycle, we store in `candidate_metrics_history.json`:
```json
{
  "15": {
    "cycle": 15,
    "f1_score": 0.545,
    "precision": 0.405,
    "recall": 0.833,
    "accuracy": 0.74,
    "num_candidates": 50,
    "true_positives": 15,
    "false_positives": 22,
    "true_negatives": 10,
    "false_negatives": 3
  }
}
```

### Rolling Statistics
To reduce noise, stopping criteria use rolling averages:
- **Window size** (default: 3): Averages over last N cycles
- **Skips missing cycles**: If cycle data is missing, it's ignored
- **Waits for full window**: Requires at least `window` cycles before checking

## Understanding the Confusion Matrix

The confusion matrix shows model performance on the labeled candidates:

```
Confusion Matrix (50 labeled candidates):
                   Predicted
                Positive  Negative
     Actual Pos      15         3     (recall: 0.833)
            Neg      10        22
                 -------  -------
     Precision   0.600
```

**Reading the matrix:**
- **True Positives (15)**: Model correctly predicted positive
- **False Negatives (3)**: Model missed these positives
- **False Positives (10)**: Model incorrectly predicted positive
- **True Negatives (22)**: Model correctly predicted negative
- **Recall (0.833)**: Of 18 actual positives, model found 15 (83%)
- **Precision (0.600)**: Of 25 predicted positives, 15 were correct (60%)

## Known Limitations

### 1. Small Sample Size
Candidate metrics are based on only ~50 examples per cycle. This creates high variance:
- Single cycle metrics can swing wildly (F1: 0.15 → 0.55 → 0.30)
- Rolling averages help but don't eliminate noise
- Even with smoothing, correlation with corpus metrics is imperfect

### 2. Boundary Sampling Bias
Candidates are selected via uncertainty sampling (high-entropy examples). These are deliberately the **hardest** examples:
- Model performs worse on candidates than on the full corpus
- As model improves, it finds even harder boundary cases
- Candidate performance doesn't always track corpus performance

Example from real experiment:
```
Cycle | Candidate F1 | Corpus F1
------|--------------|----------
  13  |    0.542     |   0.681
  19  |    0.545     |   0.761  ← Candidate F1 flat, corpus improving
  29  |    0.537     |   0.772  ← Best corpus F1!
```

### 3. Stratification Effects
If using `positive_percentage` (stratified sampling), candidates are forced to have a specific ratio of predicted-positives to predicted-negatives. This creates additional bias.

**Recommendation:** Use `positive_percentage=None` (default) for pure entropy sampling.

### 4. No Ground Truth Access
Stopping decisions must be made without access to corpus-level ground truth (since production datasets won't have labels). This limits what signals we can use.

## Best Practices

### 1. Start with Fixed Cycles
```bash
# Run 20-30 cycles without automatic stopping
python -m audioloop.automated_workflow --cycle-stopping-strategy none --cycles 25
```

Observe the candidate metrics patterns in your specific domain before trusting automated stopping.

### 2. Use Stopping as a Guideline
```bash
# Let it run but don't blindly trust the "best" model
python -m audioloop.automated_workflow --cycle-stopping-strategy label
```

After stopping, examine the metrics history and potentially test models from different cycles.

### 3. Validate with Held-Out Data
If you have labeled test data, evaluate multiple cycles:
```python
# Load and evaluate different cycle models
for cycle in [10, 15, 20, 25]:
    model = load_model(f"model_v{cycle}.pt")
    f1 = evaluate_on_test_set(model, test_data)
    print(f"Cycle {cycle}: F1 = {f1:.3f}")
```

### 4. Monitor the Metrics
The automated workflow displays detailed metrics each cycle. Watch for:
- ✅ Smooth improvement in rolling averages
- ✅ Stable confusion matrix patterns
- ⚠️ Wild swings in single-cycle metrics
- ⚠️ Very low candidate counts for a class
- ⚠️ Declining corpus-level predictions ratio

### 5. Consider Dataset Characteristics
- **Highly imbalanced** (1% positive): Candidate metrics will be noisier
- **Larger candidate sets** (100+ per cycle): More stable metrics
- **Clearer boundaries**: Better metric stability
- **Ambiguous classes**: Higher variance

### 6. Ensemble Training (Future)
When ensemble training is implemented, candidate metrics should become more stable. Consider revisiting automated stopping at that point.

## Experimental: Mixed Sampling

**Status:** Not yet implemented, under consideration

Current pure high-entropy sampling focuses only on boundary cases. Mixed sampling would include:
- 60-70% high-entropy (boundary cases for learning)
- 20-30% medium-entropy (challenging but not boundary)
- 10% low-entropy (confident predictions as sanity check)

This could improve metric stability and correlation with corpus performance.

## Troubleshooting

### Stopping too early?
- Increase `cycle_patience` (e.g., 10 instead of 5)
- Increase `cycle_min_cycles` (e.g., 15 instead of 10)
- Decrease `cycle_min_delta` (e.g., 0.01 instead of 0.02)
- Consider using `cycle_stopping_strategy=none`

### Never stopping?
- Check if candidate metrics are improving at all
- Verify minimum cycles threshold is reasonable
- Check if precision floor is too high (search mode)
- Review candidate metrics history JSON file

### Metrics very noisy?
- Increase `cycle_window` (e.g., 5 instead of 3)
- Set `positive_percentage=None` to remove stratification bias
- Increase candidates per cycle (e.g., 100 instead of 50)
- Wait for ensemble training implementation

### "Best" model not actually best?
- This is a known limitation - validate with held-out data
- Compare models from cycles near the stopping point
- Consider using fixed cycle counts instead of automated stopping

## Files and Paths

**Candidate metrics history:**
```
outputs/<experiment_name>/candidate_metrics_history.json
```

**Best model (when stopping triggered):**
```
outputs/<experiment_name>/model_best.pt
```

**All cycle models:**
```
outputs/<experiment_name>/model_v1.pt
outputs/<experiment_name>/model_v2.pt
...
```

## Implementation Details

### Code Structure
- **Metric calculation**: `audioloop/utils/metrics_utils.py::calculate_candidate_metrics()`
- **Metric persistence**: `audioloop/utils/candidate_metrics.py`
- **Stopping logic**: `audioloop/utils/cycle_stopping_criteria.py`
- **Integration**: `audioloop/merge_labels.py` and `audioloop/automated_workflow.py`

### Class Hierarchy
```python
CycleStoppingCriterion  # Base class
├── LabelModeStoppingCriterion  # Optimizes F1
└── SearchModeStoppingCriterion  # Optimizes recall + precision floor
```

### Factory Function
```python
from audioloop.utils.cycle_stopping_criteria import create_cycle_stopping_criterion

criterion = create_cycle_stopping_criterion(config, metrics_history)
if criterion and criterion.should_stop(current_cycle):
    best_cycle = criterion.get_best_cycle()
    # Copy model_v{best_cycle}.pt to model_best.pt
```

## Research and Future Work

### Open Questions
1. Can mixed sampling improve metric stability and corpus correlation?
2. Will ensemble training reduce metric volatility enough to make automated stopping reliable?
3. Are there alternative signals (entropy distribution, prediction stability) that better predict convergence?
4. Should we use different metrics (accuracy instead of F1) for stopping decisions?

### Planned Improvements
- [ ] Implement mixed-entropy sampling option
- [ ] Add ensemble training support
- [ ] Track additional signals (entropy distribution, prediction changes)
- [ ] Conduct systematic experiments on metric correlation
- [ ] Add median-based rolling statistics for outlier resistance
- [ ] Support custom stopping criterion implementations

## Related Documentation
- [Training Stopping Criteria](stopping_criteria_guide.md) - Within-training stopping
- [Candidate Selection Explained](candidate_selection_explained.md) - How candidates are chosen
- [Automated Workflow Guide](../README.md#automated-workflow) - Full active learning pipeline

## References
- Active Learning Literature: Settling 2009, Settles 2010
- Uncertainty Sampling: Lewis & Gale 1994
- Stopping Criteria Research: Bloodgood & Vijay-Shanker 2009
