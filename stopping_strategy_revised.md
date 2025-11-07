# Stopping Strategy - Patience-Based Approach

## Core Idea: Stop When Learning Plateaus

Instead of arbitrary thresholds, stop when the model stops improving (with stability checks).

---

## Proposed Configurations

### LABEL Mode (F1 Optimization)

**Goal:** Balanced model for auto-labeling

```python
stopping_strategy = 'label'

# What to watch
metric = 'f1'
window = 3  # Rolling average window

# Stopping criteria
patience = 5  # Stop after 5 cycles with no improvement
min_delta = 0.02  # Improvement means increase of at least 2pp
min_cycles = 10  # Don't stop before cycle 10

# Stability check
require_stable = True  # Must be stable when stopping
std_threshold = 0.08  # Stable means std < 0.08 over window
```

**Stopping Logic:**
```
Stop when ALL of:
1. No improvement in rolling_avg(F1) for 5 cycles (delta < 0.02)
2. At least 10 cycles completed
3. rolling_std(F1) < 0.08 (stable performance)
```

---

### SEARCH Mode (Recall Optimization with Precision Floor)

**Goal:** High recall for finding positives, but avoid complete FP explosion

```python
stopping_strategy = 'search'

# What to watch
primary_metric = 'recall'
constraint_metric = 'precision'
window = 3

# Stopping criteria
patience = 5
min_delta = 0.02
min_cycles = 10

# Precision floor (prevent FP explosion)
precision_floor = 'auto'  # max(0.30, initial_precision - 0.1)
# Or: precision_floor = 0.30  # Absolute minimum
```

**Stopping Logic:**
```
Stop when ALL of:
1. No improvement in rolling_avg(recall) for 5 cycles (delta < 0.02)
2. At least 10 cycles completed
3. rolling_avg(precision) >= precision_floor
4. rolling_std(recall) < 0.10 (slightly more tolerant than label mode)
```

**Precision Floor Options:**
- `'auto'`: Set to `max(0.30, initial_precision - 0.1)`
  - Prevents dropping too far from initial performance
  - Hard floor at 0.30 (70% FP rate is max tolerable)
- `float`: User-specified absolute minimum

---

## Why This Works Better

### 1. **Dataset Agnostic**
- No need to know if 0.6 or 0.8 is "good" for your problem
- Works on easy and hard datasets
- Adapts to class balance

### 2. **Handles Oscillations**
- Rolling average smooths out cycle-to-cycle variance
- Stability check ensures consistent performance
- Won't stop during unstable periods

### 3. **Avoids Early Plateaus**
- `min_cycles` prevents stopping too early
- `min_delta` tolerates small fluctuations
- Can escape local plateaus

### 4. **No Arbitrary Thresholds**
- User doesn't need domain expertise
- "Stop when it stops getting better" is intuitive

---

## Edge Cases & Safeguards

### 1. Early Plateau (Cycle 8, No More Progress)

**Scenario:** Model plateaus at F1=0.50, no improvement after cycle 8

**Handling:**
```python
max_cycles = 50  # Hard limit

if no_improvement_cycles >= patience and cycle >= min_cycles:
    if current_metric < reasonable_minimum:
        warning(f"Stopped early with low {metric}={current_metric:.3f}")
        warning("Consider: different class_weighting, more initial data, or data quality issues")
    stop()
```

### 2. Never Stabilizes (High Variance Throughout)

**Scenario:** F1 oscillates wildly (0.4 → 0.7 → 0.5 → 0.6...)

**Handling:**
```python
if cycle >= max_cycles:
    warning(f"Reached max_cycles without stability (std={current_std:.3f})")
    warning("Consider: ensemble training, different selection_mode, or data issues")
    stop()

# Alternative: Relax stability requirement after patience exhausted
if no_improvement_cycles >= patience * 2:
    # Stop even if not stable
    warning("Stopping without stability due to lack of improvement")
    stop()
```

### 3. Precision Floor Violated (SEARCH mode)

**Scenario:** Recall improves but precision drops below floor

**Handling:**
```python
if rolling_avg(precision) < precision_floor:
    # Don't stop - keep training to recover precision
    no_improvement_cycles = 0  # Reset patience counter

    if cycles_below_floor >= 10:
        warning(f"Precision stuck below floor ({precision:.3f} < {floor})")
        warning("Consider: higher class_weighting, different selection_mode")
```

---

## Comparison: Threshold vs Patience

### Threshold Approach (Original)
```python
# LABEL: Stop when F1 > 0.65 and stable
if rolling_avg(f1) > 0.65 and rolling_std(f1) < 0.08:
    stop()
```

**Problems:**
- What if dataset is hard? May never reach 0.65
- What if dataset is easy? Wastes cycles after 0.65
- User needs to know "good" threshold

### Patience Approach (Revised)
```python
# LABEL: Stop when F1 stops improving and stable
if no_improvement_for(5) and stable():
    stop()
```

**Benefits:**
- Adapts to dataset difficulty
- Stops as soon as progress plateaus
- Intuitive for users

---

## Configuration Examples

### Alpha Release (Simple)

```bash
# LABEL mode - optimize F1, stop when plateaus
audioloop train \
  --stopping-strategy label \
  --patience 5 \
  --min-cycles 10 \
  --max-cycles 50

# SEARCH mode - optimize recall with precision floor
audioloop train \
  --stopping-strategy search \
  --patience 5 \
  --min-cycles 10 \
  --precision-floor 0.30
```

### Post-Alpha (Advanced)

```yaml
# config.yaml
stopping:
  strategy: label  # or 'search' or 'custom'

  # Patience settings
  patience: 5
  min_delta: 0.02
  min_cycles: 10
  max_cycles: 50

  # Metric settings
  window: 3  # Rolling average window
  std_threshold: 0.08  # Stability requirement

  # SEARCH mode only
  precision_floor: auto  # or float like 0.30
```

```yaml
# Custom strategy example
stopping:
  strategy: custom
  metric: precision  # Optimize precision instead
  patience: 7
  min_delta: 0.01
  secondary_metric: recall
  secondary_floor: 0.60  # Recall must stay above 0.60
```

---

## Implementation Details

### Tracking State

```python
class StoppingTracker:
    def __init__(self, config):
        self.metric = config.metric
        self.window = config.window
        self.patience = config.patience
        self.min_delta = config.min_delta

        self.history = []  # List of metric values
        self.best_rolling_avg = -inf
        self.no_improvement_cycles = 0

    def update(self, cycle, metrics):
        self.history.append(metrics[self.metric])

        if len(self.history) < self.window:
            return False  # Not enough data yet

        # Calculate rolling average
        rolling_avg = mean(self.history[-self.window:])
        rolling_std = std(self.history[-self.window:])

        # Check for improvement
        improvement = rolling_avg - self.best_rolling_avg

        if improvement >= self.min_delta:
            self.best_rolling_avg = rolling_avg
            self.no_improvement_cycles = 0
        else:
            self.no_improvement_cycles += 1

        # Check stopping criteria
        should_stop = (
            cycle >= self.min_cycles and
            self.no_improvement_cycles >= self.patience and
            rolling_std < self.std_threshold
        )

        return should_stop
```

### Output Display

```
Cycle 15/50
==========================================
Candidate Metrics (n=50):
  F1:        0.683  (rolling avg: 0.688, std: 0.024)
  Precision: 0.708
  Recall:    0.660

Stopping Criteria (LABEL mode):
  Metric: F1 (target: maximize)
  Best rolling avg: 0.702 (cycle 12)
  Current rolling avg: 0.688
  No improvement: 3/5 cycles
  Stable: ✓ (std=0.024 < 0.08)

Status: Continue training (3 more cycles without improvement until stop)

Full Dataset Metrics (n=10000):
  F1: 0.721  (↑ 0.033 from candidate)
==========================================
```

When stopping:
```
Cycle 18/50
==========================================
Candidate Metrics (n=50):
  F1:        0.695  (rolling avg: 0.692, std: 0.021)
  Precision: 0.715
  Recall:    0.671

Stopping Criteria (LABEL mode):
  Metric: F1 (target: maximize)
  Best rolling avg: 0.702 (cycle 12)
  Current rolling avg: 0.692
  No improvement: 5/5 cycles ✓
  Stable: ✓ (std=0.021 < 0.08)

🎯 STOPPING: No improvement for 5 cycles and stable performance

Final Performance:
  Best F1: 0.702 (cycle 12)
  Final F1: 0.692 (cycle 18)

Saving best model from cycle 12...
==========================================
```

---

## Recommended Default Values

Based on siren-test analysis:

### LABEL Mode
```python
patience = 5  # Enough to escape small plateaus
min_delta = 0.02  # 2 percentage points (significant improvement)
min_cycles = 10  # Let model learn basics first
window = 3  # Smooth over 3 cycles
std_threshold = 0.08  # Seen in stable entropy runs
max_cycles = 50  # Safety limit
```

### SEARCH Mode
```python
patience = 5
min_delta = 0.02  # Same as label
min_cycles = 10
window = 3
std_threshold = 0.10  # More tolerant of variance
precision_floor = 'auto'  # max(0.30, initial_precision - 0.1)
max_cycles = 50
```

---

## Open Questions

### 1. Should we save "best" model separately from "final"?

**Proposal:** Yes
- Best model: Highest rolling average during training
- Final model: Model at stopping point
- User can choose which to use

**Use case:** If model slightly degrades before patience exhausted, best model is better

### 2. What if precision drops catastrophically in SEARCH mode?

**Example:** Cycle 12 recall=0.99, precision=0.05 (predicting everything as positive)

**Options:**
a) Hard stop training - something went wrong
b) Reset patience and continue - maybe it recovers
c) Revert to previous model

**Recommendation:** (a) - stop with error message, suggest adjusting parameters

### 3. Should min_delta be relative or absolute?

**Absolute (current proposal):** delta > 0.02
- Simple to understand
- Works across most F1/recall ranges

**Relative:** delta > 0.02 * current_metric
- Better for very low/high metrics
- More complex

**Recommendation:** Absolute for alpha, consider relative post-alpha

### 4. Allow asymmetric patience?

**Example:** More patient for improvements, less patient for plateaus
- Improvement: reset counter to 0
- Small degradation: increment by 0.5
- Plateau: increment by 1

**Recommendation:** KISS for alpha, one patience value

---

## Summary

### For Alpha Release:

**Two presets with patience-based stopping:**

```python
# LABEL: Stop when F1 plateaus
stopping_strategy = 'label'
metric = 'f1'
patience = 5
min_delta = 0.02
min_cycles = 10

# SEARCH: Stop when recall plateaus (with precision floor)
stopping_strategy = 'search'
metric = 'recall'
patience = 5
min_delta = 0.02
min_cycles = 10
precision_floor = max(0.30, initial_precision - 0.1)
```

**Key advantages:**
- No arbitrary thresholds to tune
- Adapts to dataset difficulty
- Intuitive "stop when it stops improving" logic
- Safeguards against instability and early stopping

**Next step:** Implement StoppingTracker class and integrate into training loop
