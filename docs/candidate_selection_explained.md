# How Active Learning Candidates Are Selected

This document explains the candidate selection strategy used in AudioLoop's active learning pipeline.

## Overview

The active learning system selects audio samples for human labeling based on the model's predictions. The goal is to choose samples that will be most informative for improving the model.

## Selection Process

### 1. Model Inference
First, the trained model runs inference on all available audio samples, producing for each sample:
- **Prediction**: Whether it's the target class (e.g., "siren") or not ("not_siren")
- **Confidence**: The maximum probability from the softmax output (0.0 to 1.0)
- **Entropy**: A measure of uncertainty in the prediction

### 2. Confidence Calculation
```python
# For a binary classifier with 2 classes:
probabilities = softmax(model_output)  # e.g., [0.2, 0.8]
confidence = max(probabilities)         # e.g., 0.8
```

Higher confidence means the model is more certain about its prediction.

### 3. Entropy Calculation
```python
# Entropy measures uncertainty
entropy = -sum(p * log(p) for p in probabilities)
```

- **Low entropy** (near 0): Model is very certain (e.g., [0.01, 0.99])
- **High entropy** (near 0.69 for binary): Model is uncertain (e.g., [0.5, 0.5])

### 4. Candidate Selection Strategy

The system selects candidates using a **high-confidence approach with randomization**:

1. **Randomize samples first** to prevent bias when many have identical confidence (e.g., 1.0)
2. Sort by confidence (highest first)
3. Try to find samples with confidence ≥ 0.8 (default threshold)
4. If enough high-confidence samples exist, select the top N
5. If not enough, take all high-confidence samples + next highest until we have N

This process is applied separately to positive and negative predictions.

**Important:** The randomization step prevents systematic bias when many samples have identical confidence scores, ensuring diverse selection from across the dataset rather than always picking from the same files/folds.

#### Default Parameters:
- **num_positive**: 10 samples predicted as target class
- **num_negative**: 10 samples predicted as non-target class
- **min_confidence**: 0.8 threshold for "high confidence"

## Why This Strategy?

### 1. **High Confidence Samples**
These are samples the model thinks it knows well. Getting human labels for these helps:
- Verify the model is learning correctly
- Catch systematic errors (e.g., model confidently wrong)
- Build trust in model predictions

### 2. **Balanced Selection**
By selecting both positive and negative predictions:
- Ensures diverse training data
- Prevents class imbalance in the expanded training set
- Helps model learn both what the target is AND isn't

### 3. **Adaptive Thresholds**
If there aren't enough high-confidence samples:
- Still selects the most confident available samples
- Ensures consistent number of labels per cycle
- Adapts to model's current performance level

## Example Selection

Given predictions for "dog_bark" detection:
```
Filename            Prediction    Confidence    Selected?
dog1.wav           dog_bark      0.95          ✓ (high conf positive)
dog2.wav           dog_bark      0.88          ✓ (high conf positive)
cat1.wav           not_dog_bark  0.92          ✓ (high conf negative)
dog3.wav           dog_bark      0.76          ✓ (if need more positives)
ambiguous1.wav     dog_bark      0.51          ✗ (too low confidence)
```

## Customizing Selection

You can adjust the selection strategy:

```bash
python -m audioloop.active_learning \
    --class-name dog_bark \
    --total-candidates 25 \     # Select 25 total candidates
    --positive-pct 0.8 \        # 80% positive, 20% negative
    --min-confidence 0.9        # Require higher confidence
```

## Why Randomization Matters

When training on limited data, models often become overconfident, assigning confidence 1.0 to many samples. Without randomization:
- The same samples (e.g., from fold1) would always be selected first
- You'd miss diversity from other parts of the dataset
- The model might not improve on underrepresented data

The randomization ensures that when multiple samples have identical scores, they're selected randomly rather than in file order.

## Alternative Strategies (Not Currently Implemented)

Other active learning strategies that could be added:
1. **Uncertainty Sampling**: Select samples with highest entropy
2. **Margin Sampling**: Select samples closest to decision boundary
3. **Diversity Sampling**: Ensure selected samples are different from each other
4. **Expected Model Change**: Select samples that would most change the model

The current high-confidence strategy with randomization is simple and effective for audio classification tasks where human verification of confident predictions helps build reliable models.