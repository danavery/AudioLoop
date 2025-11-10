# How Active Learning Candidates Are Selected

This document explains the candidate selection strategy used in AudioLoop's active learning pipeline.

## Overview

The active learning system selects audio samples for human labeling based on the model's predictions and confidence levels. The goal is to choose samples that will be most informative for improving the model.

## Clean Architecture

The system maintains clean separation between inference and analysis:

### Active Learning Pipeline (Inference Focus)
- **Process**: Runs model inference on all available audio files
- **Output**: Complete prediction record with confidence levels
- **Selection**: Chooses candidates based on model outputs (confidence, entropy)
- **Focus**: Lean, fast inference without statistical analysis

### Metrics Pipeline (Analysis Focus)  
- **Input**: Prediction files from active learning iterations
- **Process**: Comprehensive performance analysis and trend tracking
- **Output**: Accuracy metrics, plots, and insights across versions
- **Focus**: Rich evaluation separate from candidate selection

### What's Used for Candidate Selection
- Model predictions (positive/negative)
- Confidence scores (0.0 to 1.0)
- Entropy values (uncertainty measure)
- Prediction probabilities for each class

## Selection Process

### 1. Model Inference
First, the trained model runs inference on all available audio samples, producing for each sample:
- **Prediction**: Whether it's the target class (e.g., "siren") or not ("not_siren")
- **Confidence**: The maximum probability from the softmax output (0.0 to 1.0)
- **Entropy**: A measure of uncertainty in the prediction
- **Probabilities**: Raw probabilities for positive and negative classes

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

### 4. Candidate Selection Strategies

AudioLoop supports multiple candidate selection strategies:

#### 4.1 High-Confidence Strategy
The system selects candidates using a **high-confidence approach with randomization**:

1. **Randomize samples first** to prevent bias when many have identical confidence (e.g., 1.0)
2. Sort by confidence (highest first)
3. Try to find samples with confidence ≥ 0.8 (default threshold)
4. If enough high-confidence samples exist, select the top N
5. If not enough, take all high-confidence samples + next highest until we have N

This process is applied separately to positive and negative predictions.

**Selection Inputs**: The process uses only model outputs:
- Model confidence scores
- Model predictions (positive/negative)
- Entropy values (uncertainty measure)
- Random sampling for diversity

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

#### 4.2 Entropy-Based Strategy
The entropy-based strategy selects samples with **highest uncertainty**:

1. Sort by entropy (highest first = most uncertain)
2. Select from high-entropy samples for each class
3. Focus on samples near decision boundaries
4. Useful when model becomes overconfident

#### 4.3 Basic Transition Strategy
The basic transition strategy **automatically switches** from confidence to entropy based on model performance:

**Initial Phase**: Uses confidence-based selection (same as 4.1)

**Transition Criteria**: Switches to entropy-based when ALL three conditions are met:
- **F1 Score > threshold**: Model has learned basic patterns
- **Mean Confidence > threshold**: Model shows confidence in predictions
- **Std Confidence < threshold**: Uncertainty is decreasing (overconfidence risk)

**Adaptive Thresholds**: The system can automatically calculate optimal thresholds based on dataset characteristics:
- **Class Imbalance**: Rare classes (< 5%) get more aggressive thresholds
- **Training Set Size**: Smaller sets trigger earlier transitions
- **No Ground Truth Required**: Only needs estimated positive class percentage

**Analysis Output**:
```
🎯 Adaptive Threshold Analysis:
==================================================
Estimated positive class prevalence: 5.0%
Training set size: 80

📊 Calculated Thresholds:
F1 threshold: 0.141
Confidence threshold: 0.820
Variance threshold: 0.200

💡 Rationale:
• Rare class (< 5%) → Lower F1 threshold, aggressive confidence
==================================================

Basic Transition Analysis:
  ✓ F1 Score: 0.265 (>0.141 required)
  ✓ Mean Confidence: 0.951 (>0.820 required) 
  ✓ Std Confidence: 0.104 (<0.200 required)
  → Using ENTROPY-based selection (uncertainty sampling)
```

**Usage**:
```bash
# Use basic transition with auto-calculated thresholds (recommended for imbalanced datasets)
python -m audioloop.active_learning --class-name siren --selection-mode basic_transition --auto-thresholds --estimated-positive-pct 0.10

# Use auto-thresholds with default estimate (5%)
python -m audioloop.active_learning --class-name siren --selection-mode basic_transition --auto-thresholds

# Use basic transition with default thresholds
python -m audioloop.active_learning --class-name siren --selection-mode basic_transition

# Custom manual thresholds
python -m audioloop.active_learning --class-name siren --selection-mode basic_transition \
  --basic-transition-f1-threshold 0.25 \
  --basic-transition-confidence-threshold 0.95 \
  --basic-transition-variance-threshold 0.10
```

#### 4.4 Mixed-Entropy Strategy (Experimental)
The mixed-entropy strategy selects candidates **across multiple entropy levels** to balance learning value with representative evaluation.

**Note**: Experimental results show this strategy typically achieves lower peak performance than pure entropy, but may provide more stable candidate metrics in some scenarios.

**How it works**:
1. Divide predictions into three entropy buckets:
   - **High entropy** (top 20%): Most uncertain, near decision boundary
   - **Medium entropy** (next 40%): Moderate uncertainty
   - **Low entropy** (bottom 40%): Most confident predictions

2. Sample from each bucket with fixed ratios:
   - **70%** from high-entropy bucket (e.g., 35 of 50 candidates)
   - **20%** from medium-entropy bucket (e.g., 10 of 50 candidates)
   - **10%** from low-entropy bucket (e.g., 5 of 50 candidates)

3. Pool multiplier (default: 5x) maintains diversity within each bucket

**Why mixed-entropy?**
- **More stable metrics**: Including varied difficulty levels reduces volatility in candidate-set performance metrics
- **Better corpus correlation**: Representative sampling better reflects full dataset performance
- **Early warning system**: Low-entropy samples catch if model breaks on "easy" examples
- **Balanced learning**: Still prioritizes boundary cases (70% high) while maintaining evaluation quality

**Comparison to pure entropy sampling**:
- Pure entropy: 100% from decision boundary → volatile metrics, poor corpus correlation
- Mixed entropy: Mostly boundary cases (70%) + representative samples (30%) → stable metrics, better tracking

**Small dataset handling** (< 100 predictions):
- Falls back to pure high-entropy sampling with reduced pool multiplier
- Ensures focus on informative examples when data is limited

**Usage**:
```bash
# Explicit specification (experimental)
python -m audioloop.active_learning --class-name siren --selection-mode mixed_entropy

# Adjust pool multiplier for more/less diversity
python -m audioloop.active_learning --class-name siren \
  --selection-mode mixed_entropy \
  --candidate-pool-multiplier 10
```

**Note**: Mixed-entropy is incompatible with `--positive-pct` stratification. The strategy naturally samples across prediction types based on entropy distribution.

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