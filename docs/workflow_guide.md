# AudioLoop Workflow Guide

Complete guide to AudioLoop's active learning workflow patterns, best practices, and end-to-end processes. For specific command syntax, see [cli_reference.md](cli_reference.md).

## Overview

AudioLoop supports two fundamental workflow modes designed for different use cases:

### Production Mode (Default)
**Use case**: Real-world deployment with truly unlabeled datasets
- **Ground truth**: Not available - you don't know the true labels
- **Active learning**: Generates predictions without ground truth columns
- **Metrics tracking**: Shows prediction and confidence metrics only
- **Human labeling**: Manual review using web UI or terminal interface
- **Goal**: Build models for actual unknown audio classification tasks

### Evaluation Mode (Research/Testing)
**Use case**: Research, development, and algorithm testing with known datasets
- **Ground truth**: Available - you have the true labels for comparison
- **Active learning**: Use `--with-ground-truth` flag to include evaluation columns
- **Metrics tracking**: Shows full evaluation metrics (F1, precision, recall, accuracy)
- **Auto-labeling**: Can use auto-labeling for rapid testing
- **Goal**: Test candidate selection strategies, tune hyperparameters, compare methods

## Automated vs Manual Workflows

### Automated Workflow (Recommended)
**When to use**: Most scenarios, especially for testing or when you want to run multiple cycles efficiently

**Benefits**:
- Eliminates manual steps between cycles
- Consistent parameter application across cycles
- Automatic file organization and versioning
- Easy to reproduce and script

**Limitations**:
- Less control over individual steps
- Harder to debug issues at specific stages
- Fixed sequence of operations

### Manual Workflow (Step-by-step)
**When to use**: Learning the system, debugging issues, custom modifications, or fine-grained control

**Benefits**:
- Full control over each step
- Easy to debug and inspect intermediate results
- Can modify parameters between cycles
- Educational for understanding the process

**Limitations**:
- More error-prone (manual versioning)
- Time-consuming for multiple cycles
- Easy to make mistakes in file paths

## Complete Workflow Patterns

### Pattern 1: Production Deployment Workflow

**Scenario**: Deploy on unknown audio data for actual classification tasks

**Characteristics**:
- Real unlabeled audio dataset
- No ground truth available
- Manual human labeling required
- Focus on model confidence and prediction consistency

**Process**:
1. **Initial Setup**: Create small labeled training set from domain knowledge
2. **Iterative Training**: Train model → active learning → human labeling → merge labels
3. **Quality Control**: Focus on labeling consistency and clear criteria
4. **Deployment**: Use final model for inference on new audio
5. **Monitoring**: Track prediction confidence and distribution trends

**Example Flow**:
```bash
# One-time setup
python -m audioloop.create_specs
python -m audioloop.utils.create_bootstrap_set --class-name siren --n 40 --experiment prod_siren

# Iterative cycles (manual workflow for control)
python -m audioloop.train training_sets/prod_siren/training_set_v1.csv --experiment prod_siren
python -m audioloop.active_learning --class-name siren --run-number 1 --experiment prod_siren
# Manual labeling via web UI or terminal
python -m audioloop.merge_labels training_sets/prod_siren/training_set_v1.csv outputs/prod_siren/labeling_candidates_v1.csv --experiment prod_siren

# Monitor progress
python -m audioloop.track_metrics --experiment prod_siren --plot
```

### Pattern 2: Research/Evaluation Workflow

**Scenario**: Test algorithms, compare strategies, tune parameters with known datasets

**Characteristics**:
- Known dataset with ground truth
- Rapid iteration and testing
- Comprehensive metrics available
- Auto-labeling possible for speed

**Process**:
1. **Dataset Preparation**: Use existing labeled dataset for evaluation
2. **Bootstrap Sampling**: Create initial training set from ground truth
3. **Strategy Testing**: Compare different selection strategies
4. **Parameter Tuning**: Optimize thresholds and selection criteria
5. **Performance Analysis**: Full evaluation metrics across iterations

**Example Flow**:
```bash
# Automated workflow for rapid testing
python -m audioloop.automated_workflow --class-name Drill --cycles 3 --evaluation-mode --auto-label --experiment research_drill

# Compare different strategies
python -m audioloop.automated_workflow --class-name Drill --cycles 2 --selection-mode confidence --evaluation-mode --auto-label --experiment strategy_confidence
python -m audioloop.automated_workflow --class-name Drill --cycles 2 --selection-mode entropy --evaluation-mode --auto-label --experiment strategy_entropy
python -m audioloop.automated_workflow --class-name Drill --cycles 2 --selection-mode basic_transition --evaluation-mode --auto-label --experiment strategy_transition

# Analyze results
python -m audioloop.track_metrics --experiment strategy_confidence --plot
python -m audioloop.track_metrics --experiment strategy_entropy --plot
python -m audioloop.track_metrics --experiment strategy_transition --plot
```

### Pattern 3: Multi-Class Comparison Workflow

**Scenario**: Compare model performance across different audio classes

**Characteristics**:
- Multiple binary classification tasks
- Consistent methodology across classes
- Comparative analysis of class difficulty
- Experiment organization prevents file conflicts

**Process**:
1. **Class Selection**: Choose representative classes for comparison
2. **Parallel Experiments**: Run identical workflows for each class
3. **Consistent Parameters**: Use same training/selection parameters
4. **Comparative Analysis**: Analyze performance differences
5. **Insights**: Understand which classes are easier/harder to learn

**Example Flow**:
```bash
# Run identical workflows for different classes
for class_name in siren dog_bark gun_shot drilling; do
    python -m audioloop.automated_workflow --class-name $class_name --cycles 3 --evaluation-mode --auto-label --experiment eval_$class_name
done

# Analyze comparative results
for class_name in siren dog_bark gun_shot drilling; do
    python -m audioloop.track_metrics --experiment eval_$class_name --plot
done
```

## Selection Strategy Workflows

### Strategy 1: High-Confidence Start

**When to use**: Early training cycles, establishing basic patterns

**Approach**:
- Start with confidence-based selection
- Focus on samples model is most sure about
- Build reliable training foundation
- Switch to entropy later if needed

**Progression**:
1. Cycle 1-2: Confidence-based selection
2. Monitor model confidence statistics
3. Switch to entropy if overconfidence detected
4. Continue with uncertainty sampling

### Strategy 2: Automatic Transition

**When to use**: Most scenarios, especially with imbalanced datasets

**Approach**:
- Use basic transition strategy with auto-thresholds
- System automatically switches from confidence to entropy
- Adapts to dataset characteristics
- Minimal manual tuning required

**Benefits**:
- Automatic adaptation to model behavior
- Optimal threshold calculation
- Works well with rare classes
- Reduces manual parameter tuning

### Strategy 3: Pure Uncertainty Sampling

**When to use**: When model quickly becomes overconfident, later training cycles

**Approach**:
- Start with entropy-based selection immediately
- Focus on samples near decision boundaries
- Good for challenging or ambiguous datasets
- Helps prevent overconfidence issues

## Experiment Organization Best Practices

### File Organization Strategy

**Default Organization** (simple projects):
```
outputs/
├── model_v1.pt
├── model_v2.pt
├── predictions_v1.csv
└── labeling_candidates_v1.csv

training_sets/
├── training_set_v1.csv
└── training_set_v2.csv
```

**Experiment Organization** (recommended):
```
outputs/
├── siren_detection/
│   ├── model_v1.pt
│   ├── predictions_v1.csv
│   └── labeling_candidates_v1.csv
└── drill_detection/
    ├── model_v1.pt
    ├── predictions_v1.csv
    └── labeling_candidates_v1.csv

training_sets/
├── siren_detection/
│   ├── training_set_v1.csv
│   └── training_set_v2.csv
└── drill_detection/
    ├── training_set_v1.csv
    └── training_set_v2.csv
```

### Naming Conventions

**Experiment Names**:
- `{class}_{purpose}`: e.g., `siren_production`, `drill_research`
- `{strategy}_{class}`: e.g., `entropy_gunshot`, `confidence_music`
- `{dataset}_{class}`: e.g., `fsd50k_piano`, `urban8k_siren`

**Benefits**:
- Clear separation of different experiments
- No file conflicts between projects
- Easy to compare results across experiments
- Organized storage for long-term projects

## Quality Control Workflows

### Training Set Quality

**Initial Creation**:
1. Use balanced sampling (70-80% positive)
2. Ensure sufficient samples (40-60 total)
3. Verify class availability with `--list-classes`
4. Use consistent seeds for reproducibility

**Iterative Improvement**:
1. Monitor training accuracy (should reach 95%+)
2. Check label distribution after each merge
3. Review difficult cases during labeling
4. Maintain labeling consistency

### Labeling Quality Control

**Session Management**:
1. Label in batches (20-50 samples per session)
2. Take breaks to avoid ear fatigue
3. Use consistent criteria throughout
4. Skip unclear samples rather than guessing

**Quality Assurance**:
1. Review labeling criteria before starting
2. Double-check difficult or ambiguous cases
3. Use web UI for better user experience
4. Save progress frequently

### Model Quality Assessment

**Training Convergence**:
- Should reach high accuracy (95%+) on training set
- Convergence typically within 200-500 epochs
- Monitor for training instability or slow convergence

**Active Learning Effectiveness**:
- Track confidence statistics across cycles
- Monitor selection diversity (avoid always selecting same files)
- Assess improvement in model performance

## Troubleshooting Workflows

### Common Workflow Issues

**Version Mismatches**:
- Symptom: Model not found errors
- Solution: Check file naming consistency, use explicit version parameters
- Prevention: Use automated workflows or careful manual versioning

**Poor Model Performance**:
- Symptom: Low training accuracy, poor convergence
- Solution: Check training set balance, increase sample size, verify labels
- Prevention: Quality control during labeling, sufficient initial samples

**Selection Bias**:
- Symptom: Always selecting similar samples, poor diversity
- Solution: Use entropy-based selection, check confidence thresholds
- Prevention: Use basic transition strategy, monitor selection statistics

**Labeling Inconsistency**:
- Symptom: Contradictory labels, poor model performance
- Solution: Review labeling criteria, re-label inconsistent samples
- Prevention: Clear criteria documentation, regular quality checks

**Performance Oscillation Across Cycles**:
- Symptom: F1 score or accuracy jumps up and down significantly between cycles (e.g., 0.3 → 0.5 → 0.3 → 0.6)
- Cause: Training set class ratio changes each cycle due to incorrect model predictions
- Solution: Use fixed class weighting to maintain consistent decision boundaries
  ```bash
  # Use fixed weighting targeting your estimated positive percentage
  --class-weighting 0.25  # For ~25% positive class
  --class-weighting 0.05  # For ~5% positive class
  ```
- Alternative: Use adaptive weighting for balanced classes or no weighting as baseline
  ```bash
  --class-weighting adaptive  # Adapts to current training set ratio
  --class-weighting null      # No weighting (default is 0.70 fixed)
  ```

### Debugging Strategies

**Incremental Validation**:
1. Test with small datasets first
2. Verify each step produces expected outputs
3. Check intermediate file contents
4. Use evaluation mode for ground truth comparison

**Performance Analysis**:
1. Use metrics tracking to identify issues
2. Compare confidence distributions across cycles
3. Analyze selection patterns for bias
4. Review difficult cases manually

## Advanced Workflow Patterns

### Parameter Sweep Workflow

**Purpose**: Find optimal parameters for specific use case

**Process**:
1. Define parameter ranges to test
2. Run systematic experiments
3. Use evaluation mode for objective comparison
4. Analyze results to find best configuration

**Example**:
```bash
# Test different confidence thresholds
for threshold in 0.7 0.8 0.9 0.95; do
    python -m audioloop.automated_workflow --class-name siren --cycles 2 --evaluation-mode --auto-label --experiment threshold_$threshold --min-confidence $threshold
done
```

### Cross-Dataset Validation

**Purpose**: Test model generalization across datasets

**Process**:
1. Train on one dataset
2. Test on different dataset
3. Analyze performance differences
4. Identify domain adaptation needs

### Continuous Learning Workflow

**Purpose**: Ongoing model improvement in production

**Process**:
1. Deploy initial model
2. Collect new audio samples
3. Periodic retraining with new data
4. Monitor for distribution drift
5. Update model as needed

## Best Practices Summary

### Setup Phase
1. **Plan experiment organization**: Use descriptive experiment names
2. **Prepare data properly**: Pre-generate spectrograms for large datasets used repeatedly; use lazy generation (automatic when CSVs include `audio_path`) for small subsets and quick experiments
3. **Start small**: Begin with 40-60 samples in initial training set
4. **Document criteria**: Clear labeling guidelines for consistency

### Execution Phase
1. **Choose appropriate workflow**: Automated for efficiency, manual for control
2. **Select strategy wisely**: Basic transition for most cases, specific strategies for special needs
3. **Monitor progress**: Track metrics, review selection patterns
4. **Maintain quality**: Consistent labeling, regular quality checks

### Analysis Phase
1. **Review metrics**: Use comprehensive tracking for insights
2. **Compare experiments**: Analyze different strategies and parameters
3. **Document findings**: Record what works for future reference
4. **Plan next steps**: Based on results and remaining challenges

For detailed command syntax and parameters, see [cli_reference.md](cli_reference.md).
For development and architecture information, see [DEV_GUIDE.md](../DEV_GUIDE.md).