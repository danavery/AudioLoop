# Audio Labeling Guide

This guide covers practical usage of the audio labeling tool for AudioLoop's active learning workflow.

## Quick Start

```bash
# Basic usage - audio files in current directory
python -m audioloop.label_audio outputs/labeling_candidates_v1.csv

# Specify where audio files are located
python -m audioloop.label_audio outputs/labeling_candidates_v1.csv --audio-dir data/audio
```

## Before You Start

### 1. Check Your Audio Setup
The tool uses system audio commands. Make sure you have:

- **macOS**: Nothing needed (uses built-in `afplay`)
- **Linux**: Install sox for best compatibility
  ```bash
  sudo apt-get install sox  # Debian/Ubuntu
  sudo yum install sox      # RedHat/Fedora
  ```
- **Windows**: Default audio player will be used

### 2. Verify File Paths
Open your candidates CSV and check:
- The `filename` column contains valid paths
- Audio files exist at those locations
- Use `--audio-dir` if paths are relative

## Labeling Workflow

### Efficient Labeling Pattern
1. **Listen First**: Let the entire audio clip play
2. **Quick Decision**: Press 1 (positive) or 0 (negative) immediately
3. **Auto-advance**: Tool moves to next sample automatically
4. **Repeat**: Continue until done or need a break

### Keyboard Commands

**Essential Commands:**
- `1` or `y` - This IS the target sound (positive)
- `0` or `n` - This is NOT the target sound (negative)
- `u` - Jump to next unlabeled (great for resuming)
- `q` - Quit (auto-prompts to save)

**Additional Commands:**
- `p` - Replay current audio
- `n` - Skip to next without labeling
- `b` - Go back to previous sample
- `j` - Jump to specific sample number
- `s` - Save progress manually
- `h` - Show help

## Common Scenarios

### Resuming a Session
```bash
# The tool automatically finds the first unlabeled sample
python -m audioloop.label_audio outputs/labeling_candidates_v1.csv

# Or use 'u' command to jump between unlabeled samples
```

### Unclear Audio
- Press `p` to replay
- If still unclear, press `n` to skip (leave unlabeled)
- Focus on clear examples for better model training

### Taking Breaks
- Press `s` to save progress
- Or just press `q` - it will ask if you want to save
- Your position is preserved when you restart

## Best Practices

### 1. Establish Clear Criteria
Before starting, decide what counts as positive:
- **Siren Detection**: Any emergency vehicle siren, even distant
- **Dog Bark**: Clear barks only, not whines or other dog sounds
- **Gun Shot**: Actual gunshots, not fireworks or car backfires

### 2. Be Consistent
- Apply the same criteria throughout
- When in doubt, be conservative
- Quality > Quantity

### 3. Handle Edge Cases
- **Mixed Sounds**: If target sound is clearly present, label as positive
- **Distant/Quiet**: If you can identify it, label it
- **Ambiguous**: Skip it (press `n` to move on without labeling)

### 4. Optimize Your Environment
- Use good headphones or speakers
- Quiet environment helps
- Take breaks every 50-100 samples to avoid ear fatigue

## Troubleshooting

### Audio Won't Play

1. **Check file exists**:
   ```bash
   ls data/audio/filename.wav  # Replace with actual path
   ```

2. **Try manual playback**:
   ```bash
   # macOS
   afplay data/audio/filename.wav
   
   # Linux
   play data/audio/filename.wav
   aplay data/audio/filename.wav
   ```

3. **Use correct audio directory**:
   ```bash
   python -m audioloop.label_audio candidates.csv --audio-dir /full/path/to/audio
   ```

### Wrong Audio Playing
- Check the CSV filename column matches actual files
- Verify no duplicate filenames in different directories

### Can't Save
- Check write permissions on the CSV file
- Make sure the file isn't open in another program

## Progress Tracking

The tool shows a progress bar:
```
Progress: [████████████████████░░░░░░░░░░░░░░░░░░░░] 20/50 (40.0%)
```

This helps you:
- See how many samples remain
- Decide when to take breaks
- Track overall progress

## Tips for Large Datasets

### Batch Strategy
1. Do 20-30 samples at a time
2. Take a 5-minute break
3. Resume with the `u` command
4. Save after each batch

### Multiple Sessions
Split work across days:
- Day 1: High-confidence positive predictions
- Day 2: High-confidence negative predictions  
- Day 3: Review any corrections needed

### Team Labeling
If multiple people are labeling:
1. Split the candidates file into chunks
2. Each person labels their chunk
3. Merge the results afterwards

## Quality Control

### Self-Check Questions
- Am I being consistent with my criteria?
- Am I rushing through samples?
- Should I take a break?
- Am I confident in this label?

### When to Skip
It's better to skip (leave unlabeled) than to guess:
- Audio is corrupted or too noisy
- Multiple sounds make it ambiguous
- You're unsure about the criteria
- The sound is at the very edge of audibility

## File Management

### Input File
`outputs/labeling_candidates_v1.csv` contains:
- `filename`: Path to audio file
- `prediction`: What the model thinks
- `confidence`: How sure the model is
- `needs_human_label`: Your input goes here

### Output
The same file is updated with your labels:
- `needs_human_label` = "1" for positive
- `needs_human_label` = "0" for negative
- Empty = skipped/unlabeled

### Backup
The tool doesn't auto-backup, so consider:
```bash
# Make a backup before starting
cp outputs/labeling_candidates_v1.csv outputs/labeling_candidates_v1.csv.backup
```

## Example Session

```
$ python -m audioloop.label_audio outputs/labeling_candidates_v1.csv --audio-dir data/audio

Loaded 50 candidates for labeling
10 samples already labeled

Simple Audio Labeling Tool
Type 'h' for help
Commands: 1=positive, 0=negative, u=next unlabeled, q=quit

============================================================
Sample 11 of 50
============================================================
Filename: fold5/24347-8-0-22.wav
Prediction: siren
Confidence: 0.951
Current Label: [Not yet labeled]

Progress: [████████░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░] 10/50 (20.0%)

Playing audio...

Command: 1
✓ Labeled as POSITIVE (1)

============================================================
Sample 12 of 50
============================================================
Filename: fold3/157821-3-0-15.wav
Prediction: siren  
Confidence: 0.872
Current Label: [Not yet labeled]

Progress: [█████████░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░] 11/50 (22.0%)

Playing audio...

Command: 0
✓ Labeled as NEGATIVE (0)

[... continues ...]

Command: q
Save changes before quitting? (y/n): y

Saved labels to outputs/labeling_candidates_v1.csv
Goodbye!
```

## Next Steps

After labeling, merge your labels into the training set:
```bash
python -m audioloop.merge_labels training_sets/training_set_v1.csv outputs/labeling_candidates_v1.csv
```

This creates `training_sets/training_set_v2.csv` ready for the next training cycle.