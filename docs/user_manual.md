# AudioLoop User Manual

## The Loop

The core workflow of AudioLoop is training->candidate_selection->labeling->training->candidate_selection->labeling->...

* A initial training set with positive and negative examples of a particular class is used to train a model from scratch
* That model is used to select candidates from the larger dataset for human labeling 
* A person labels the candidate clips as positive or negative for the active class
* The labeling results are added to the training set, and the cycle starts again
* When a certain set of criteria are met (see [cycle_stopping_criteria.md](cycle_stopping_criteria.md) for details), the work stops
* The most recently trained model can then be used to classify or search the remainder of the dataset

One round of training (training/candidate_selection/labeling) is called a "cycle".

Most of the time all you'll need to do is set up your dataset config, adjust the audioloop.yaml config for everything else, and start automated_workflow. 

You can run each part of the cycle individually (see the action list in the README), but in general you shouldn't need to.

Example: `python -m audioloop.automated_workflow --class-name siren --cycles 3` will start the process on the dataset and with the other configuration options specified in the project directory's `audioloop.yaml`. `python -m audioloop.automated_workflow --class-name siren --cycles 6 --start-cycle 4` will pick up at cycle 4 and run another three cycles.

AudioLoop has two running modes: **production mode** (the default) for real-world use with unlabeled data, and **evaluation mode** (`--evaluation-mode` as an optional `automated_workflow` CLI parameter) for system/parameter testing with datasets that already have ground truth labels. Evaluation mode enables full corpus metrics and optional auto-labeling (`--auto-label`). See the README for details.

## General Configuration

### Datasets

Per-dataset handling is configured in specific instances of `DatasetConfig` in the `datasets` project directory (apart from pre-supplied dataset configs for Audioset, FSD50K, and UrbanSound8K). 

Dataset configuration tells AudioLoop where the clip files are, where the metadata is, what splits are available, a list of classes, and any custom spectrogram parameters.

Examples of dataset configuration are available in the current project directory's `datasets/templates`. Copy that template to `datasets` and alter as needed. Examples of complete pre-supplied configurations are in the AudioLoop source directory `src/audioloop/datasets`

### audioloop.yaml

`audioloop.yaml` contains the rest of the configuration. The version installed in new project directories contains commented out examples of all available configuration options.

You'll almost certainly want to update at least the following:

* `dataset`: Your dataset name, based on your dataset configuration's name
* `experiment_name`: A name for the set of training cycles you're about to run. This creates subdirectories under `outputs/` and `training_sets/` where artifacts of the training process are saved. If no experiment name is given, the artifacts will be saved directly in `outputs/` and `training_sets/`. An experiment name is strongly suggested.

### Configuration Precedence

When the same setting is specified in multiple places, AudioLoop resolves it in this order (highest priority first):

1. **CLI flags** — always win
2. **`audioloop.yaml`** — project-level defaults
3. **Environment variables** — fallback when no CLI flag or yaml value
4. **Built-in defaults**

### Environment Variables

All environment variables are optional. They serve as fallbacks when values aren't set via CLI or yaml.

| Variable | Description | Default |
|----------|-------------|---------|
| `AUDIOLOOP_DATASET` | Default dataset name | _(none)_ |
| `AUDIOLOOP_EXPERIMENT` | Default experiment name | _(none)_ |
| `AUDIOLOOP_PROJECT_ROOT` | Override project root detection | _(auto-detected)_ |
| `AUDIOLOOP_DATA_ROOT` | Root directory for data files | `data` |
| `AUDIOLOOP_OUTPUT_ROOT` | Root directory for outputs | `.` |
| `AUDIOLOOP_SPECS_DIR` | Spectrogram subdirectory name | `all_specs` |

Note: The path variables (`PROJECT_ROOT`, `DATA_ROOT`, `OUTPUT_ROOT`) are not configurable via `audioloop.yaml` — they are only read from environment variables or defaults. `SPECS_DIR` can also be configured via `specs_dir_path` in `audioloop.yaml`.

## Spectrogram Generation

Each example audio clip needs to be provided in an individual file. Those files can be converted into spectrograms:

  * Ahead of time and all at once with the `create_specs` action. (Example: `python -m audioloop.create_specs`)
  * Lazily as needed--if a spectrogram doesn't exist, it will be created during training. This will make the first round of training and candidate selection for a dataset quite slow.
  
For typical custom datasets with all audio files in one directory, populating `_audio_root` and `_audio_extension` should be enough, but you can create a custom `get_audio_path()` to work with more complex datasets.

### Output

The newly-created spectrograms live in the `specs_dir_path` directory (default: `data/all_specs`), configurable in `audioloop.yaml` or via the `AUDIOLOOP_SPECS_DIR` env var.

### Options

Note that none of these values have built-in defaults, and they all need to be specified in the dataset configuration. Examples can be found in the built-in dataset configs:

* _audio_root: The directory where your audio files live
* _audio_extension: The file extension for your audio files (.wav, .flac, etc.)
* _max_spectrogram_length: Maximum length for the created spectrograms. Useful if you have some particularly large outliers. (`create_specs` will give you a histogram of created lengths when it's done)
* _sample_rate (Hz)
* _n_fft
* _hop_length
* _n_mels

## Training

Given a training set, a new model is trained to a certain level of accuracy on that set. No validation set is used, so overtraining is not only a risk, but expected, especially during the first few cycles. By default, the model trains until 20 epochs have passed without a 0.01 decrease in training loss. However, there's also an `accuracy_floor` that by default requires a minimum accuracy of the majority class percentage plus 15% before stopping, to prevent slow starts from causing too-early stopping. A learning rate scheduler is also used by default (see parameters `lr_scheduler_patience`, `lr_scheduler_factor`, and `lr_scheduler_min_lr`).

The specific model architecture and many hyperparameters used are configurable.

### Output

The new working model is saved in `outputs/{experiment_name}/model_v{N}.pt` where `N` is the current cycle number.

### Options

Some training parameters:

* model_type (default: `cnn5layer`): model architecture to use. Current options are `cnn5layer`, `cnn7layer`, and `simplecnn`. The default works reasonably well, but other models (coming soon), especially pre-trained ones, will probably work better. See also [extending.md](extending.md) for how to add your own.

* class_weighting (default: 0.70): controls how the loss function balances positive and negative examples. A value of 0.70 down-weights positive-class errors relative to negative-class errors, which biases the model toward predicting positive — it would rather flag something incorrectly than miss a true positive. This surfaces more rare positive examples during candidate selection.

  - A float (0.0–1.0): higher values bias the model more toward positive predictions. The default 0.70 works well for rare-event search.
  - "adaptive": recalculates weights from the current training set ratio each cycle
  - null: no weighting. This rarely produces good results with unbalanced datasets.

* stopping_criterion_type (default: `plateau`): what training early-stopping mode to use. Options are `plateau` and `accuracy`. See [stopping_criteria_guide.md](stopping_criteria_guide.md) for details. 
  
* batch_size (default: 32): training batch size. Bigger will generally train faster, too big and you'll run out of VRAM. Likely no real need to adjust this one.

## Candidate Selection

After training completes, the candidate selection process is performed to determine which clips should be selected for human review. `total_candidates` determines how many candidates to select (default: 50). By default those candidates are selected at random from a set of the highest-qualifying `total_candidates` * `candidate_pool_multiplier` clips (`candidate_pool_multiplier` default is 5, so the 50 are selected from a pool of 250). 


By default, the "entropy" (uncertainty) selection strategy is used. This selects clips that are the closest to the model's decision boundary. Other strategies and combinations of strategies are built-in or can be added.

Also see the `positive_percentage` parameter for stratification options if you want to force the selected candidates to have a certain percentage of positive examples. This may be particularly useful for very unbalanced datasets.

### Output

The set of output candidates is saved in `outputs/{experiment_name}/labeling_candidates_v{N}.csv`, where `N` is the cycle number.

## Labeling

Once candidate selection completes, you'll need to label those candidates and save those labels into a new expanded training set for the next cycle. There are two UIs for labeling:

* Web UI (recommended): A small web application you can connect to with a browser to listen to and label the candidates. The browser can be on a different machine than the audio files, as long as the application is run on the same machine as the audio files.
  * `python -m audioloop.webui` (add `--host 0.0.0.0` if your browser is on a different machine)
  * Now open the labeling UI in a browser at `http://localhost:5000` (or `http://[server]:5000` if browser is on a different machine)
  * The latest labeling_candidates_v{N}.csv file should be selected automatically, otherwise select it from the dropdown list 
  * Choose Positive or Negative for each presented candidate
  * Make sure to save your labels before closing the window!
  
* CLI: A command-line version. This must be run on the same machine as the audio files to allow audio playback. Linux users need `sox` installed for audio playback.
  * `python -m audioloop.label_audio <candidates_file>`, where `candidates_file` is the most recent file (`outputs/{experiment_name}/labeling_candidates_v{N}.csv` where `N` is the largest value present)
  * Type "h" for help, and follow the instructions in the interface to label and save your work

### Output

A new training set: `training_sets/{experiment_name}/training_set_v{N+1}.csv` where `N+1` is the _next_ cycle number, since this set will be fed into the next training cycle.

## Cycle Stopping

By default, the active learning loop runs for the number of cycles you specify. Automatic stopping based on candidate metrics is available but experimental — see [cycle_stopping_criteria.md](cycle_stopping_criteria.md) for details.

## Using the Final Model

Whether you stop manually or use automatic cycle stopping, you end up with a trained model you can use. Each cycle produces a model file in the experiment output directory:

```
outputs/<experiment_name>/
    model_v1.pt
    model_v2.pt
    ...
    model_v10.pt           # Latest cycle's model
    model_best.pt          # Only if cycle stopping was used
    predictions_v10.csv    # Predictions from latest cycle
```

The most recent model (`model_v<last_cycle>.pt`) is typically the best one. The corresponding `predictions_v<last_cycle>.csv` contains per-file predictions, confidences, and entropy scores for every file in the dataset (excluding training set members). You can sort by confidence to find the strongest matches, or by entropy to find examples the model is most uncertain about.

Templates to use the model directly will be available in a later alpha release.

### File Formats

**Bootstrap training set:**
```csv
filename,label,audio_path
clip_001.pt,1,/path/to/audio/clip_001.wav
```

**Merged training set:**
```csv
filename,label
data/all_specs/100032-3-0-0.pt,1
```

**Predictions CSV — production mode:**
```csv
filename,prediction,predicted_class,target_class,confidence,entropy,prob_negative,prob_positive,original_class,audio_path,filepath
```

**Predictions CSV — evaluation mode:**
```csv
filename,ground_truth,prediction,predicted_class,target_class,confidence,entropy,prob_negative,prob_positive,original_class,correct,audio_path,filepath
```

**Candidates CSV:**
```csv
filename,prediction,predicted_class,target_class,confidence,entropy,prob_negative,prob_positive,original_class,audio_path,filepath,needs_human_label,human_confidence
```

You can track how the model is progressing at any point during or after the active learning loop:

```bash
python -m audioloop.track_metrics --experiment my_experiment --plot
```

This shows learning curves for prediction metrics across cycles. In evaluation mode (with ground truth), it also tracks F1, precision, and recall for the entire corpus.

## Experiment Organization

A **project** is a directory containing `audioloop.yaml` and the standard subdirectories. New projects are created with `python -m audioloop.init_project`.

An **experiment** is a named run within a project, set via `experiment_name` in `audioloop.yaml` or `--experiment-name` on the command-line. The experiment name creates subdirectories that keep artifacts separated:

```
my_project/
    audioloop.yaml
    datasets/
        my_dataset_config.py
    outputs/
        siren_search/              # experiment_name: siren_search
            model_v1.pt
            model_v2.pt
            predictions_v2.csv
            labeling_candidates_v1.csv
            labeling_candidates_v2.csv
        alarm_detection/           # experiment_name: alarm_detection
            model_v1.pt
            ...
    training_sets/
        siren_search/
            training_set_v1.csv    # bootstrap set
            training_set_v2.csv
            training_set_v3.csv
        alarm_detection/
            training_set_v1.csv
            ...
    data/
        all_specs/                 # shared across experiments
```

You can run multiple experiments on the same dataset (different target classes, different parameters) without conflicts. You can run multiple experiments on different datasets as long as there's no overlap in audio file names, since at the moment all generated spectrograms live in the same project data directory. 

Without an experiment name, artifacts go directly into `outputs/` and `training_sets/`. This works but gets messy quickly — an experiment name is strongly recommended.

## Batch Runner

The batch runner runs multiple experiment configurations sequentially — useful for parameter sweeps or running the same class across different settings.

Each batch config is a YAML file with two sections: `workflow` (class name, cycle count, etc.) and `config` (all `audioloop.yaml` parameters):

```yaml
workflow:
  class_name: "dog_bark"
  num_cycles: 10
  auto_label: true
  evaluation_mode: true

config:
  experiment_name: "dog_entropy"
  dataset: "urbansound8k"
  selection_mode: "entropy"
  total_candidates: 50
```

Run one or more configs:

```bash
python -m audioloop.batch_runner configs/sweep/*.yaml

# With a shared bootstrap training set
python -m audioloop.batch_runner \
    --initial-training-set training_sets/shared/bootstrap_dog_40.csv \
    configs/sweep/*.yaml
```

Outputs are organized under a timestamped batch directory to keep runs separate:

```
outputs/batch_20251215_143022/dog_entropy/
training_sets/batch_20251215_143022/dog_entropy/
```

See `configs/README.md` for the full configuration reference, example configs, and configuration precedence rules.

## Troubleshooting

**No AudioLoop project found**
You're running a command outside a project directory. Make sure you've created a project directory using the `init_project` action. Either `cd` to your project directory (where `audioloop.yaml` lives) or set `AUDIOLOOP_PROJECT_ROOT=/path/to/project`.

**Spectrograms are very slow to generate**
This is normal for large datasets on the first run if the spectrograms haven't been generated yet. Use `python -m audioloop.create_specs` to generate them all at once rather than lazily during training.

**Training is very slow**
Check that PyTorch is using your GPU: `python -c "import torch; print(torch.cuda.is_available())"`. If you have a GPU but CUDA isn't available, you may need to reinstall PyTorch with CUDA support.

**"Out of memory" during training**
Reduce `batch_size` in `audioloop.yaml` (default is 32). If you're on CPU, also check that spectrograms aren't unusually large — review `_max_spectrogram_length` in your dataset config.

**Automated workflow stops and asks me to label**
This is expected. The workflow pauses after each candidate selection so you can label in a separate terminal (web UI or CLI). It can be resumed once labeling is saved and merged.

**Web UI can't find audio files**
The web UI must be run on the same machine where the audio files are stored. If using `--host 0.0.0.0` for remote browser access, the audio is streamed from the server — the browser doesn't need local access to the files.

**Predictions CSV is missing files**
Files that are in the current training set are excluded from predictions. This is intentional — the model has already seen those examples.

**Crash (segfault) during spectrogram generation or training**
Some audio files in public datasets are corrupt and cause `torchaudio.load()` to segfault — a crash that can't be caught with try/except. AudioLoop has a bad-file exclusion mechanism to handle this. See [bad_files.md](bad_files.md) for how to identify and exclude corrupt files.

## Closing Warning

Many details of this project are subject to change. Action names and CLI parameters may change, configuration parameters' names and functions will almost certainly change, the entire workflow may change. No software architecture survives contact with the user. (That's a good thing!)
