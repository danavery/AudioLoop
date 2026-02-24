# AudioLoop: Active Learning Framework for Audio Classification

AudioLoop is an active learning framework for binary audio classification that minimizes human labeling effort through strategic sample selection.

## Quick Start/Demo

AudioLoop is installed once and used from separate project directories for each classification task.

### 1. Install AudioLoop (One-Time)

```bash
# Clone to a permanent location
export AUDIOLOOP_REPO="~/audioloop"
git clone git@github.com:danavery/audioloop.git $AUDIOLOOP_REPO 
cd $AUDIOLOOP_REPO

# Install with uv (recommended)
uv sync

# activate the python environment
source .venv/bin/activate

# test the installation
python -m audioloop.train --help

# Optional: Add alias to ~/.zshrc or ~/.bashrc for easy activation
echo 'alias al="source ${AUDIOLOOP_REPO}/.venv/bin/activate"' >> ~/.zshrc
```

### 2. Create a Working Project Directory

```bash
# Activate audioloop (or use your alias: al)
source ${AUDIOLOOP_REPO}/.venv/bin/activate

# Create and initialize a new project directory
export PROJECT_DIR="~/audioloop-projects/demo_project"
mkdir -p $PROJECT_DIR
cd $PROJECT_DIR
python -m audioloop.init_project
```

### 3. Get and install the UrbanSound8K Dataset

```bash
# Download dataset (UrbanSound8K)
# from https://urbansounddataset.weebly.com/urbansound8k.html (bottom of the page)
# You'll get `UrbanSound8K.tar.gz`

tar xzf UrbanSound8K.tar.gz

mkdir ${PROJECT_DIR}/data/urbansound8k
mv UrbanSound8K/audio/* ${PROJECT_DIR}/data/urbansound8k
mv UrbanSound8K/metadata/UrbanSound8K.csv ${PROJECT_DIR}/data/urbansound8k
```

### 4. Edit the Audioloop config file to use the UrbanSound8K presets

```bash
# Uncomment and replace `dataset: fsd50k` in ${PROJECT_DIR}/audioloop.yaml with `dataset: urbansound8k`
```

### 5. Generate spectrograms

```bash
python -m audioloop.create_specs
```

### 6. Create a seed training set 

```bash
# List UrbanSound8K's classes
python -m audioloop.utils.create_bootstrap_set --list-classes 

# Create a seed training set for the siren class
python -m audioloop.utils.create_bootstrap_set --dataset urbansound8k --class-name siren --n 40

```

### 7. Run three cycles with auto labeling

```bash
# Since we have ground truth, let's just let it run for three cycles and 
# see how well the training works assuming correct labels are assigned

python -m audioloop.automated_workflow --class-name siren --cycles 3 --evaluation-mode --auto-label
```

### 8. Run two cycles with manual labeling

```bash
# Continue training the model for cycles 4 and 5, but this time with manual labeling
python -m audioloop.automated_workflow --class-name siren --cycles 5 --start-cycle 4 --evaluation-mode

# When prompted, start the web UI in another terminal session
python -m audioloop.webui [--host 0.0.0.0 if not localhost]

# Now open the labeling UI in a browser at 
# http://localhost:5000
# Remember to save your labels before exiting the web UI!
```

For detailed usage instructions, see **[USAGE_GUIDE.md](USAGE_GUIDE.md)**.


## Core Concepts

Audioloop is a framework for iterative, human-in-the-loop binary labeling over large, unlabeled audio corpora using uncertainty-driven sampling.

It supports two types of use cases:
1) Dataset constuction ('label' mode): train a model to produce dataset-wide labels with less manual effort than traditional random sampling. Example: I want to label all the positive and negative examples of blue whale "A" calls in a dataset too large to reasonably label by hand.
2) Rare-event search ('search' mode): surface positive examples of whatever you're looking for in your dataset. This use case creates a model that prioritizes recall over calibrated classification. Example: I want to find more blue whale "A" calls in my hydrophone recording dataset.

### Active learning loop

Audioloop's core idea is that we can save human labeling time by starting with a small known set of labels and iteratively creating models that get better and better at currently identifying positive and negative examples. Audio clips that would be especially helpful to add to the training set get selected for explicit human classification. This way the human labelers can focus on the useful cases and not on random sets of examples.

1) A user manually creates an initial small positive/negative training set CSV of discrete audio clips for classification
2) A model is trained on that training set
3) Audioloop selects candidate example clips--ones particulaly helpful for models to train on based on the current candidate selection strategy--for human review
4) A human hand-labels the candidate examples (using web UI or CLI), which are added to the current training set
5) Audioloop returns to step 2 with the new larger training set, unless the cycle stopping criteria are met, in which case we're done and hopefully have a model we can use to label the remainder of our dataset

Each of these steps can be run individually, but the automated_workflow action, which guides the workflow in the above sequence of steps, is strongly preferred.

### Candidate Selection Strategies

This is one of the core mechanisms of the Audioloop process. Examples are selected for human review via the current active candidate selection method. Candidate selection strategies are tunable, pluggable, and extensible, but some basic strategies are provided in the Audioloop package--current options are entropy, confidence, and mixed strategies.

### Production vs evaluation mode

There are two running modes:
1) Production mode: This mode is the normal working mode of Audioloop. It runs the loop as expected, creating models for dataset labeling. It doesn't report overall metrics because it doesn't have ground truth data, which makes sense because producing ground truth data is the main point of Audioloop. It does report metrics on the current model's performance on the newest set of candidates once they are human-labeled.
2) Evaluation mode (ground truth mode):  This mode runs the active learning loop like production mode but produces metrics for the current working model based on the actual ground truth labels for the _entire_ dataset being labeled. A CSV of ground truth labels for the dataset in question is required. The pre-existing labels are compared to the current model's performance over the entire dataset. This mode is used to test selection strategies and simulate performance on known, labeled datasets.

### Experiments/Projects:

1) Experiment: Each Audioloop run is given an "experiment" name to distinguish it from other experiments for storing and retrieving training sets and models for each cycle of the experiment. Experiment directories hold the models, predictions, candidates, and training sets generated.
2) Project: A project is a container for a set of experiments. It includes data, configuration, and experiments which are confined to a single project directory. All Audioloop experiments happen inside a project directory, which is created with the "init_project" action. Your dataset needs to be present inside this directory or symlinked to it.

### Extensibility:
Audioloop is designed to be pluggable and extensible--see EXTENDING.md to add custom datasets, models, and strategies.

## Current Limitations

### Binary classification

In the alpha release, Audioloop is focused on binary classification--is a specific example audio clip a "positive" or "negative" example of a desired sound--for simplicity. Multi-class and multi-label datasets are not supported at the moment. Multi-label or multi-class labeling would currently have to be performed one class at a time.

### Single-user labeling

There are currently no facilities for multi-user, asynchronous labeling. All labeling needs to be done by one user at a time. Future releases will allow for spreading labeling workload across multiple users.
