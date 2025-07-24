#!/usr/bin/env python3
"""
Audio labeling tool for active learning workflow with multiple dataset support.

This tool is designed to work with the output of the active learning pipeline,
specifically the labeling_candidates CSV files that contain a 'filepath' column.

Supports both UrbanSound8K and FSD50K datasets with automatic dataset detection
or explicit dataset specification.

Requirements:
- Candidates CSV must have a 'filepath' column (from active_learning.py output)
- Audio files must be in proper dataset structure
- Use --audio-dir to point to the appropriate audio directory

Usage:
  # Auto-detect dataset from filepath
  python -m audioloop.label_audio outputs/labeling_candidates_v1.csv

  # Explicitly specify UrbanSound8K
  python -m audioloop.label_audio outputs/labeling_candidates_v1.csv --dataset urbansound8k --audio-dir data/urbansound8k

  # Explicitly specify FSD50K
  python -m audioloop.label_audio outputs/labeling_candidates_v1.csv --dataset fsd50k --audio-dir data/FSD50K/FSD50K.dev_audio

Controls:
- 1 or y: Label as POSITIVE (target sound detected)
- 0 or n: Label as NEGATIVE (target sound NOT detected)
- p: Play/replay audio
- x: Stop currently playing audio
- u: Jump to next unlabeled sample
- q: Quit and save

The tool automatically plays audio after each label for efficient workflow.
"""

import argparse
import csv
import os
import subprocess
import sys

from .config import AudioLoopConfig


class SimpleAudioLabeler:
    """Simple terminal-based audio labeling interface with multi-dataset support."""

    def __init__(self, candidates_csv, dataset_name, audio_dir=None):
        """
        Initialize the audio labeler.

        Args:
            candidates_csv: Path to candidates CSV file
            dataset_name: Dataset name ('urbansound8k' or 'fsd50k')
            audio_dir: Directory containing audio files
        """
        self.candidates_csv = candidates_csv
        self.dataset_name = dataset_name
        self.current_index = 0
        self.candidates = []
        self.fieldnames = []
        self.changes_made = False
        self.audio_process = None  # For tracking currently playing audio

        # Load candidates
        self._load_candidates()

        # Set up dataset processor and config
        dataset_kwargs = {}
        if audio_dir:
            dataset_kwargs["audio_root"] = audio_dir

        config = AudioLoopConfig(dataset=self.dataset_name)
        self.dataset_config = config.get_dataset_config()

        # Set default audio directory if not provided
        if audio_dir is None:
            self.audio_dir = str(self.dataset_config.audio_root)
        else:
            self.audio_dir = audio_dir

        print(f"Using dataset: {self.dataset_name}")
        print(f"Audio directory: {self.audio_dir}")

    def _load_candidates(self):
        """Load candidates from CSV file."""
        with open(self.candidates_csv) as f:
            reader = csv.DictReader(f)
            self.candidates = list(reader)
            self.fieldnames = reader.fieldnames

        print(f"\nLoaded {len(self.candidates)} candidates for labeling")

        # Find already labeled samples
        labeled_count = sum(
            1 for c in self.candidates if c.get("needs_human_label", "").strip() != ""
        )
        if labeled_count > 0:
            print(f"{labeled_count} samples already labeled")

            # Find first unlabeled sample
            for i, candidate in enumerate(self.candidates):
                if candidate.get("needs_human_label", "").strip() == "":
                    self.current_index = i
                    break

    def _get_audio_path(self, candidate):
        """Get the full path to the audio file from the filepath column."""
        # Get filepath from predictions CSV (required)
        filepath = candidate.get("filepath", "")
        if not filepath:
            print(
                "ERROR: No 'filepath' column found in candidate. This tool requires candidates from active learning workflow."
            )
            print(f"Candidate columns: {list(candidate.keys())}")
            return None

        # The filepath is to a spectrogram (.pt), convert to audio (.wav)
        audio_filepath = filepath.replace(".pt", ".wav")
        filename = os.path.basename(audio_filepath)

        if self.dataset_name == "urbansound8k":
            # For UrbanSound8K: search through all fold directories
            if "-" in filename and filename.endswith(".wav"):
                for fold_num in range(1, 11):  # fold1 through fold10
                    fold_path = os.path.join(self.audio_dir, f"fold{fold_num}", filename)
                    if os.path.exists(fold_path):
                        return fold_path

            print(f"ERROR: Audio file not found: {filename}")
            print(f"Searched in: {self.audio_dir}/fold1/ through {self.audio_dir}/fold10/")
            return None

        if self.dataset_name == "fsd50k":
            # For FSD50K: files are directly in the audio directory
            audio_path = os.path.join(self.audio_dir, filename)
            if os.path.exists(audio_path):
                return audio_path

            print(f"ERROR: Audio file not found: {filename}")
            print(f"Searched in: {self.audio_dir}")
            return None

        print(f"ERROR: Unsupported dataset: {self.dataset_name}")
        return None

    def _play_audio(self, audio_path):
        """Play an audio file using system command (non-blocking)."""
        if not audio_path or not os.path.exists(audio_path):
            print(f"Audio file not found: {audio_path}")
            return False

        # Stop any currently playing audio
        self._stop_audio()

        try:
            if sys.platform == "darwin":  # macOS
                self.audio_process = subprocess.Popen(
                    ["afplay", audio_path], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
                )
            elif sys.platform.startswith("linux"):
                # Try multiple Linux audio players
                for player in ["aplay", "play", "cvlc --play-and-exit", "ffplay -nodisp -autoexit"]:
                    if (
                        subprocess.run(
                            f"which {player.split()[0]}", shell=True, capture_output=True
                        ).returncode
                        == 0
                    ):
                        self.audio_process = subprocess.Popen(
                            f"{player} '{audio_path}'",
                            shell=True,
                            stdout=subprocess.DEVNULL,
                            stderr=subprocess.DEVNULL,
                        )
                        break
            elif sys.platform == "win32":  # Windows
                # For Windows, use PowerShell to play audio in a way we can control
                self.audio_process = subprocess.Popen(
                    [
                        "powershell",
                        "-c",
                        f"(New-Object Media.SoundPlayer '{audio_path}').PlaySync()",
                    ],
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                )
            else:
                print(f"Unsupported platform for audio playback: {sys.platform}")
                return False
            return True
        except Exception as e:
            print(f"Error playing audio: {e}")
            return False

    def _stop_audio(self):
        """Stop currently playing audio."""
        if self.audio_process and self.audio_process.poll() is None:
            try:
                self.audio_process.terminate()
                # Give it a moment to terminate gracefully
                try:
                    self.audio_process.wait(timeout=1)
                except subprocess.TimeoutExpired:
                    # Force kill if it doesn't terminate gracefully
                    self.audio_process.kill()
                print("Audio stopped")
            except Exception as e:
                print(f"Error stopping audio: {e}")
            finally:
                self.audio_process = None

    def _save_candidates(self):
        """Save candidates back to CSV file."""
        if not self.fieldnames:
            print("Error: No fieldnames available for CSV writing")
            return

        with open(self.candidates_csv, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=self.fieldnames)
            writer.writeheader()
            writer.writerows(self.candidates)
        self.changes_made = False
        print(f"\nSaved labels to {self.candidates_csv}")

    def _auto_play_current(self):
        """Auto-play current sample's audio."""
        if self.current_index < len(self.candidates):
            audio_path = self._get_audio_path(self.candidates[self.current_index])
            if audio_path:
                print("\nPlaying audio...")
                self._play_audio(audio_path)

    def _advance_and_play(self):
        """Advance to next sample and auto-play audio."""
        if self.current_index < len(self.candidates) - 1:
            self.current_index += 1
            self._display_current()
            self._auto_play_current()

    def _label_and_advance(self, label: str, message: str):
        """Label current sample and advance to next with auto-play."""
        if self.current_index < len(self.candidates):
            self.candidates[self.current_index]["needs_human_label"] = label
            self.changes_made = True
            print(f"✓ {message}")
            self._advance_and_play()

    def _jump_to_sample(self, index: int):
        """Jump to a specific sample and display it."""
        if 0 <= index < len(self.candidates):
            self.current_index = index
            self._display_current()
            return True
        return False

    def _display_current(self):
        """Display current candidate information."""
        if self.current_index >= len(self.candidates):
            print("\nNo more candidates to label!")
            return

        candidate = self.candidates[self.current_index]

        print("\n" + "=" * 60)
        print(
            f"Sample {self.current_index + 1} of {len(self.candidates)} ({self.dataset_name.upper()})"
        )
        print("=" * 60)

        # Display candidate info
        print(f"Filename: {candidate.get('filename', 'Unknown')}")
        print(f"Filepath: {candidate.get('filepath', 'Unknown')}")
        print(f"Prediction: {candidate.get('prediction', 'Unknown')}")
        print(f"Confidence: {candidate.get('confidence', 'Unknown')}")
        print(f"Original Class: {candidate.get('original_class', 'Unknown')}")

        current_label = candidate.get("needs_human_label", "").strip()
        if current_label:
            print(f"Current Label: {current_label}")
        else:
            print("Current Label: [Not yet labeled]")

        # Progress bar
        labeled = sum(1 for c in self.candidates if c.get("needs_human_label", "").strip() != "")
        progress = labeled / len(self.candidates) * 100
        bar_length = 40
        filled = int(bar_length * labeled / len(self.candidates))
        bar = "█" * filled + "░" * (bar_length - filled)
        print(f"\nProgress: [{bar}] {labeled}/{len(self.candidates)} ({progress:.1f}%)")

    def _show_help(self):
        """Display help information."""
        print("\nCommands:")
        print("  1 or y - Label as POSITIVE (1) - This IS the target sound")
        print("  0 or n - Label as NEGATIVE (0) - This is NOT the target sound")
        print("  p      - Play/replay audio")
        print("  x      - Stop currently playing audio")
        print("  n      - Next sample (without labeling)")
        print("  b      - Previous sample")
        print("  j      - Jump to sample number")
        print("  u      - Jump to next unlabeled (great for resuming)")
        print("  s      - Save progress (also auto-saves on quit)")
        print("  h or ? - Show this help")
        print("  q      - Quit (will prompt to save if needed)")
        print("\nTip: After labeling with 1/0, audio auto-plays for the next sample")

    def run(self):
        """Run the interactive labeling session."""
        print(f"\nSimple Audio Labeling Tool - {self.dataset_name.upper()}")
        print("Type 'h' for help")
        print("Commands: 1=positive, 0=negative, u=next unlabeled, x=stop audio, q=quit\n")

        # Display first candidate and auto-play
        self._display_current()
        self._auto_play_current()

        try:
            while True:
                try:
                    command = input("\nCommand: ").strip().lower()

                    if not command:
                        continue

                    if command in ["q", "quit"]:
                        self._stop_audio()  # Stop any playing audio before quitting
                        if self.changes_made:
                            save = input("Save changes before quitting? (y/n): ").strip().lower()
                            if save in ["y", "yes"]:
                                self._save_candidates()
                        print("Goodbye!")
                        break

                    elif command in ["h", "help", "?"]:
                        self._show_help()

                    elif command in ["1", "y"]:
                        self._label_and_advance("1", "Labeled as POSITIVE (1)")

                    elif command in ["0", "n"]:
                        self._label_and_advance("0", "Labeled as NEGATIVE (0)")

                    elif command == "p":
                        self._auto_play_current()
                        if not self._get_audio_path(self.candidates[self.current_index]):
                            print("ERROR: No valid audio file path found")

                    elif command == "x":
                        self._stop_audio()

                    elif command == "n":  # Next without labeling
                        if self.current_index < len(self.candidates) - 1:
                            self.current_index += 1
                            self._display_current()

                    elif command == "b":  # Back/previous
                        if self.current_index > 0:
                            self.current_index -= 1
                            self._display_current()

                    elif command == "j":
                        try:
                            jump_to = int(input("Jump to sample number (1-based): ")) - 1
                            if not self._jump_to_sample(jump_to):
                                print("Invalid sample number!")
                        except ValueError:
                            print("Invalid number!")

                    elif command == "u":
                        # Jump to next unlabeled
                        found = False
                        for i in range(self.current_index + 1, len(self.candidates)):
                            if self.candidates[i].get("needs_human_label", "").strip() == "":
                                self.current_index = i
                                found = True
                                break

                        if not found:
                            # Wrap around to beginning
                            for i in range(0, self.current_index):
                                if self.candidates[i].get("needs_human_label", "").strip() == "":
                                    self.current_index = i
                                    found = True
                                    break

                        if found:
                            self._display_current()
                            self._auto_play_current()
                        else:
                            print("No unlabeled samples found!")

                    elif command == "s":
                        self._save_candidates()

                    else:
                        print(f"Unknown command: '{command}'. Type 'h' for help.")

                except KeyboardInterrupt:
                    print("\n\nInterrupted!")
                    self._stop_audio()  # Stop any playing audio
                    if self.changes_made:
                        save = input("\nSave changes before quitting? (y/n): ").strip().lower()
                        if save in ["y", "yes"]:
                            self._save_candidates()
                    break
        finally:
            # Ensure audio is stopped when exiting
            self._stop_audio()

    def _parse_target_class(self, prediction: str) -> str:
        """Parse the target class from prediction field.

        Args:
            prediction: Prediction string like "Guitar" or "not_Guitar"

        Returns:
            Target class name (e.g., "Guitar")
        """
        if prediction.startswith("not_"):
            return prediction[4:]  # Remove "not_" prefix
        return prediction

    def auto_label(self):
        """Automatically label all candidates using ground truth."""
        print(f"\nAuto-labeling candidates using ground truth - {self.dataset_name.upper()}")
        print("=" * 60)

        if not self.candidates:
            print("No candidates to label!")
            return

        # Get target class from first candidate's prediction
        first_prediction = self.candidates[0].get("prediction", "")
        if not first_prediction:
            print("Error: No prediction field found in candidates CSV")
            return

        target_class = self._parse_target_class(first_prediction)
        print(f"Target class: {target_class}")

        positive_count = 0
        negative_count = 0

        for i, candidate in enumerate(self.candidates):
            # Use the true_is_positive field instead of original_class comparison
            # This field is already correctly computed during inference
            true_is_positive = candidate.get("true_is_positive", "")

            # Convert string representation to label
            if str(true_is_positive).lower() == "true":
                ground_truth_label = "1"
                positive_count += 1
            else:
                ground_truth_label = "0"
                negative_count += 1

            # Set the label
            candidate["needs_human_label"] = ground_truth_label

            # Show progress
            if (i + 1) % 10 == 0 or i == len(self.candidates) - 1:
                print(f"Processed {i + 1}/{len(self.candidates)} samples...")

        self.changes_made = True

        # Print summary
        print("\nAuto-labeling complete:")
        print(f"  Positive samples: {positive_count}")
        print(f"  Negative samples: {negative_count}")
        print(f"  Total samples: {len(self.candidates)}")

        # Save the results
        self._save_candidates()
        print(f"Results saved to: {self.candidates_csv}")


def test_audio_playback(audio_file):
    """Test audio playback with a specific file."""
    print(f"\nTesting audio playback with: {audio_file}")

    if not os.path.exists(audio_file):
        print("File not found")
        return

    if sys.platform == "darwin":
        try:
            result = subprocess.run(["afplay", audio_file], capture_output=True, text=True)
            if result.returncode == 0:
                print("Audio playback successful")
            else:
                print(f"Audio playback failed: {result.stderr}")
        except Exception as e:
            print(f"Error: {e}")


def main():
    parser = argparse.ArgumentParser(
        description="Audio labeling tool for active learning with multi-dataset support",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Example workflows:

FSD50K (default):
  # Use default dataset (or set AUDIOLOOP_DATASET=fsd50k)
  python -m audioloop.label_audio outputs/labeling_candidates_v1.csv

  # Explicit dataset specification
  python -m audioloop.label_audio outputs/labeling_candidates_v1.csv --dataset fsd50k --audio-dir data/FSD50K/FSD50K.dev_audio

UrbanSound8K:
  # Set environment variable
  AUDIOLOOP_DATASET=urbansound8k python -m audioloop.label_audio outputs/labeling_candidates_v1.csv

  # Explicit dataset specification
  python -m audioloop.label_audio outputs/labeling_candidates_v1.csv --dataset urbansound8k --audio-dir data/urbansound8k

Full workflow example:
  # Run active learning to generate candidates
  python -m audioloop.active_learning --class-name dog_bark --run-number 1

  # Label the candidates (uses AUDIOLOOP_DATASET or default)
  python -m audioloop.label_audio outputs/labeling_candidates_v1.csv

  # Merge labels back into training set
  python -m audioloop.merge_labels training_sets/training_set_v1.csv outputs/labeling_candidates_v1.csv

Automated labeling (for testing):
  # Automatically label using ground truth instead of human input
  python -m audioloop.label_audio outputs/labeling_candidates_v1.csv --auto-label

  # With explicit dataset
  python -m audioloop.label_audio outputs/labeling_candidates_v1.csv --auto-label --dataset fsd50k

Requirements:
  - Candidates CSV must come from active_learning.py (has 'filepath' column)
  - Audio files must be in proper dataset structure:
    * UrbanSound8K: fold1-fold10 directories
    * FSD50K: flat directory structure
  - For custom audio directories, use --audio-dir

Audio playback:
  - macOS: uses built-in afplay
  - Linux: install sox ('sudo apt-get install sox') for play command
  - Windows: uses default audio player

Dataset to use ('fsd50k' or 'urbansound8k'). Can also be set via AUDIOLOOP_DATASET environment variable.
        """,
    )

    parser.add_argument(
        "candidates_csv", help="Path to labeling candidates CSV from active learning"
    )

    # Dynamic dataset discovery
    from audioloop.datasets.dataset_registry import list_available_datasets
    available_datasets = list_available_datasets()
    
    parser.add_argument(
        "--dataset",
        choices=available_datasets,
        help=f"Dataset type. Available: {', '.join(available_datasets)}. Uses AUDIOLOOP_DATASET env var or default if not specified.",
    )

    parser.add_argument(
        "--audio-dir",
        help="Audio directory (uses dataset default if not specified)",
    )

    parser.add_argument(
        "--test-audio", help="Test audio playback with a specific file", metavar="FILE"
    )

    parser.add_argument(
        "--auto-label",
        action="store_true",
        help="Automatically label candidates using ground truth (for testing)",
    )

    args = parser.parse_args()

    # Handle dataset resolution
    config = AudioLoopConfig(dataset=args.dataset)
    dataset_name = config.dataset

    # Handle test mode
    if args.test_audio:
        test_audio_playback(args.test_audio)
        return

    if not os.path.exists(args.candidates_csv):
        print(f"Error: Candidates file not found: {args.candidates_csv}")
        sys.exit(1)

    # Verify it's from active learning workflow
    with open(args.candidates_csv) as f:
        reader = csv.DictReader(f)
        first_row = next(reader, None)
        if first_row and "filepath" not in first_row:
            print("Error: This tool requires a candidates CSV from the active learning workflow.")
            print("The CSV must have a 'filepath' column.")
            print(f"Found columns: {list(first_row.keys())}")
            sys.exit(1)

    # Create labeler and run
    labeler = SimpleAudioLabeler(
        args.candidates_csv, dataset_name=dataset_name, audio_dir=args.audio_dir
    )

    try:
        if args.auto_label:
            labeler.auto_label()
        else:
            labeler.run()
    except Exception as e:
        print(f"\nError: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
