#!/usr/bin/env python3
"""
Simple Flask web UI for AudioLoop audio labeling.

This provides a web-based interface that reuses the existing SimpleAudioLabeler
logic but presents it through a modern web interface instead of terminal commands.
"""

import os
import sys
from pathlib import Path

from flask import Flask, jsonify, render_template, request, send_file

# Add the parent directory to Python path so we can import audioloop
sys.path.insert(0, str(Path(__file__).parent.parent))

# Set default output root to project root (webui runs from subdirectory)
# Users can override by setting AUDIOLOOP_OUTPUT_ROOT env var before running
if "AUDIOLOOP_OUTPUT_ROOT" not in os.environ:
    os.environ["AUDIOLOOP_OUTPUT_ROOT"] = str(Path(__file__).parent.parent)

from audioloop.label_audio import SimpleAudioLabeler

app = Flask(__name__)

# Global labeler instance (simple approach for now)
current_labeler = None


@app.route("/")
def index():
    """Main labeling interface."""
    return render_template("labeling.html")


@app.route("/api/datasets")
def get_datasets():
    """Get list of available datasets from the registry."""
    from audioloop.datasets.dataset_registry import list_available_datasets

    datasets = list_available_datasets()
    return jsonify({"datasets": datasets})


@app.route("/api/load", methods=["POST"])
def load_candidates():
    """Load a candidates CSV file."""
    global current_labeler

    data = request.json
    if data is None:
        return jsonify({"error": "No JSON data provided"}), 400
    candidates_csv = data.get("candidates_csv")
    dataset = data.get("dataset", "fsd50k")  # Default to fsd50k
    audio_dir = data.get("audio_dir")

    # Resolve candidates CSV relative to base outputs directory
    # This allows users to type just "labeling_candidates_v1.csv" or "myexp/labeling_candidates_v1.csv"
    from audioloop.utils.paths import get_output_dir

    if candidates_csv:
        csv_path = Path(candidates_csv)
        if not csv_path.is_absolute():
            # Resolve relative to base outputs directory (not experiment-specific)
            base_outputs_dir = get_output_dir()  # Returns outputs/ without experiment suffix
            candidates_csv = str(base_outputs_dir / csv_path)
        else:
            candidates_csv = str(csv_path)

    # Resolve audio_dir relative to project root if needed
    if audio_dir:
        audio_path = Path(audio_dir)
        if not audio_path.is_absolute():
            project_root = Path(__file__).parent.parent
            audio_dir = str(project_root / audio_path)
        else:
            audio_dir = str(audio_path)

    if not candidates_csv or not os.path.exists(candidates_csv):
        return jsonify({"error": "Candidates CSV file not found"}), 400

    try:
        # Create labeler instance (reusing existing logic)
        current_labeler = SimpleAudioLabeler(
            candidates_csv=candidates_csv, dataset_name=dataset, audio_dir=audio_dir
        )

        # Return basic info about loaded candidates
        labeled_count = sum(
            1 for c in current_labeler.candidates if c.get("needs_human_label", "").strip() != ""
        )

        return jsonify(
            {
                "success": True,
                "total_candidates": len(current_labeler.candidates),
                "labeled_count": labeled_count,
                "dataset": dataset,
            }
        )

    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/candidate/<int:index>")
def get_candidate(index):
    """Get candidate information by index."""
    if not current_labeler or index >= len(current_labeler.candidates):
        return jsonify({"error": "Invalid candidate index"}), 400

    candidate = current_labeler.candidates[index]

    # Calculate progress
    labeled_count = sum(
        1 for c in current_labeler.candidates if c.get("needs_human_label", "").strip() != ""
    )

    return jsonify(
        {
            "index": index,
            "total": len(current_labeler.candidates),
            "labeled_count": labeled_count,
            "candidate": {
                "filename": candidate.get("filename", "Unknown"),
                "filepath": candidate.get("filepath", "Unknown"),
                "prediction": candidate.get("predicted_class", "Unknown"),
                "target_class": candidate.get("target_class", "Unknown"),
                "confidence": candidate.get("confidence", "Unknown"),
                "original_class": candidate.get("original_class", "Unknown"),
                "current_label": candidate.get("needs_human_label", "").strip(),
            },
        }
    )


@app.route("/api/audio/<int:index>")
def serve_audio(index):
    """Serve audio file for a candidate."""
    if not current_labeler or index >= len(current_labeler.candidates):
        return jsonify({"error": "Invalid candidate index"}), 400

    candidate = current_labeler.candidates[index]
    audio_path = current_labeler._get_audio_path(candidate)

    if not audio_path or not os.path.exists(audio_path):
        return jsonify({"error": "Audio file not found"}), 404

    return send_file(audio_path, mimetype="audio/wav")


@app.route("/api/label", methods=["POST"])
def label_candidate():
    """Label a candidate."""
    if not current_labeler:
        return jsonify({"error": "No candidates loaded"}), 400

    data = request.json
    if data is None:
        return jsonify({"error": "No JSON data provided"}), 400
    index = data.get("index")
    label = data.get("label")  # "1" for positive, "0" for negative

    if index is None or index >= len(current_labeler.candidates):
        return jsonify({"error": "Invalid candidate index"}), 400

    if label not in ["0", "1"]:
        return jsonify({"error": 'Label must be "0" or "1"'}), 400

    # Apply the label
    current_labeler.candidates[index]["needs_human_label"] = label
    current_labeler.changes_made = True

    return jsonify({"success": True, "label": label})


@app.route("/api/save", methods=["POST"])
def save_labels():
    """Save current labels to CSV."""
    if not current_labeler:
        return jsonify({"error": "No candidates loaded"}), 400

    try:
        current_labeler._save_candidates()
        return jsonify({"success": True, "message": "Labels saved successfully"})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/next_unlabeled/<int:start_index>")
def find_next_unlabeled(start_index):
    """Find the next unlabeled candidate."""
    if not current_labeler:
        return jsonify({"error": "No candidates loaded"}), 400

    # Search from start_index forward
    for i in range(start_index, len(current_labeler.candidates)):
        if current_labeler.candidates[i].get("needs_human_label", "").strip() == "":
            return jsonify({"next_index": i})

    # Wrap around to beginning
    for i in range(0, start_index):
        if current_labeler.candidates[i].get("needs_human_label", "").strip() == "":
            return jsonify({"next_index": i})

    return jsonify({"next_index": None})  # All labeled


if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)
