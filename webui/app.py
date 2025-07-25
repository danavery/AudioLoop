#!/usr/bin/env python3
"""
Simple Flask web UI for AudioLoop audio labeling.

This provides a web-based interface that reuses the existing SimpleAudioLabeler
logic but presents it through a modern web interface instead of terminal commands.
"""

import os
import sys
from flask import Flask, render_template, request, jsonify, send_file
from pathlib import Path

# Add the parent directory to Python path so we can import audioloop
sys.path.insert(0, str(Path(__file__).parent.parent))

from audioloop.label_audio import SimpleAudioLabeler

app = Flask(__name__)

# Global labeler instance (simple approach for now)
current_labeler = None

@app.route('/')
def index():
    """Main labeling interface."""
    return render_template('labeling.html')

@app.route('/api/load', methods=['POST'])
def load_candidates():
    """Load a candidates CSV file."""
    global current_labeler
    
    data = request.json
    candidates_csv = data.get('candidates_csv')
    dataset = data.get('dataset', 'fsd50k')  # Default to fsd50k
    audio_dir = data.get('audio_dir')
    
    if not candidates_csv or not os.path.exists(candidates_csv):
        return jsonify({'error': 'Candidates CSV file not found'}), 400
    
    try:
        # Create labeler instance (reusing existing logic)
        current_labeler = SimpleAudioLabeler(
            candidates_csv=candidates_csv,
            dataset_name=dataset,
            audio_dir=audio_dir
        )
        
        # Return basic info about loaded candidates
        labeled_count = sum(
            1 for c in current_labeler.candidates 
            if c.get("needs_human_label", "").strip() != ""
        )
        
        return jsonify({
            'success': True,
            'total_candidates': len(current_labeler.candidates),
            'labeled_count': labeled_count,
            'dataset': dataset
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/candidate/<int:index>')
def get_candidate(index):
    """Get candidate information by index."""
    if not current_labeler or index >= len(current_labeler.candidates):
        return jsonify({'error': 'Invalid candidate index'}), 400
    
    candidate = current_labeler.candidates[index]
    
    # Calculate progress
    labeled_count = sum(
        1 for c in current_labeler.candidates 
        if c.get("needs_human_label", "").strip() != ""
    )
    
    return jsonify({
        'index': index,
        'total': len(current_labeler.candidates),
        'labeled_count': labeled_count,
        'candidate': {
            'filename': candidate.get('filename', 'Unknown'),
            'filepath': candidate.get('filepath', 'Unknown'),
            'prediction': candidate.get('prediction', 'Unknown'),
            'confidence': candidate.get('confidence', 'Unknown'),
            'original_class': candidate.get('original_class', 'Unknown'),
            'current_label': candidate.get("needs_human_label", "").strip()
        }
    })

@app.route('/api/audio/<int:index>')
def serve_audio(index):
    """Serve audio file for a candidate."""
    if not current_labeler or index >= len(current_labeler.candidates):
        return jsonify({'error': 'Invalid candidate index'}), 400
    
    candidate = current_labeler.candidates[index]
    audio_path = current_labeler._get_audio_path(candidate)
    
    if not audio_path or not os.path.exists(audio_path):
        return jsonify({'error': 'Audio file not found'}), 404
    
    return send_file(audio_path)

@app.route('/api/label', methods=['POST'])
def label_candidate():
    """Label a candidate."""
    if not current_labeler:
        return jsonify({'error': 'No candidates loaded'}), 400
    
    data = request.json
    index = data.get('index')
    label = data.get('label')  # "1" for positive, "0" for negative
    
    if index is None or index >= len(current_labeler.candidates):
        return jsonify({'error': 'Invalid candidate index'}), 400
    
    if label not in ["0", "1"]:
        return jsonify({'error': 'Label must be "0" or "1"'}), 400
    
    # Apply the label
    current_labeler.candidates[index]["needs_human_label"] = label
    current_labeler.changes_made = True
    
    return jsonify({'success': True, 'label': label})

@app.route('/api/save', methods=['POST'])
def save_labels():
    """Save current labels to CSV."""
    if not current_labeler:
        return jsonify({'error': 'No candidates loaded'}), 400
    
    try:
        current_labeler._save_candidates()
        return jsonify({'success': True, 'message': 'Labels saved successfully'})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/next_unlabeled/<int:start_index>')
def find_next_unlabeled(start_index):
    """Find the next unlabeled candidate."""
    if not current_labeler:
        return jsonify({'error': 'No candidates loaded'}), 400
    
    # Search from start_index forward
    for i in range(start_index, len(current_labeler.candidates)):
        if current_labeler.candidates[i].get("needs_human_label", "").strip() == "":
            return jsonify({'next_index': i})
    
    # Wrap around to beginning
    for i in range(0, start_index):
        if current_labeler.candidates[i].get("needs_human_label", "").strip() == "":
            return jsonify({'next_index': i})
    
    return jsonify({'next_index': None})  # All labeled

if __name__ == '__main__':
    app.run(debug=True, port=5000)