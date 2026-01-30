"""
Launch the AudioLoop web UI from a project directory.

Usage:
    cd ~/projects/my-classifier
    python -m audioloop.webui

The web UI will detect the project from the current working directory
and auto-populate available candidate files for labeling.
"""

import os
import sys
from pathlib import Path


def detect_project_root() -> Path | None:
    """Detect if cwd is an AudioLoop project directory.

    Returns the project root if found, None otherwise.
    Looks for audioloop.yaml or outputs/ directory as indicators.
    """
    cwd = Path.cwd()

    # Check for project indicators
    if (cwd / "audioloop.yaml").exists():
        return cwd
    if (cwd / "outputs").is_dir():
        return cwd
    if (cwd / "training_sets").is_dir():
        return cwd

    return None


def main():
    """Launch the web UI with project context."""
    project_root = detect_project_root()

    if project_root is None:
        print("Warning: Current directory doesn't look like an AudioLoop project.")
        print("  (No audioloop.yaml or outputs/ found)")
        print("  The web UI will still run, but candidate auto-detection won't work.")
        print()
        project_root = Path.cwd()
    else:
        print(f"Detected project: {project_root}")

    # Set environment so app.py knows the project root
    os.environ["AUDIOLOOP_PROJECT_ROOT"] = str(project_root)
    os.environ.setdefault("AUDIOLOOP_OUTPUT_ROOT", str(project_root))

    # Import and run the Flask app
    # We need to add the webui directory to find the app module
    webui_dir = Path(__file__).parent.parent.parent / "webui"

    if not webui_dir.exists():
        print(f"Error: webui directory not found at {webui_dir}")
        sys.exit(1)

    # Change to webui dir so Flask can find templates/static
    # (We'll fix this properly in app.py, but this ensures it works)
    original_cwd = os.getcwd()
    os.chdir(webui_dir)

    sys.path.insert(0, str(webui_dir))

    try:
        from app import app  # type: ignore[import-not-found]
        print(f"Starting web UI at http://127.0.0.1:5000")
        print(f"Project root: {project_root}")
        print()
        app.run(debug=True, host="127.0.0.1", port=5000)
    finally:
        os.chdir(original_cwd)


if __name__ == "__main__":
    main()
