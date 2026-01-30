"""
Launch the AudioLoop web UI from a project directory.

Usage:
    cd ~/projects/my-classifier
    python -m audioloop.webui
    python -m audioloop.webui --host 0.0.0.0  # Allow external connections
    python -m audioloop.webui --port 8080     # Custom port

The web UI will detect the project from the current working directory
and auto-populate available candidate files for labeling.
"""

import argparse
import os
import sys
from pathlib import Path

DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 5000


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
    # Check if this is a Flask reloader restart (env vars already set)
    is_reloader_restart = "AUDIOLOOP_PROJECT_ROOT" in os.environ

    if is_reloader_restart:
        project_root = Path(os.environ["AUDIOLOOP_PROJECT_ROOT"])
        host = os.environ.get("AUDIOLOOP_WEBUI_HOST", DEFAULT_HOST)
        port = int(os.environ.get("AUDIOLOOP_WEBUI_PORT", DEFAULT_PORT))
    else:
        # Parse command-line arguments
        parser = argparse.ArgumentParser(
            description="Launch the AudioLoop web labeling interface"
        )
        parser.add_argument(
            "--host",
            default=DEFAULT_HOST,
            help=f"Host address (default: {DEFAULT_HOST}, use 0.0.0.0 for external access)",
        )
        parser.add_argument(
            "--port",
            type=int,
            default=DEFAULT_PORT,
            help=f"Port number (default: {DEFAULT_PORT})",
        )
        args = parser.parse_args()
        host = args.host
        port = args.port

        # Detect project
        project_root = detect_project_root()

        if project_root is None:
            print("Warning: Current directory doesn't look like an AudioLoop project.")
            print("  (No audioloop.yaml or outputs/ found)")
            print("  The web UI will still run, but candidate auto-detection won't work.")
            print()
            project_root = Path.cwd()
        else:
            print(f"Detected project: {project_root}")

        # Set environment so app.py knows the project root (and for reloader)
        os.environ["AUDIOLOOP_PROJECT_ROOT"] = str(project_root)
        os.environ["AUDIOLOOP_WEBUI_HOST"] = host
        os.environ["AUDIOLOOP_WEBUI_PORT"] = str(port)
        os.environ.setdefault("AUDIOLOOP_OUTPUT_ROOT", str(project_root))

    # Import and run the Flask app
    webui_dir = Path(__file__).parent.parent.parent / "webui"

    if not webui_dir.exists():
        print(f"Error: webui directory not found at {webui_dir}")
        sys.exit(1)

    # Change to webui dir so Flask can find templates/static
    original_cwd = os.getcwd()
    os.chdir(webui_dir)

    sys.path.insert(0, str(webui_dir))

    try:
        from app import app  # type: ignore[import-not-found]

        if not is_reloader_restart:
            print(f"Starting web UI at http://{host}:{port}")
            print(f"Project root: {project_root}")
            print()

        app.run(debug=True, host=host, port=port)
    finally:
        os.chdir(original_cwd)


if __name__ == "__main__":
    main()
