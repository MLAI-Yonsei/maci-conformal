"""Backward-compatible entry point; equivalent to `maci run` with repo-relative defaults.

Prefer the installed CLI:  pip install -e .  &&  maci run --help
"""

import os
import sys

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, os.path.join(REPO_ROOT, 'src'))

from maci_conformal.cli import main  # noqa: E402

if __name__ == "__main__":
    defaults = [
        '--data-dir', os.path.join(REPO_ROOT, 'data'),
        '--log-dir', os.path.join(REPO_ROOT, 'logs'),
        '--results-dir', os.path.join(REPO_ROOT, 'analysis', 'experiment_results'),
    ]
    # Flags given on the command line come later and therefore win.
    sys.exit(main(['run', *defaults, *sys.argv[1:]]))
