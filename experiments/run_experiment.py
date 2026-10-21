"""Backward-compatible entry point; equivalent to `maci run` with repo-relative defaults.

Prefer the installed CLI:  pip install -e .  &&  maci run --help
"""

import os
import sys

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, os.path.join(REPO_ROOT, 'src'))

from maci_conformal.cli import main  # noqa: E402

if __name__ == "__main__":
    # Used only for settings that neither a config file nor a flag provides.
    repo_defaults = {
        ('data', 'data_dir'): os.path.join(REPO_ROOT, 'data'),
        ('output', 'log_dir'): os.path.join(REPO_ROOT, 'logs'),
        ('output', 'results_dir'): os.path.join(REPO_ROOT, 'analysis', 'experiment_results'),
    }
    sys.exit(main(['run', *sys.argv[1:]], run_defaults=repo_defaults))
