"""Command-line interface: ``maci run | config | data | info``."""

import argparse
import os
import sys
import warnings
from typing import List, Optional

from . import __version__
from .config import (BCI_SCORE_TYPES, BINNING_METHODS, CONFIG_TEMPLATE, DATASETS, MACI_MODES, METHODS,
                     SOLVERS, Config, dump_config, load_config)
from .data import DATA_FILES, data_status, default_data_dir, download_data
from .groupers import GROUPER_ALIASES

DEFAULT_CONFIG_NAME = 'maci.toml'

# (flag, config section, config key, argparse kwargs). Flag spellings from the
# original experiments/run_experiment.py are kept as aliases.
_RUN_FLAGS = [
    (['--dataset', '--dataset-type'], 'data', 'dataset', dict(choices=DATASETS)),
    (['--data-dir'], 'data', 'data_dir', dict(metavar='DIR')),
    (['--scores-dir'], 'data', 'scores_dir', dict(metavar='DIR')),
    (['--view-metadata-csv'], 'data', 'view_metadata_csv', dict(metavar='CSV')),
    (['--limit-samples'], 'data', 'limit_samples', dict(type=int, metavar='N')),
    (['--methods'], 'experiment', 'methods', dict(nargs='+', choices=METHODS)),
    (['--alpha'], 'experiment', 'alpha', dict(type=float)),
    (['--n-runs'], 'experiment', 'n_runs', dict(type=int, metavar='N')),
    (['--random-seed'], 'experiment', 'random_seed', dict(type=int, metavar='SEED')),
    (['--calib-ratio'], 'experiment', 'calib_ratio', dict(type=float)),
    (['--test-ratio'], 'experiment', 'test_ratio', dict(type=float)),
    (['--conditional-groups'], 'experiment', 'conditional_groups',
     dict(nargs='*', choices=('false_claim_risk', 'medical_content', 'view_count', *GROUPER_ALIASES))),
    (['--binning-method'], 'experiment', 'binning_method', dict(choices=BINNING_METHODS)),
    (['--models'], 'maci', 'models', dict(nargs='+', metavar='MODEL')),
    (['--maci-mode', '--as-mode'], 'maci', 'mode', dict(choices=MACI_MODES)),
    (['--n-bins'], 'maci', 'n_bins', dict(type=int)),
    (['--ensemble-train-ratio'], 'maci', 'ensemble_train_ratio', dict(type=float)),
    (['--target-tpr'], 'maci', 'target_tpr', dict(type=float)),
    (['--solver'], 'maci', 'solver', dict(choices=SOLVERS)),
    (['--bci-score-type', '--bcp-score-type'], 'bci', 'score_type', dict(choices=BCI_SCORE_TYPES)),
    (['--boosting-epochs'], 'cci', 'boosting_epochs', dict(type=int, metavar='N')),
    (['--boosting-lr'], 'cci', 'boosting_lr', dict(type=float)),
    (['--adaptive-alpha'], 'cci', 'adaptive_alpha', dict(action='store_true')),
    (['--retention-target'], 'cci', 'retention_target', dict(type=float)),
    (['--output-dir'], 'output', 'output_dir', dict(metavar='DIR')),
    (['--log-dir'], 'output', 'log_dir', dict(metavar='DIR')),
    (['--results-dir', '--time-out'], 'output', 'results_dir', dict(metavar='DIR')),
    (['--time-profile'], 'output', 'time_profile', dict(action='store_true')),
]


def _dest(section: str, key: str) -> str:
    return f"{section}__{key}"


def _add_config_flags(parser: argparse.ArgumentParser) -> None:
    parser.add_argument('-c', '--config', metavar='FILE',
                        help=f"TOML config file (default: ./{DEFAULT_CONFIG_NAME} if it exists)")
    groups = {}
    for flags, section, key, kwargs in _RUN_FLAGS:
        if section not in groups:
            groups[section] = parser.add_argument_group(f"[{section}] overrides")
        groups[section].add_argument(*flags, dest=_dest(section, key), default=None,
                                     help=f"{section}.{key}", **kwargs)
    # Accepted for compatibility with the original script; it only ever had one choice.
    parser.add_argument('--model-set', choices=['fixed'], help=argparse.SUPPRESS)


def _resolve_config(args: argparse.Namespace) -> Config:
    path = args.config
    if path is None and os.path.exists(DEFAULT_CONFIG_NAME):
        path = DEFAULT_CONFIG_NAME
    cfg = load_config(path)
    for _, section, key, _ in _RUN_FLAGS:
        value = getattr(args, _dest(section, key))
        if value is None:
            continue
        if key.endswith('_dir') or key.endswith('_csv'):
            value = os.path.abspath(os.path.expanduser(value))
        setattr(getattr(cfg, section), key, value)
    return cfg.validate()


def cmd_run(args: argparse.Namespace) -> int:
    import numpy as np

    from .experiment import log_to, run_experiments

    cfg = _resolve_config(args)
    with warnings.catch_warnings(), np.errstate(all='warn'):
        warnings.filterwarnings('default')
        warnings.simplefilter('ignore', category=FutureWarning)
        with log_to(cfg.output.resolved_log_dir):
            run_experiments(cfg)
    return 0


def cmd_config_init(args: argparse.Namespace) -> int:
    path = args.path
    if os.path.exists(path) and not args.force:
        print(f"{path} already exists; pass --force to overwrite.", file=sys.stderr)
        return 1
    with open(path, 'w', encoding='utf-8') as f:
        f.write(CONFIG_TEMPLATE)
    print(f"Wrote {path}. Edit it, then run: maci run --config {path}")
    return 0


def cmd_config_show(args: argparse.Namespace) -> int:
    print(dump_config(_resolve_config(args)), end='')
    return 0


def _expand_datasets(names: List[str]) -> List[str]:
    if not names or 'all' in names:
        return list(DATA_FILES)
    return names


def cmd_data_download(args: argparse.Namespace) -> int:
    import logging
    logging.basicConfig(level=logging.INFO, format='%(message)s')
    datasets = _expand_datasets(args.datasets or ['wikibio'])
    data_dir = download_data(datasets, data_dir=args.data_dir, force=args.force)
    print(f"Data ready in {data_dir}")
    if 'medlfqa' in datasets:
        print("Note: the MedLFQA claim dataset (med_scores/medlfqa_dataset.pkl) is not published yet; "
              "only its score files were downloaded.")
    if args.data_dir is None and 'MACI_DATA_DIR' not in os.environ:
        print("maci run will find it there by default.")
    return 0


def cmd_data_status(args: argparse.Namespace) -> int:
    data_dir = os.path.abspath(os.path.expanduser(args.data_dir or default_data_dir()))
    print(f"Data directory: {data_dir}")
    all_ok = True
    for name, files in data_status(data_dir, _expand_datasets(args.datasets)).items():
        print(f"\n{name}:")
        for rel_path, state in files.items():
            mark = '✅' if state == 'ok' else '❌'
            all_ok &= state == 'ok'
            print(f"  {mark} {rel_path}" + ('' if state == 'ok' else f"  ({state})"))
    return 0 if all_ok else 1


def cmd_info(args: argparse.Namespace) -> int:
    from importlib import metadata

    print(f"maci-conformal {__version__}")
    print(f"Python {sys.version.split()[0]} ({sys.executable})")
    print(f"Default data directory: {default_data_dir()}")
    for dist in ('numpy', 'scipy', 'scikit-learn', 'pandas', 'cvxpy', 'torch', 'conditionalconformal'):
        try:
            version = metadata.version(dist)
        except metadata.PackageNotFoundError:
            version = 'not installed'
        print(f"  {dist:<22} {version}")
    try:
        import cvxpy
        print(f"  cvxpy solvers: {', '.join(cvxpy.installed_solvers())}")
    except ImportError:
        pass
    try:
        from . import conditional  # noqa: F401
        print("CCI baseline: available")
    except ImportError:
        print('CCI baseline: unavailable (pip install torch conditionalconformal)')
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog='maci',
        description='Multi-LLM Adaptive Conformal Inference (MACI) for claim-level factuality filtering.',
    )
    parser.add_argument('-V', '--version', action='version', version=f"%(prog)s {__version__}")
    sub = parser.add_subparsers(dest='command', metavar='COMMAND')
    sub.required = True

    p_run = sub.add_parser('run', help='run the BCI / CCI / MACI benchmark',
                           description='Run the benchmark. Settings come from the config file, '
                                       'and any flag below overrides it.')
    _add_config_flags(p_run)
    p_run.set_defaults(func=cmd_run)

    p_cfg = sub.add_parser('config', help='create or inspect a config file')
    cfg_sub = p_cfg.add_subparsers(dest='config_command', metavar='ACTION')
    cfg_sub.required = True
    p_init = cfg_sub.add_parser('init', help=f'write a commented {DEFAULT_CONFIG_NAME} template')
    p_init.add_argument('path', nargs='?', default=DEFAULT_CONFIG_NAME)
    p_init.add_argument('-f', '--force', action='store_true', help='overwrite an existing file')
    p_init.set_defaults(func=cmd_config_init)
    p_show = cfg_sub.add_parser('show', help='print the effective config (file + flags)')
    _add_config_flags(p_show)
    p_show.set_defaults(func=cmd_config_show)

    p_data = sub.add_parser('data', help='download or check the benchmark data')
    data_sub = p_data.add_subparsers(dest='data_command', metavar='ACTION')
    data_sub.required = True
    choices = [*DATA_FILES, 'all']
    p_dl = data_sub.add_parser('download', help='download benchmark files (default: wikibio)')
    p_dl.add_argument('datasets', nargs='*', choices=choices, metavar='DATASET',
                      help=f"one or more of: {', '.join(choices)}")
    p_dl.add_argument('--data-dir', metavar='DIR', help=f'destination (default: {default_data_dir()})')
    p_dl.add_argument('-f', '--force', action='store_true', help='re-download files that are already present')
    p_dl.set_defaults(func=cmd_data_download)
    p_st = data_sub.add_parser('status', help='check which benchmark files are present and intact')
    p_st.add_argument('datasets', nargs='*', choices=choices, metavar='DATASET')
    p_st.add_argument('--data-dir', metavar='DIR')
    p_st.set_defaults(func=cmd_data_status)

    p_info = sub.add_parser('info', help='show versions, data location and optional features')
    p_info.set_defaults(func=cmd_info)
    return parser


def main(argv: Optional[List[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return args.func(args)
    except (ValueError, FileNotFoundError) as e:
        print(f"maci: error: {e}", file=sys.stderr)
        return 2


if __name__ == '__main__':
    sys.exit(main())
