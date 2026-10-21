"""Command-line interface: ``maci run | models | score | calibrate | check | inspect | config | data | info``."""

import argparse
import difflib
import logging
import os
import sys
import warnings
from typing import Any, Dict, List, Optional, Sequence

from . import __version__
from .config import (BCI_SCORE_TYPES, BINNING_METHODS, CONFIG_TEMPLATE, DATASETS, GROUPS, MACI_MODES, METHODS,
                     SOLVERS, Config, _load_toml, config_from_dict, dump_config)
from .data import DATA_FILES, data_status, default_data_dir, download_data
from .groupers import GROUPER_ALIASES

DEFAULT_CONFIG_NAME = 'maci.toml'
DEFAULT_CALIBRATOR = 'maci_calibrator.json'
DEFAULT_SCORES_DIR = 'maci_scores'

_GROUP_CHOICES = (*GROUPS, *GROUPER_ALIASES)

# (flags, config section, config key, argparse kwargs). Flag spellings from the
# original experiments/run_experiment.py are kept as aliases.
_FLAGS = [
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
    (['--conditional-groups'], 'experiment', 'conditional_groups', dict(nargs='+', choices=_GROUP_CHOICES)),
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
    (['--adaptive-alpha'], 'cci', 'adaptive_alpha', dict(action=argparse.BooleanOptionalAction)),
    (['--retention-target'], 'cci', 'retention_target', dict(type=float)),
    (['--base-url'], 'llm', 'base_url', dict(metavar='URL')),
    (['--api-key-env'], 'llm', 'api_key_env', dict(metavar='VAR')),
    (['--extractor-model'], 'llm', 'extractor_model', dict(metavar='MODEL')),
    (['--temperature'], 'llm', 'temperature', dict(type=float)),
    (['--max-workers'], 'llm', 'max_workers', dict(type=int, metavar='N')),
    (['--cache'], 'llm', 'cache', dict(action=argparse.BooleanOptionalAction)),
    (['--cache-dir'], 'llm', 'cache_dir', dict(metavar='DIR')),
    (['--output-dir'], 'output', 'output_dir', dict(metavar='DIR')),
    (['--log-dir'], 'output', 'log_dir', dict(metavar='DIR')),
    (['--results-dir', '--time-out'], 'output', 'results_dir', dict(metavar='DIR')),
    (['--time-profile'], 'output', 'time_profile', dict(action=argparse.BooleanOptionalAction)),
    (['--predictions'], 'output', 'save_predictions', dict(action=argparse.BooleanOptionalAction)),
    (['--report'], 'output', 'report', dict(action=argparse.BooleanOptionalAction)),
    (['--report-samples'], 'output', 'report_samples', dict(type=int, metavar='N')),
]

_RUN_SECTIONS = ('data', 'experiment', 'maci', 'bci', 'cci', 'output')
_LLM_KEYS = {('maci', 'models'), ('data', 'data_dir')} | {('llm', k) for _, s, k, _ in _FLAGS if s == 'llm'}


def _dest(section: str, key: str) -> str:
    return f"{section}__{key}"


def _add_config_flags(parser: argparse.ArgumentParser, include) -> None:
    """Add --config plus the override flags whose (section, key) passes ``include``."""
    parser.add_argument('-c', '--config', metavar='FILE',
                        help=f"TOML config file (default: ./{DEFAULT_CONFIG_NAME} if it exists)")
    groups: Dict[str, Any] = {}
    for flags, section, key, kwargs in _FLAGS:
        if not include(section, key):
            continue
        if section not in groups:
            groups[section] = parser.add_argument_group(f"[{section}] overrides")
        groups[section].add_argument(*flags, dest=_dest(section, key), default=None,
                                     help=f"{section}.{key}", **kwargs)


_OUTPUT_LOCATION_KEYS = {('output', 'output_dir'), ('output', 'log_dir'), ('output', 'results_dir')}


def _resolve_config(args: argparse.Namespace, defaults: Optional[Dict[tuple, Any]] = None) -> Config:
    """Defaults < ``defaults`` (only for keys nobody set) < config file < command-line flags."""
    path = args.config
    if path is None and os.path.exists(DEFAULT_CONFIG_NAME):
        path = DEFAULT_CONFIG_NAME
    raw: Dict[str, Any] = {}
    cfg = Config()
    if path is not None:
        path = os.path.abspath(os.path.expanduser(path))
        if not os.path.exists(path):
            raise FileNotFoundError(f"Config file not found: {path}. Create one with `maci config init`.")
        raw = _load_toml(path)
        cfg = config_from_dict(raw, base_dir=os.path.dirname(path))
    explicit = {(sec, k) for sec, vals in raw.items() if isinstance(vals, dict) for k in vals}
    for _, section, key, _ in _FLAGS:
        value = getattr(args, _dest(section, key), None)
        if value is None:
            continue
        explicit.add((section, key))
        if key.endswith('_dir') or key.endswith('_csv'):
            value = os.path.abspath(os.path.expanduser(value))
        setattr(getattr(cfg, section), key, value)
    for (section, key), value in (defaults or {}).items():
        if (section, key) in explicit:
            continue
        if (section, key) in _OUTPUT_LOCATION_KEYS and explicit & _OUTPUT_LOCATION_KEYS:
            continue
        setattr(getattr(cfg, section), key, value)
    return cfg.validate()


def _quiet_library_logs(level: int = logging.WARNING) -> None:
    logging.basicConfig(level=level, format='%(message)s')
    logging.getLogger('maci_conformal').setLevel(level)


class _Progress:
    """One updating line on a terminal, a few lines otherwise."""

    def __init__(self, label: str):
        self.label = label
        self.tty = sys.stderr.isatty()
        self._last = -1

    def __call__(self, done: int, total: int) -> None:
        if self.tty:
            print(f"\r  {self.label} {done}/{total}", end='' if done < total else '\n', file=sys.stderr, flush=True)
        else:
            step = max(1, total // 10)
            if done == total or done // step != self._last:
                self._last = done // step
                print(f"  {self.label} {done}/{total}", file=sys.stderr, flush=True)


# ---------------------------------------------------------------------------
# run / inspect
# ---------------------------------------------------------------------------

def cmd_run(args: argparse.Namespace) -> int:
    import numpy as np

    from .experiment import log_to, prepare_output, run_experiments

    cfg = _resolve_config(args, getattr(args, 'run_defaults', None))
    paths = prepare_output(cfg)
    with warnings.catch_warnings(), np.errstate(all='warn'):
        warnings.filterwarnings('default')
        warnings.simplefilter('ignore', category=FutureWarning)
        with log_to(paths.log_file):
            out = run_experiments(cfg, paths)
    print()
    if out['results_path'] is None:
        print("maci: error: the results could not be saved (see the log above).", file=sys.stderr)
        return 1
    print(f"Results:     {out['results_path']}")
    if out['predictions_path']:
        print(f"Predictions: {out['predictions_path']}")
    if out['report_path']:
        print(f"Report:      {out['report_path']}  (open it in a browser)")
        print(f"Browse in the terminal: maci inspect {out['run_dir'] or out['predictions_path']}")
    if out['failed_methods']:
        print(f"maci: error: no successful runs for {', '.join(out['failed_methods'])}.", file=sys.stderr)
        return 1
    return 0


def _find_predictions(path: str) -> str:
    if os.path.isdir(path):
        candidate = os.path.join(path, 'predictions.jsonl')
        if os.path.exists(candidate):
            return candidate
        found = sorted(f for f in os.listdir(path) if f.startswith('predictions') and f.endswith('.jsonl'))
        if found:
            return os.path.join(path, found[-1])
        raise FileNotFoundError(f"No predictions file in {path}")
    if not os.path.exists(path):
        raise FileNotFoundError(f"Not found: {path}")
    return path


def cmd_inspect(args: argparse.Namespace) -> int:
    from .report import LEGEND, claim_outcomes, format_row, method_key, read_jsonl

    path = _find_predictions(args.path)
    rows = read_jsonl(path)
    if not rows:
        print(f"{path} has no predictions.")
        return 1
    runs = sorted({r.get('run', 1) for r in rows})
    run = args.run or runs[0]
    rows = [r for r in rows if r.get('run', 1) == run]
    if args.method:
        wanted = args.method.lower()
        rows = [r for r in rows if method_key(r).lower().startswith(wanted)]
    print(f"{path}  (run {run} of {len(runs)})\n")
    print(f"{'method':<28} {'false removed':>14} {'false kept':>11} {'answers w/ false kept':>22} {'true kept':>12}")
    for name, c in claim_outcomes(rows).items():
        n_false = c['false_kept'] + c['false_removed']
        n_true = c['true_kept'] + c['true_removed']
        print(f"{name:<28} {c['false_removed']:>7}/{n_false:<6} {c['false_kept']:>11} "
              f"{c['samples_with_false_kept']:>12}/{c['samples']:<9} {c['true_kept']:>6}/{n_true:<5}")
    if args.only_missed:
        rows = [r for r in rows if r.get('n_false_kept')]
    if args.search:
        needle = args.search.lower()
        rows = [r for r in rows if needle in r['prompt'].lower()
                or any(needle in c['text'].lower() for c in r['claims'])]
    by_prompt: Dict[str, List[Dict[str, Any]]] = {}
    for r in rows:
        by_prompt.setdefault(r['prompt'], []).append(r)
    print(f"\n{LEGEND}\n")
    for i, (prompt, group) in enumerate(list(by_prompt.items())[:args.limit]):
        print(f"[{i + 1}] {prompt}")
        for r in group:
            print('  ' + format_row(r, show_model_scores=args.show_model_scores).replace('\n', '\n  '))
        print()
    if len(by_prompt) > args.limit:
        print(f"... {len(by_prompt) - args.limit} more answers; use --limit, --search or --only-missed.")
    return 0


# ---------------------------------------------------------------------------
# OpenRouter: models / score / calibrate / check
# ---------------------------------------------------------------------------

def _client(cfg: Config):
    from .llm import LLMClient
    return LLMClient.from_config(cfg.llm)


def _model_index(client) -> Optional[Dict[str, Dict[str, Any]]]:
    """Model id -> info from the endpoint's /models list, or None if it has none."""
    from .llm import LLMError
    try:
        return {m['id']: m for m in client.list_models() if 'id' in m} or None
    except LLMError as e:
        if client.is_openrouter:
            print(f"(could not list models at {client.base_url}: {e})", file=sys.stderr)
        return None


def _validate_models(models: Sequence[str], index: Optional[Dict[str, Dict[str, Any]]]) -> None:
    if not index:
        return
    problems = []
    for m in models:
        if m not in index:
            close = difflib.get_close_matches(m, list(index), n=3, cutoff=0.5)
            problems.append(f"'{m}'" + (f" (did you mean {', '.join(close)}?)" if close else ""))
    if problems:
        raise ValueError("Unknown model id: " + "; ".join(problems) + ". Browse ids with `maci models <search words>`.")


def _confirm(plan: List[Dict[str, Any]], client, index, assume_yes: bool, what: str) -> None:
    from .llm import estimate_cost
    if not plan:
        print(f"All {what} requests are already cached; no API calls needed.")
        return
    per_model: Dict[str, int] = {}
    for r in plan:
        per_model[r['model']] = per_model.get(r['model'], 0) + 1
    est = estimate_cost(index, plan) if index else None
    print(f"About to send {len(plan)} {what} requests to {client.base_url}:")
    for m, n in per_model.items():
        print(f"  {m}: {n}")
    if est is not None:
        print(f"Estimated cost: about ${est:.2f} (rough; based on OpenRouter list prices)")
    client._key()  # after the estimate, so it is shown even before a key is set up
    if assume_yes:
        return
    if not sys.stdin.isatty():
        raise ValueError("Not running interactively; pass --yes to confirm the API requests.")
    if input("Continue? [y/N] ").strip().lower() not in ('y', 'yes'):
        raise KeyboardInterrupt


def cmd_models(args: argparse.Namespace) -> int:
    cfg = _resolve_config(args)
    client = _client(cfg)
    models = client.list_models()
    words = [w.lower() for w in args.query]

    def price(m, kind):
        try:
            return float((m.get('pricing') or {}).get(kind) or 0) * 1e6
        except ValueError:
            return float('nan')

    rows = [m for m in models if all(w in (m['id'] + ' ' + str(m.get('name', ''))).lower() for w in words)]
    if args.free:
        rows = [m for m in rows if price(m, 'prompt') == 0 and price(m, 'completion') == 0]
    sort_key = {'name': lambda m: m['id'], 'price': lambda m: (price(m, 'prompt') + price(m, 'completion'), m['id']),
                'context': lambda m: (-(m.get('context_length') or 0), m['id']),
                'newest': lambda m: (-(m.get('created') or 0), m['id'])}[args.sort]
    rows.sort(key=sort_key)
    if not rows:
        print("No models match." + (" Try fewer words." if words else ""))
        return 1
    print(f"{'model id':<52} {'context':>9} {'$/M in':>8} {'$/M out':>8}")
    for m in rows[:args.limit]:
        print(f"{m['id']:<52} {m.get('context_length') or 0:>9,} {price(m, 'prompt'):>8.3f} {price(m, 'completion'):>8.3f}")
    if len(rows) > args.limit:
        print(f"... {len(rows) - args.limit} more; add search words or --limit.")
    print("\nUse ids with --models, e.g.: maci calibrate --models " + ' '.join(m['id'] for m in rows[:2]))
    return 0


def cmd_score(args: argparse.Namespace) -> int:
    from .pipeline import SCORE_PREFIX, load_benchmark_samples, planned_requests, score_entries, write_score_files
    from .llm import SCORING_PROMPT

    _quiet_library_logs(logging.INFO)
    cfg = _resolve_config(args)
    dataset = 'wikibio' if cfg.data.dataset == 'auto' else cfg.data.dataset
    limit = args.limit or cfg.data.limit_samples
    client = _client(cfg)
    index = _model_index(client)
    _validate_models(cfg.maci.models, index)
    samples = load_benchmark_samples(dataset, cfg.data.data_dir, limit=limit)
    entries = [{'sample': s} for s in samples]
    template = cfg.llm.scoring_prompt or SCORING_PROMPT
    _confirm(planned_requests(client, cfg.maci.models, entries, template), client, index, args.yes, 'scoring')
    score_entries(client, cfg.maci.models, entries, template, progress=_Progress('scoring'))
    scores_dir = os.path.abspath(cfg.data.scores_dir or DEFAULT_SCORES_DIR)
    written = write_score_files(entries, cfg.maci.models, scores_dir, SCORE_PREFIX[dataset])
    print(client.usage_summary())
    for path in written:
        print(f"Wrote {path}")
    print("\nRun the benchmark with these models:\n  maci run --dataset {} --scores-dir {} --limit-samples {} "
          "--models {}".format(dataset, scores_dir, limit, ' '.join(cfg.maci.models)))
    return 0


def cmd_calibrate(args: argparse.Namespace) -> int:
    from .llm import SCORING_PROMPT
    from .pipeline import (calibrate, calibrator_metadata, load_benchmark_samples, planned_requests, read_records,
                           score_entries)

    _quiet_library_logs()
    cfg = _resolve_config(args)
    client = _client(cfg)
    index = _model_index(client)
    _validate_models(cfg.maci.models, index)
    if args.input:
        entries = read_records(args.input, need_labels=True)
        source = os.path.abspath(args.input)
    else:
        dataset = 'wikibio' if cfg.data.dataset == 'auto' else cfg.data.dataset
        entries = [{'sample': s} for s in load_benchmark_samples(dataset, cfg.data.data_dir, limit=args.limit)]
        source = f"{dataset} (first {args.limit} answers)"
    template = cfg.llm.scoring_prompt or SCORING_PROMPT
    print(f"Calibrating on {len(entries)} labelled answers from {source}.")
    _confirm(planned_requests(client, cfg.maci.models, entries, template), client, index, args.yes, 'scoring')
    score_entries(client, cfg.maci.models, entries, template, progress=_Progress('scoring'))
    model, report = calibrate(entries, cfg.maci.models, alpha=cfg.experiment.alpha, grouper=args.grouper,
                              holdout=args.holdout, random_seed=cfg.experiment.random_seed, n_bins=cfg.maci.n_bins,
                              target_tpr=cfg.maci.target_tpr, ensemble_train_ratio=cfg.maci.ensemble_train_ratio,
                              solver=cfg.maci.solver)
    model.save(args.output, calibrator_metadata(cfg, cfg.maci.models, source, report, template))
    print(client.usage_summary())
    print(f"\nSaved {os.path.abspath(args.output)}")
    print(f"  models:  {', '.join(cfg.maci.models)}")
    print(f"  alpha:   {cfg.experiment.alpha}  (target: all kept claims true for ≥ {1 - cfg.experiment.alpha:.0%} "
          "of answers)")
    print(f"  groups:  {args.grouper} → {', '.join(f'{k}: budget {v:.3f}' for k, v in model.get_budgets().items())}")
    if 'holdout_coverage' in report:
        print(f"  held-out check on {report['n_holdout']} answers: coverage {report['holdout_coverage']:.3f}, "
              f"retention {report['holdout_retention']:.3f}")
    print('\nCheck an answer:\n  maci check --question "..." --answer "..."')
    return 0


def cmd_check(args: argparse.Namespace) -> int:
    import json

    from .adaptive import SubgroupOptimizedMACI
    from .llm import EXTRACTION_PROMPT, SCORING_PROMPT
    from .pipeline import extraction_plan, normalize_record, planned_requests, read_records, check_answers
    from .report import LEGEND, format_row, write_jsonl

    _quiet_library_logs()
    cfg = _resolve_config(args)
    if not os.path.exists(args.calibrator):
        raise FileNotFoundError(f"Calibrator not found: {args.calibrator}. Create one with `maci calibrate`.")
    calibrator = SubgroupOptimizedMACI.load(args.calibrator)
    meta = getattr(calibrator, 'metadata', {}) or {}
    llm_meta = meta.get('llm', {})
    # Score exactly as during calibration, otherwise the guarantee does not carry over.
    template = llm_meta.get('scoring_prompt') or SCORING_PROMPT
    cfg.llm.temperature = llm_meta.get('temperature', cfg.llm.temperature)
    cfg.llm.max_tokens = llm_meta.get('max_tokens', cfg.llm.max_tokens)
    if llm_meta.get('base_url') and llm_meta['base_url'] != cfg.llm.base_url:
        print(f"Note: calibrated against {llm_meta['base_url']}, now using {cfg.llm.base_url}.", file=sys.stderr)
    client = _client(cfg)

    if args.input:
        records = read_records(args.input, need_labels=False)
    else:
        if not args.question or not (args.answer or args.claims):
            raise ValueError("Give --question with --answer (or --claims), or --input FILE.")
        rec = {'question': args.question, 'answer': args.answer or '', 'claims': args.claims or []}
        records = [normalize_record(rec, need_labels=False)]
    extractor = cfg.llm.extractor_model or calibrator.model_names[0]
    plan = (extraction_plan(client, extractor, records, cfg.llm.extraction_prompt or EXTRACTION_PROMPT)
            + planned_requests(client, calibrator.model_names, records, template))
    # Answers whose claims are extracted first also need one scoring request per model.
    plan += [{'model': m, 'prompt_chars': len(r['sample']['response']) + 800, 'completion_tokens': 100}
             for r in records if not r['sample']['atomic_facts'] for m in calibrator.model_names]
    index = _model_index(client) if plan else None
    _validate_models([extractor], index)
    if len(plan) > 50:
        _confirm(plan, client, index, args.yes, 'LLM')
    rows = check_answers(calibrator, client, records, extractor, template,
                         cfg.llm.extraction_prompt or EXTRACTION_PROMPT,
                         progress=_Progress('working') if len(records) > 1 else None)

    if args.output:
        write_jsonl(args.output, rows)
    if args.json:
        for row in rows:
            print(json.dumps(row, ensure_ascii=False))
        return 0
    alpha = meta.get('alpha', getattr(calibrator, 'alpha', None))
    print(f"Calibrator: {args.calibrator} (alpha={alpha}, models: {', '.join(calibrator.model_names)})")
    print(LEGEND + "\n")
    for i, row in enumerate(rows[:args.limit]):
        print(f"[{i + 1}] {row['prompt']}")
        print('  ' + format_row(row, show_model_scores=args.show_model_scores).replace('\n', '\n  '))
        print(f"  Filtered answer: {row['filtered_answer'] or '(nothing kept)'}\n")
    if len(rows) > args.limit:
        print(f"... {len(rows) - args.limit} more" + (f"; all rows are in {args.output}" if args.output else
                                                       "; pass --output FILE to save all of them"))
    labelled = [r for r in rows if r.get('covered') is not None]
    if labelled:
        cov = sum(r['covered'] for r in labelled) / len(labelled)
        kept = sum(r['n_kept'] for r in labelled) / max(1, sum(r['n_claims'] for r in labelled))
        print(f"On {len(labelled)} labelled answers: coverage {cov:.3f}, retention {kept:.3f}")
    print(client.usage_summary(), file=sys.stderr)
    return 0


# ---------------------------------------------------------------------------
# config / data / info
# ---------------------------------------------------------------------------

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


def _expand_datasets(names: List[str], default: Sequence[str] = ('wikibio',)) -> List[str]:
    names = list(names or default)
    unknown = [n for n in names if n != 'all' and n not in DATA_FILES]
    if unknown:
        raise ValueError(f"Unknown dataset {', '.join(unknown)}; choose from {', '.join([*DATA_FILES, 'all'])}")
    return list(DATA_FILES) if 'all' in names else names


def cmd_data_download(args: argparse.Namespace) -> int:
    _quiet_library_logs(logging.INFO)
    datasets = _expand_datasets(args.datasets)
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
    key_env = Config().llm.api_key_env
    print(f"OpenRouter key ({key_env}): {'set' if os.environ.get(key_env) else 'not set'}")
    return 0


# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog='maci',
        description='Multi-LLM Adaptive Conformal Inference (MACI) for claim-level factuality filtering.',
        epilog="Typical use: `maci run` (paper benchmark), or `maci calibrate` then `maci check` (your own answers).",
    )
    parser.add_argument('-V', '--version', action='version', version=f"%(prog)s {__version__}")
    sub = parser.add_subparsers(dest='command', metavar='COMMAND')
    sub.required = True

    p_run = sub.add_parser('run', help='run the BCI / CCI / MACI benchmark',
                           description='Run the benchmark. Settings come from the config file, '
                                       'and any flag below overrides it.')
    _add_config_flags(p_run, lambda s, k: s in _RUN_SECTIONS)
    # Accepted for compatibility with the original script; it only ever had one choice.
    p_run.add_argument('--model-set', choices=['fixed'], help=argparse.SUPPRESS)
    p_run.set_defaults(func=cmd_run)

    p_ins = sub.add_parser('inspect', help='show claim-by-claim results of a run in the terminal',
                           description='Show which claims each method kept or removed, and which were false.')
    p_ins.add_argument('path', help='run folder (maci_outputs/<time>) or predictions .jsonl file')
    p_ins.add_argument('--method', help='only this method, e.g. MACI, BCI, CCI')
    p_ins.add_argument('--run', type=int, help='which run (default: the first)')
    p_ins.add_argument('--only-missed', action='store_true', help='only answers where a false claim was kept')
    p_ins.add_argument('--search', metavar='TEXT', help='only answers whose prompt or claims contain TEXT')
    p_ins.add_argument('--limit', type=int, default=5, help='how many answers to print (default 5)')
    p_ins.add_argument('--show-model-scores', action='store_true', help='also print each LLM scorer\'s score')
    p_ins.set_defaults(func=cmd_inspect)

    p_models = sub.add_parser('models', help='search OpenRouter models to use as scorers',
                              description='List models from OpenRouter (no API key needed). '
                                          'All search words must appear in the id or name.')
    p_models.add_argument('query', nargs='*', help='search words, e.g. llama 70b')
    p_models.add_argument('--sort', choices=['name', 'price', 'context', 'newest'], default='name')
    p_models.add_argument('--free', action='store_true', help='only free models')
    p_models.add_argument('--limit', type=int, default=40)
    _add_config_flags(p_models, lambda s, k: (s, k) in {('llm', 'base_url')})
    p_models.set_defaults(func=cmd_models)

    p_score = sub.add_parser('score', help='score benchmark claims with OpenRouter models (for `maci run`)',
                             description='Ask each model for the probability that each benchmark claim is true, '
                                         'and save score files that `maci run --scores-dir` can use.')
    p_score.add_argument('--limit', type=int, help='number of benchmark answers (default: data.limit_samples)')
    p_score.add_argument('-y', '--yes', action='store_true', help='do not ask before sending API requests')
    _add_config_flags(p_score, lambda s, k: (s, k) in _LLM_KEYS | {('data', 'dataset'), ('data', 'scores_dir')})
    p_score.set_defaults(func=cmd_score)

    p_cal = sub.add_parser('calibrate', help='fit MACI for your chosen OpenRouter models',
                           description='Score labelled answers with the chosen models and fit MACI. '
                                       'Uses WikiBio unless --input gives your own labelled data.')
    p_cal.add_argument('--input', metavar='FILE',
                       help='labelled .jsonl/.json: {"question", "answer", "claims": [{"text", "is_supported"}]}')
    p_cal.add_argument('--limit', type=int, default=500, help='benchmark answers to use (default 500)')
    p_cal.add_argument('--grouper', choices=_GROUP_CHOICES, default='false_claim_risk',
                       help='how answers are grouped for calibration (default false_claim_risk)')
    p_cal.add_argument('--holdout', type=float, default=0.2,
                       help='share kept aside to report coverage and retention (default 0.2)')
    p_cal.add_argument('-o', '--output', default=DEFAULT_CALIBRATOR, help=f'(default {DEFAULT_CALIBRATOR})')
    p_cal.add_argument('-y', '--yes', action='store_true', help='do not ask before sending API requests')
    _add_config_flags(p_cal, lambda s, k: (s, k) in _LLM_KEYS | {
        ('data', 'dataset'), ('experiment', 'alpha'), ('experiment', 'random_seed'), ('maci', 'n_bins'),
        ('maci', 'target_tpr'), ('maci', 'ensemble_train_ratio'), ('maci', 'solver')})
    p_cal.set_defaults(func=cmd_calibrate)

    p_chk = sub.add_parser('check', help='filter the claims of new answers with a calibrator',
                           description='Split answers into claims, score them with the calibrated models, and keep '
                                       'only the claims MACI trusts.')
    p_chk.add_argument('--calibrator', default=DEFAULT_CALIBRATOR, help=f'(default {DEFAULT_CALIBRATOR})')
    p_chk.add_argument('-q', '--question')
    p_chk.add_argument('-a', '--answer')
    p_chk.add_argument('--claims', nargs='+', metavar='CLAIM', help='use these claims instead of extracting them')
    p_chk.add_argument('--input', metavar='FILE', help='.jsonl/.json of {"question", "answer", "claims"?}')
    p_chk.add_argument('-o', '--output', metavar='FILE', help='write every result as JSONL')
    p_chk.add_argument('--json', action='store_true', help='print JSON lines instead of text')
    p_chk.add_argument('--limit', type=int, default=20, help='answers to print (default 20)')
    p_chk.add_argument('--show-model-scores', action='store_true', help="also print each model's score")
    p_chk.add_argument('-y', '--yes', action='store_true', help='do not ask before large batches')
    _add_config_flags(p_chk, lambda s, k: s == 'llm' and k not in ('temperature',))
    p_chk.set_defaults(func=cmd_check)

    p_cfg = sub.add_parser('config', help='create or inspect a config file')
    cfg_sub = p_cfg.add_subparsers(dest='config_command', metavar='ACTION')
    cfg_sub.required = True
    p_init = cfg_sub.add_parser('init', help=f'write a commented {DEFAULT_CONFIG_NAME} template')
    p_init.add_argument('path', nargs='?', default=DEFAULT_CONFIG_NAME)
    p_init.add_argument('-f', '--force', action='store_true', help='overwrite an existing file')
    p_init.set_defaults(func=cmd_config_init)
    p_show = cfg_sub.add_parser('show', help='print the effective config (file + flags)')
    _add_config_flags(p_show, lambda s, k: True)
    p_show.set_defaults(func=cmd_config_show)

    p_data = sub.add_parser('data', help='download or check the benchmark data')
    data_sub = p_data.add_subparsers(dest='data_command', metavar='ACTION')
    data_sub.required = True
    choices = [*DATA_FILES, 'all']
    p_dl = data_sub.add_parser('download', help='download benchmark files (default: wikibio)')
    p_dl.add_argument('datasets', nargs='*', metavar='DATASET',
                      help=f"one or more of: {', '.join(choices)} (default: wikibio)")
    p_dl.add_argument('--data-dir', metavar='DIR', help=f'destination (default: {default_data_dir()})')
    p_dl.add_argument('-f', '--force', action='store_true', help='re-download files that are already present')
    p_dl.set_defaults(func=cmd_data_download)
    p_st = data_sub.add_parser('status', help='check which benchmark files are present and intact')
    p_st.add_argument('datasets', nargs='*', metavar='DATASET',
                      help=f"one or more of: {', '.join(choices)} (default: wikibio)")
    p_st.add_argument('--data-dir', metavar='DIR')
    p_st.set_defaults(func=cmd_data_status)

    p_info = sub.add_parser('info', help='show versions, data location and optional features')
    p_info.set_defaults(func=cmd_info)
    return parser


def _safe_streams() -> None:
    # Consoles that cannot show a character (e.g. emoji on a Windows code page) get '?' instead of a crash.
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(errors='replace')
        except (AttributeError, ValueError):
            pass


def main(argv: Optional[List[str]] = None, run_defaults: Optional[Dict[tuple, Any]] = None) -> int:
    """Entry point. ``run_defaults`` ({(section, key): value}) fills keys that neither the config file
    nor a flag sets; the legacy experiments/run_experiment.py uses it for its repository paths."""
    from .llm import LLMError

    _safe_streams()
    parser = build_parser()
    args = parser.parse_args(argv)
    args.run_defaults = run_defaults
    try:
        return args.func(args)
    except (ValueError, LLMError, OSError) as e:
        print(f"maci: error: {e}", file=sys.stderr)
        return 2
    except KeyboardInterrupt:
        print("Cancelled.", file=sys.stderr)
        return 130


if __name__ == '__main__':
    sys.exit(main())
