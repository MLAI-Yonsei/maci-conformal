"""End-to-end use with live LLMs: score claims, calibrate MACI, and check new answers.

    samples  --score_entries-->  entries  --calibrate-->  calibrator file
    new answer --extract claims--> --score--> --calibrator.predict--> kept / removed claims

The scores used for calibration and for checking must come from the same models
and the same scoring prompt; the calibrator file records both, and ``check``
reuses them.
"""

import hashlib
import json
import logging
import os
import pickle
from datetime import datetime
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from . import __version__
from .adaptive import SubgroupOptimizedMACI
from .data import DATASET_PATHS, default_data_dir, download_data, score_file_name
from .groupers import get_grouper
from .llm import (EXTRACTION_PROMPT, SCORING_PROMPT, LLMClient, extract_claims, extraction_messages, score_claims,
                  scoring_messages)
from .metrics import compute_marginal_coverage, evaluate_retention
from .report import claim_text, prediction_row

logger = logging.getLogger(__name__)

SCORE_PREFIX = {'wikibio': 'wikibio_scores', 'medlfqa': 'medlfqa_scores'}


def prompt_fingerprint(template: str) -> str:
    return hashlib.sha256(template.encode('utf-8')).hexdigest()[:12]


# ---------------------------------------------------------------------------
# Inputs
# ---------------------------------------------------------------------------

def load_benchmark_samples(dataset: str = 'wikibio', data_dir: Optional[str] = None,
                           limit: Optional[int] = None, auto_download: bool = True) -> List[Dict[str, Any]]:
    """Raw labelled samples (prompt / response / atomic_facts) from the benchmark, in file order."""
    if dataset not in DATASET_PATHS:
        raise ValueError(f"Unknown dataset '{dataset}'. Available: {', '.join(DATASET_PATHS)}")
    data_dir = os.path.abspath(os.path.expanduser(data_dir or default_data_dir()))
    path = os.path.join(data_dir, DATASET_PATHS[dataset])
    if not os.path.exists(path):
        if dataset != 'wikibio' or not auto_download:
            raise FileNotFoundError(f"Dataset file not found: {path}")
        logger.info(f"WikiBio data not found in {data_dir}; downloading it first.")
        download_data(['wikibio'], data_dir=data_dir)
    with open(path, 'rb') as f:
        samples = pickle.load(f)
    if limit is not None:
        samples = samples[:limit]
    return [s for s in samples if s.get('atomic_facts')]


def _first(d: Dict[str, Any], *keys: str, default: Any = None) -> Any:
    for k in keys:
        if k in d and d[k] is not None:
            return d[k]
    return default


def normalize_record(rec: Dict[str, Any], need_labels: bool) -> Dict[str, Any]:
    """Accept {question|prompt, answer|response, claims:[str | {text|atom, is_supported|label}], scores?}."""
    question = str(_first(rec, 'question', 'prompt', default=''))
    answer = str(_first(rec, 'answer', 'response', default=''))
    facts = []
    for c in _first(rec, 'claims', 'atomic_facts', default=[]) or []:
        if isinstance(c, dict):
            fact = {'atom': str(_first(c, 'text', 'atom', 'claim', default=''))}
            label = _first(c, 'is_supported', 'label', 'supported')
            if label is not None:
                fact['is_supported'] = bool(label)
        else:
            fact = {'atom': str(c)}
        facts.append(fact)
    if need_labels:
        if not facts:
            raise ValueError(f"Calibration record has no claims: {question[:80]!r}")
        if any('is_supported' not in f for f in facts):
            raise ValueError(f"Calibration claims need an is_supported label: {question[:80]!r}")
    out = {'sample': {'prompt': question, 'response': answer or ' '.join(f['atom'] for f in facts),
                      'atomic_facts': facts}}
    if isinstance(rec.get('scores'), dict):
        out['scores'] = {m: np.asarray(v, dtype=float) for m, v in rec['scores'].items()}
    for key in ('id', 'meta'):
        if key in rec:
            out[key] = rec[key]
    return out


def read_records(path: str, need_labels: bool) -> List[Dict[str, Any]]:
    """Read a .jsonl (one record per line) or .json (a list of records) file."""
    with open(path, 'r', encoding='utf-8') as f:
        if path.endswith('.json'):
            raw = json.load(f)
        else:
            raw = [json.loads(line) for line in f if line.strip()]
    if not isinstance(raw, list):
        raise ValueError(f"{path} must hold a list of records")
    return [normalize_record(r, need_labels) for r in raw]


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------

def planned_requests(client: LLMClient, models: Sequence[str], entries: Sequence[Dict[str, Any]],
                     template: str = SCORING_PROMPT) -> List[Dict[str, Any]]:
    """The scoring requests still to be sent (not already provided or cached), for cost estimates."""
    plan = []
    for entry in entries:
        claims = [claim_text(c) for c in entry['sample']['atomic_facts']]
        if not claims:
            continue
        messages = scoring_messages(entry['sample']['prompt'], claims, template)
        for m in models:
            if m in entry.get('scores', {}) or client.is_cached(m, messages):
                continue
            plan.append({'model': m, 'prompt_chars': len(messages[0]['content']),
                         'completion_tokens': 8 * len(claims) + 20})
    return plan


def score_entries(client: LLMClient, models: Sequence[str], entries: List[Dict[str, Any]],
                  template: str = SCORING_PROMPT, progress=None) -> List[Dict[str, Any]]:
    """Fill ``entry['scores'][model]`` for every model that is not already there."""
    jobs = []
    for entry in entries:
        entry.setdefault('scores', {})
        claims = [claim_text(c) for c in entry['sample']['atomic_facts']]
        for m in models:
            if m not in entry['scores']:
                jobs.append((entry, m, claims))

    def run(job):
        entry, m, claims = job
        return score_claims(client, m, entry['sample']['prompt'], claims, template)

    for (entry, m, _), scores in zip(jobs, client.map(run, jobs, progress=progress)):
        entry['scores'][m] = scores
    return entries


def write_score_files(entries: Sequence[Dict[str, Any]], models: Sequence[str], scores_dir: str,
                      prefix: str) -> List[str]:
    """Save per-model scores as ``<prefix>_<model id>.npz`` so that `maci run --scores-dir` can use them.

    Existing files are merged: prompts scored before are kept unless re-scored now.
    """
    os.makedirs(scores_dir, exist_ok=True)
    written = []
    for m in models:
        path = os.path.join(scores_dir, score_file_name(prefix, m))
        merged: Dict[str, List[float]] = {}
        if os.path.exists(path):
            old = np.load(path, allow_pickle=True)
            merged.update(zip(old['prompts'].tolist(), [list(s) for s in old['scores_list'].tolist()]))
        for e in entries:
            if m in e.get('scores', {}):
                merged[e['sample']['prompt']] = [float(v) for v in e['scores'][m]]
        prompts = np.array(list(merged.keys()), dtype=object)
        scores_list = np.empty(len(merged), dtype=object)
        for i, s in enumerate(merged.values()):
            scores_list[i] = s
        np.savez(path, prompts=prompts, scores_list=scores_list)
        written.append(path)
    return written


# ---------------------------------------------------------------------------
# Calibration
# ---------------------------------------------------------------------------

def calibrate(entries: List[Dict[str, Any]], models: Sequence[str], alpha: float = 0.1,
              grouper: str = 'false_claim_risk', holdout: float = 0.2, random_seed: int = 123,
              n_bins: int = 3, target_tpr: float = 0.8, ensemble_train_ratio: float = 0.5,
              solver: str = 'osqp') -> Tuple[SubgroupOptimizedMACI, Dict[str, Any]]:
    """Fit MACI on labelled, scored entries. A held-out share is used to report coverage and retention."""
    if len(entries) < 20:
        raise ValueError(f"Need at least 20 labelled answers to calibrate, got {len(entries)}")
    rng = np.random.default_rng(random_seed)
    idx = rng.permutation(len(entries))
    n_hold = int(round(len(entries) * holdout)) if holdout > 0 else 0
    hold = [entries[i] for i in idx[:n_hold]]
    fit_on = [entries[i] for i in idx[n_hold:]]
    model = SubgroupOptimizedMACI(model_names=list(models), grouper=get_grouper(grouper), n_bins=n_bins,
                                  random_state=random_seed, solver=solver)
    model.fit(fit_on, alpha=alpha, ensemble_train_ratio=ensemble_train_ratio, target_tpr=target_tpr)
    report: Dict[str, Any] = {'n_fit': len(fit_on), 'n_holdout': len(hold)}
    if hold:
        filtered, _ = model.predict(hold)
        report['holdout_coverage'] = float(compute_marginal_coverage(filtered))
        report['holdout_retention'] = float(evaluate_retention(filtered)['overall_retention_rate'])
    return model, report


def calibrator_metadata(cfg, models: Sequence[str], source: str, report: Dict[str, Any],
                        scoring_template: str) -> Dict[str, Any]:
    return {
        'created': datetime.now().isoformat(timespec='seconds'),
        'maci_conformal': __version__,
        'alpha': cfg.experiment.alpha,
        'models': list(models),
        'source': source,
        'evaluation': report,
        'llm': {'base_url': cfg.llm.base_url, 'temperature': cfg.llm.temperature, 'max_tokens': cfg.llm.max_tokens,
                'scoring_prompt': scoring_template, 'scoring_prompt_id': prompt_fingerprint(scoring_template)},
    }


# ---------------------------------------------------------------------------
# Checking new answers
# ---------------------------------------------------------------------------

def check_answers(calibrator: SubgroupOptimizedMACI, client: LLMClient, records: List[Dict[str, Any]],
                  extractor_model: str, scoring_template: str = SCORING_PROMPT,
                  extraction_template: str = EXTRACTION_PROMPT, progress=None) -> List[Dict[str, Any]]:
    """Extract claims where none are given, score them, and filter with the calibrator.

    Returns one prediction row per record (see :func:`maci_conformal.report.prediction_row`).
    """
    to_extract = [r for r in records if not r['sample']['atomic_facts']]
    if to_extract:
        def run(rec):
            s = rec['sample']
            return extract_claims(client, extractor_model, s['prompt'], s['response'], extraction_template)
        for rec, claims in zip(to_extract, client.map(run, to_extract, progress=progress)):
            rec['sample']['atomic_facts'] = [{'atom': c} for c in claims]
    score_entries(client, calibrator.model_names, records, scoring_template, progress=progress)

    rows: List[Dict[str, Any]] = [None] * len(records)
    with_claims = [i for i, r in enumerate(records) if r['sample']['atomic_facts']]
    if with_claims:
        filtered, _ = calibrator.predict([records[i] for i in with_claims])
        for i, item in zip(with_claims, filtered):
            rows[i] = prediction_row(item['sample'], records[i].get('scores'), calibrator.model_names,
                                     method='MACI', group=item.get('group'), budget=item.get('budget'),
                                     uncalibrated_group=bool(item.get('uncalibrated_group')),
                                     id=records[i].get('id'))
    for i, r in enumerate(records):
        if rows[i] is None:
            rows[i] = prediction_row(dict(r['sample'], kept_mask=[]), {}, (), method='MACI', group=None,
                                     id=r.get('id'))
    for row in rows:
        row['filtered_answer'] = ' '.join(c['text'] for c in row['claims'] if c['kept'])
    return rows


def extraction_plan(client: LLMClient, model: str, records: Sequence[Dict[str, Any]],
                    template: str = EXTRACTION_PROMPT) -> List[Dict[str, Any]]:
    plan = []
    for r in records:
        if r['sample']['atomic_facts']:
            continue
        messages = extraction_messages(r['sample']['prompt'], r['sample']['response'], template)
        if not client.is_cached(model, messages):
            plan.append({'model': model, 'prompt_chars': len(messages[0]['content']),
                         'completion_tokens': len(r['sample']['response']) // 3 + 50})
    return plan
