"""Per-claim predictions: building, saving, summarising and rendering them (HTML and terminal)."""

import html
import json
from collections import OrderedDict, defaultdict
from typing import Any, Dict, Iterable, List, Optional, Sequence

import numpy as np


def claim_text(claim: Any) -> str:
    if not isinstance(claim, dict):
        return str(claim)
    return claim.get('atom') or claim.get('text') or claim.get('claim') or claim.get('fact') or ''


def _label(claim: Any) -> Optional[bool]:
    if isinstance(claim, dict) and 'is_supported' in claim and claim['is_supported'] is not None:
        return bool(claim['is_supported'])
    return None


def _float_or_none(v: Any) -> Optional[float]:
    try:
        f = float(v)
    except (TypeError, ValueError):
        return None
    return f if np.isfinite(f) else None


def _kept_flags(sample: Dict[str, Any], claims: List[Any]) -> List[bool]:
    mask = sample.get('kept_mask')
    if mask is not None and len(mask) == len(claims):
        return [bool(k) for k in mask]
    kept_ids = {id(c) for c in sample.get('filtered_claims', [])}
    return [id(c) in kept_ids for c in claims]


def prediction_row(sample: Dict[str, Any], entry_scores: Optional[Dict[str, Any]] = None,
                   model_names: Sequence[str] = (), **fields: Any) -> Dict[str, Any]:
    """One JSON-ready record for a filtered sample: every claim with its label, score and kept flag."""
    claims = list(sample.get('atomic_facts', []))
    kept = _kept_flags(sample, claims)
    scores = list(sample.get('claim_scores') or [])
    entry_scores = entry_scores or {}
    rows = []
    for j, c in enumerate(claims):
        model_scores = {}
        for m in model_names:
            vals = entry_scores.get(m)
            if vals is not None and j < len(vals):
                model_scores[m] = _float_or_none(vals[j])
        rows.append({
            'text': claim_text(c),
            'is_supported': _label(c),
            'kept': kept[j],
            'score': _float_or_none(scores[j]) if j < len(scores) else None,
            'model_scores': model_scores,
        })
    n_false = sum(1 for r in rows if r['is_supported'] is False)
    n_false_kept = sum(1 for r in rows if r['is_supported'] is False and r['kept'])
    labelled = all(r['is_supported'] is not None for r in rows)
    record = dict(fields)
    record.update({
        'prompt': sample.get('prompt', ''),
        'response': sample.get('response', ''),
        'claims': rows,
        'n_claims': len(rows),
        'n_kept': sum(1 for r in rows if r['kept']),
        'n_false': n_false if labelled else None,
        'n_false_kept': n_false_kept if labelled else None,
        'covered': (n_false_kept == 0) if labelled else None,
    })
    return record


def write_jsonl(path: str, rows: Iterable[Dict[str, Any]]) -> None:
    with open(path, 'w', encoding='utf-8') as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + '\n')


def read_jsonl(path: str) -> List[Dict[str, Any]]:
    with open(path, 'r', encoding='utf-8') as f:
        return [json.loads(line) for line in f if line.strip()]


def method_key(row: Dict[str, Any]) -> str:
    return row['method'] if not row.get('grouper') else f"{row['method']} ({row['grouper']})"


def claim_outcomes(rows: Iterable[Dict[str, Any]]) -> Dict[str, Dict[str, int]]:
    """Per method: hallucinations caught/missed and true claims kept/dropped."""
    out: Dict[str, Dict[str, int]] = OrderedDict()
    for row in rows:
        c = out.setdefault(method_key(row), {'false_removed': 0, 'false_kept': 0, 'true_kept': 0, 'true_removed': 0,
                                             'samples': 0, 'samples_with_false_kept': 0})
        c['samples'] += 1
        if row.get('n_false_kept'):
            c['samples_with_false_kept'] += 1
        for cl in row['claims']:
            if cl['is_supported'] is None:
                continue
            key = ('true_' if cl['is_supported'] else 'false_') + ('kept' if cl['kept'] else 'removed')
            c[key] += 1
    return out


# ---------------------------------------------------------------------------
# Terminal
# ---------------------------------------------------------------------------

def _marker(cl: Dict[str, Any]) -> str:
    if not cl['kept']:
        return '-  removed'
    return '!! kept   ' if cl['is_supported'] is False else '✓  kept   '


def _label_text(cl: Dict[str, Any]) -> str:
    return {True: 'true ', False: 'FALSE', None: '  ?  '}[cl['is_supported']]


def format_row(row: Dict[str, Any], show_model_scores: bool = False, width: int = 100) -> str:
    """Human-readable block for one sample (used by `maci inspect` and `maci check`)."""
    lines = []
    head = method_key(row) if row.get('method') else 'MACI'
    extra = []
    if row.get('group') is not None:
        extra.append(f"group={row['group']}")
    if row.get('uncalibrated_group'):
        extra.append("group not calibrated: all claims removed")
    summary = f"kept {row['n_kept']}/{row['n_claims']} claims"
    if row.get('n_false') is not None:
        summary += f", {row['n_false_kept']} of {row['n_false']} false claims kept"
    lines.append(f"{head}{' [' + ', '.join(extra) + ']' if extra else ''}: {summary}")
    for cl in row['claims']:
        score = '  -  ' if cl['score'] is None else f"{cl['score']:.3f}"
        text = cl['text'] if len(cl['text']) <= width else cl['text'][:width - 1] + '…'
        lines.append(f"  {_marker(cl)}  {score}  {_label_text(cl)}  {text}")
        if show_model_scores and cl['model_scores']:
            ms = ', '.join(f"{m}={'-' if v is None else f'{v:.2f}'}" for m, v in cl['model_scores'].items())
            lines.append(f"{'':27}{ms}")
    return '\n'.join(lines)


LEGEND = ("Legend: ✓ kept · - removed · !! kept a FALSE claim (missed hallucination). "
          "Score: the method's claim score, higher = more likely kept. MACI removes the lowest scores first; "
          "among equal scores, the claim that comes earlier in the answer goes first.")


# ---------------------------------------------------------------------------
# HTML
# ---------------------------------------------------------------------------

_CSS = """
:root{--bg:#fbfbfa;--fg:#1d1d1b;--muted:#6b6b66;--line:#e3e2dd;--card:#fff;--ok:#1f7a4d;--okbg:#e7f4ec;
--bad:#b3261e;--badbg:#fbe9e7;--warn:#8a5a00;--warnbg:#fdf3dc;--chip:#f0efea}
@media (prefers-color-scheme:dark){:root{--bg:#161614;--fg:#ecebe6;--muted:#a3a29c;--line:#33322e;--card:#1e1e1b;
--ok:#6fd19c;--okbg:#17301f;--bad:#ff8a80;--badbg:#3a1a17;--warn:#f0c36a;--warnbg:#33290f;--chip:#2a2a26}}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--fg);font:15px/1.5 system-ui,-apple-system,
"Segoe UI",sans-serif}main{max-width:1100px;margin:0 auto;padding:24px 16px 64px}h1{font-size:24px;margin:0 0 4px}
h2{font-size:18px;margin:32px 0 8px}.meta{color:var(--muted);font-size:13px}table{border-collapse:collapse;width:100%;
background:var(--card);border:1px solid var(--line);border-radius:8px;overflow:hidden;font-size:14px}
th,td{padding:6px 10px;border-bottom:1px solid var(--line);text-align:left;vertical-align:top}
th{font-weight:600;background:var(--chip)}td.num,th.num{text-align:right;font-variant-numeric:tabular-nums}
.wrap{overflow-x:auto}.note{color:var(--muted);font-size:13px;margin:6px 0}
details.s{background:var(--card);border:1px solid var(--line);border-radius:8px;margin:8px 0}
details.s>summary{cursor:pointer;padding:10px 12px;list-style:none}details.s>summary::-webkit-details-marker{display:none}
details.s[open]>summary{border-bottom:1px solid var(--line)}.body{padding:10px 12px}
.chip{display:inline-block;background:var(--chip);border-radius:999px;padding:1px 8px;margin:2px 4px 2px 0;font-size:12px}
.chip.bad{background:var(--badbg);color:var(--bad)}.resp{white-space:pre-wrap;color:var(--muted);font-size:13px;margin:0 0 10px}
td.kept{color:var(--ok)}td.removed{color:var(--muted)}tr.false td.claim{color:var(--bad)}
td.miss{background:var(--badbg);color:var(--bad);font-weight:600}td.lost{background:var(--warnbg);color:var(--warn)}
td.caught{background:var(--okbg);color:var(--ok)}.controls{margin:8px 0;display:flex;gap:16px;flex-wrap:wrap;font-size:14px}
"""

_JS = """
const only=document.getElementById('only-missed');const q=document.getElementById('q');
function apply(){const t=(q.value||'').toLowerCase();document.querySelectorAll('details.s').forEach(d=>{
const okMiss=!only.checked||d.dataset.missed==='1';const okQ=!t||d.textContent.toLowerCase().includes(t);
d.style.display=(okMiss&&okQ)?'':'none';});}
only.addEventListener('change',apply);q.addEventListener('input',apply);
"""


def _fmt_mean_sd(values: Sequence[float], digits: int = 3) -> str:
    vals = [v for v in values if v is not None]
    if not vals:
        return '–'
    return f"{np.mean(vals):.{digits}f} ± {np.std(vals):.{digits}f}"


def _summary_table(results: Dict[str, Any], alpha: float) -> str:
    rows = []
    for method, data in results.items():
        cov = data.get('coverage') or []
        ok = np.mean(cov) >= 1 - alpha - 1e-9 if cov else False
        rows.append(
            f"<tr><td>{html.escape(method)}</td><td class=num>{len(cov)}</td>"
            f"<td class=num>{_fmt_mean_sd(cov)} {'✅' if ok else '⚠️'}</td>"
            f"<td class=num>{_fmt_mean_sd(data.get('retention_rate') or [])}</td>"
            f"<td class=num>{_fmt_mean_sd(data.get('retained_claims') or [], 1)} / "
            f"{_fmt_mean_sd(data.get('total_claims') or [], 1).split(' ±')[0]}</td></tr>")
    return ("<div class=wrap><table><tr><th>Method</th><th class=num>Runs</th>"
            f"<th class=num>Coverage (target ≥ {1 - alpha:.2f})</th><th class=num>Retention</th>"
            "<th class=num>Claims kept / total</th></tr>" + ''.join(rows) + "</table></div>")


def _outcome_table(rows: List[Dict[str, Any]]) -> str:
    body = []
    for method, c in claim_outcomes(rows).items():
        n_false = c['false_kept'] + c['false_removed']
        n_true = c['true_kept'] + c['true_removed']
        kept = c['true_kept'] + c['false_kept']
        body.append(
            f"<tr><td>{html.escape(method)}</td>"
            f"<td class=num>{c['false_removed']} / {n_false}</td>"
            f"<td class=num>{c['false_kept']}</td>"
            f"<td class=num>{c['samples_with_false_kept']} / {c['samples']}</td>"
            f"<td class=num>{c['true_kept']} / {n_true}</td>"
            f"<td class=num>{(c['true_kept'] / kept if kept else float('nan')):.3f}</td></tr>")
    return ("<div class=wrap><table><tr><th>Method</th><th class=num>False claims removed</th>"
            "<th class=num>False claims kept</th><th class=num>Answers with a false claim kept</th>"
            "<th class=num>True claims kept</th><th class=num>Share of kept claims that are true</th></tr>"
            + ''.join(body) + "</table></div>")


def _subgroup_table(results: Dict[str, Any]) -> str:
    body = []
    for method, data in results.items():
        merged: Dict[str, Dict[str, Dict[str, List[float]]]] = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
        for run in data.get('subgroup_results') or []:
            for grouper, groups in run.items():
                for group, g in groups.items():
                    for k in ('coverage', 'retention_rate', 'size'):
                        merged[grouper][group][k].append(g[k])
        for grouper, groups in merged.items():
            for group, g in groups.items():
                body.append(f"<tr><td>{html.escape(method)}</td><td>{html.escape(grouper)}</td><td>{html.escape(group)}</td>"
                            f"<td class=num>{np.mean(g['size']):.0f}</td><td class=num>{_fmt_mean_sd(g['coverage'])}</td>"
                            f"<td class=num>{_fmt_mean_sd(g['retention_rate'])}</td></tr>")
    if not body:
        return "<p class=note>No subgroup results.</p>"
    return ("<div class=wrap><table><tr><th>Method</th><th>Grouper</th><th>Group</th><th class=num>Samples</th>"
            "<th class=num>Coverage</th><th class=num>Retention</th></tr>" + ''.join(body) + "</table></div>")


def _cell(cl: Dict[str, Any]) -> str:
    score = '' if cl['score'] is None else f" <span class=meta>{cl['score']:.2f}</span>"
    if cl['is_supported'] is False:
        cls, word = ('miss', 'kept ✗') if cl['kept'] else ('caught', 'removed')
    elif cl['is_supported'] is True:
        cls, word = ('kept', 'kept') if cl['kept'] else ('lost', 'removed')
    else:
        cls, word = ('kept', 'kept') if cl['kept'] else ('removed', 'removed')
    return f"<td class={cls}>{word}{score}</td>"


def _samples_html(rows: List[Dict[str, Any]], max_samples: int) -> str:
    by_prompt: Dict[str, Dict[str, Dict[str, Any]]] = OrderedDict()
    for row in rows:
        by_prompt.setdefault(row['prompt'], OrderedDict())[method_key(row)] = row
    parts = []
    for prompt, methods in list(by_prompt.items())[:max_samples]:
        first = next(iter(methods.values()))
        missed_any = any(r.get('n_false_kept') for r in methods.values())
        chips = ''.join(
            f"<span class='chip{' bad' if r.get('n_false_kept') else ''}'>{html.escape(name)}: "
            f"{r['n_kept']}/{r['n_claims']} kept"
            + (f", {r['n_false_kept']} false" if r.get('n_false_kept') else '') + "</span>"
            for name, r in methods.items())
        header = ''.join(f"<th>{html.escape(name)}</th>" for name in methods)
        body = []
        for j, cl in enumerate(first['claims']):
            label = {True: 'true', False: 'false', None: '?'}[cl['is_supported']]
            cells = ''.join(_cell(r['claims'][j]) if j < len(r['claims']) else '<td></td>' for r in methods.values())
            body.append(f"<tr class={label}><td class=claim>{html.escape(cl['text'])}</td><td>{label}</td>{cells}</tr>")
        parts.append(
            f"<details class=s data-missed={'1' if missed_any else '0'}><summary><strong>{html.escape(prompt)}</strong><br>"
            f"{chips}</summary><div class=body><p class=resp>{html.escape(first.get('response') or '')}</p>"
            f"<div class=wrap><table><tr><th>Claim</th><th>Label</th>{header}</tr>{''.join(body)}</table></div>"
            "</div></details>")
    return ''.join(parts)


def render_html(meta: Dict[str, Any], results: Dict[str, Any], rows: List[Dict[str, Any]],
                max_samples: int = 200) -> str:
    alpha = float(meta.get('alpha', 0.1))
    first_run = min((r['run'] for r in rows), default=None)
    run_rows = [r for r in rows if r['run'] == first_run]
    # One MACI grouper (the one used for the marginal numbers) keeps the sample view readable.
    maci_groupers = [r['grouper'] for r in run_rows if r['method'] == 'MACI']
    if maci_groupers:
        keep = maci_groupers[0]
        sample_rows = [r for r in run_rows if r['method'] != 'MACI' or r['grouper'] == keep]
    else:
        sample_rows = run_rows
    n_prompts = len({r['prompt'] for r in sample_rows})
    meta_line = ' · '.join(html.escape(f"{k}: {v}") for k, v in meta.items())
    return f"""<!doctype html>
<html lang=en><head><meta charset=utf-8><meta name=viewport content="width=device-width,initial-scale=1">
<title>MACI report</title><style>{_CSS}</style></head><body><main>
<h1>MACI benchmark report</h1><p class=meta>{meta_line}</p>
<h2>Coverage and retention (all runs)</h2>
<p class=note>Coverage: share of answers whose kept claims are all true. Retention: share of claims kept.</p>
{_summary_table(results, alpha)}
<h2>What happened to each claim (run {first_run})</h2>
{_outcome_table(run_rows) if run_rows else '<p class=note>No predictions saved.</p>'}
<h2>Subgroups (all runs)</h2>
{_subgroup_table(results)}
<h2>Answers (run {first_run}, showing {min(n_prompts, max_samples)} of {n_prompts})</h2>
<p class=note>Red cells are false claims a method kept (missed hallucinations); green are false claims it removed;
amber are true claims it removed. Numbers are the method's claim score (higher = more likely kept).
MACI removes the lowest-scoring claims first; among equal scores, the claim that comes earlier in the answer
goes first, so the same score can appear as both kept and removed.</p>
<div class=controls><label><input type=checkbox id=only-missed> Only answers where a method kept a false claim</label>
<label>Search <input id=q type=search placeholder="name, claim…"></label></div>
{_samples_html(sample_rows, max_samples)}
</main><script>{_JS}</script></body></html>
"""
