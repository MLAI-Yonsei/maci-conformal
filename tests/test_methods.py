import numpy as np

from conftest import MODELS, make_entries
from maci_conformal import (BasicConformal, FalseClaimRiskGrouper, MACIAdaptiveConformal, SubgroupOptimizedMACI,
                            compute_marginal_coverage, evaluate_retention)


def strip_labels(entries):
    """Copies of the entries without is_supported, as at prediction time."""
    out = []
    for e in entries:
        sample = dict(e['sample'], atomic_facts=[{'atom': f['atom']} for f in e['sample']['atomic_facts']])
        out.append(dict(e, sample=sample))
    return out


def test_subgroup_maci_reaches_target_coverage(rng):
    calib, test = make_entries(1200, rng), make_entries(1000, rng, offset=10_000)
    maci = SubgroupOptimizedMACI(model_names=MODELS, grouper=FalseClaimRiskGrouper(), random_state=0, solver='osqp')
    maci.fit(calib, alpha=0.1, target_tpr=0.8)

    filtered, rates = maci.predict(test)
    coverage = compute_marginal_coverage(filtered)
    retention = evaluate_retention(filtered)['overall_retention_rate']

    assert coverage >= 0.9 - 0.04
    assert 0.0 < retention < 1.0
    assert len(rates) == len(test)
    weights = maci.get_weights()['fallback_weights']
    assert np.isclose(np.sum(weights), 1.0) and np.all(weights >= 0)
    assert set(maci.get_budgets()) <= {'low', 'medium', 'high'}


def test_maci_predicts_without_labels(rng):
    calib, test = make_entries(600, rng), make_entries(50, rng, offset=10_000)
    maci = SubgroupOptimizedMACI(model_names=MODELS, grouper=FalseClaimRiskGrouper(), random_state=0)
    maci.fit(calib, alpha=0.1)
    filtered, _ = maci.predict(strip_labels(test))
    for item, original in zip(filtered, test):
        kept = item['sample']['filtered_claims']
        assert len(kept) <= len(original['sample']['atomic_facts'])
        assert all('is_supported' not in c for c in kept)


def test_adaptive_conformal_with_custom_score_function(rng):
    calib, test = make_entries(800, rng), make_entries(800, rng, offset=10_000)
    model = MACIAdaptiveConformal(score_function=lambda data: [np.mean([d['scores'][m] for m in MODELS], axis=0)
                                                               for d in data], random_state=0)
    model.fit_on_calib(calib, alpha=0.2)
    filtered, _ = model.predict(test)
    assert model.tau_hat is not None
    assert compute_marginal_coverage(filtered) >= 0.8 - 0.05


def test_basic_conformal_reaches_target_coverage(rng):
    calib, test = make_entries(800, rng), make_entries(800, rng, offset=10_000)
    by_prompt = {e['sample']['prompt']: e for e in calib + test}

    # BasicConformal keeps claims whose score is above the threshold (higher = more trustworthy).
    def score_function(samples):
        return [by_prompt[s['prompt']]['scores'][MODELS[0]] for s in samples]

    bci = BasicConformal(score_function=score_function, random_state=0)
    bci.fit_on_calib([e['sample'] for e in calib], alpha=0.2)
    filtered, _ = bci.predict([e['sample'] for e in test])
    assert compute_marginal_coverage(filtered) >= 0.8 - 0.05
