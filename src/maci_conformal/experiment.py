"""Benchmark runner comparing BCI, CCI and MACI over repeated random splits."""

import json
import logging
import os
import time
from collections import defaultdict
from contextlib import contextmanager
from datetime import datetime
from typing import Any, Dict, List, Optional

import numpy as np

from .adaptive import MACIAdaptiveConformal, SubgroupOptimizedMACI
from .basic import BasicConformal
from .config import Config
from .data import DEFAULT_MODEL_NAMES, default_data_dir, load_dataset
from .groupers import get_grouper, set_view_metadata_csv
from .metrics import compute_conditional_coverage, compute_marginal_coverage, evaluate_retention

logger = logging.getLogger(__name__)

_PACKAGE_LOGGER = logging.getLogger('maci_conformal')


@contextmanager
def log_to(log_dir: str, console: bool = True):
    """Send package logs to a timestamped file in ``log_dir`` (and the console) for the duration."""
    os.makedirs(log_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    log_filename = os.path.join(log_dir, f"experiment_log_{timestamp}.log")

    handlers: List[logging.Handler] = [logging.FileHandler(log_filename)]
    if console:
        handlers.append(logging.StreamHandler())
    for handler in handlers:
        handler.setFormatter(logging.Formatter('%(message)s'))
        _PACKAGE_LOGGER.addHandler(handler)
    previous_level = _PACKAGE_LOGGER.level
    _PACKAGE_LOGGER.setLevel(logging.INFO)
    logger.info(f"📝 Logging to {log_filename}")
    try:
        yield log_filename
    finally:
        _PACKAGE_LOGGER.setLevel(previous_level)
        for handler in handlers:
            _PACKAGE_LOGGER.removeHandler(handler)
            handler.close()


def create_splits(data, calib_ratio=0.7, test_ratio=0.3, random_seed=42):
    """Create calibration and test splits based on ratios with random shuffling"""
    total_size = len(data)
    calib_size = int(total_size * calib_ratio)
    test_size = int(total_size * test_ratio)

    if calib_size + test_size > total_size:
        test_size = total_size - calib_size

    logger.info(f"📊 Creating splits: {calib_size} calib ({calib_ratio*100:.0f}%), {test_size} test ({test_ratio*100:.0f}%)")

    np.random.seed(random_seed)
    indices = np.random.permutation(total_size)

    calib_idx = indices[:calib_size]
    test_idx = indices[calib_size:calib_size + test_size]

    calib_data = [data[i] for i in calib_idx]
    test_data = [data[i] for i in test_idx]

    logger.info(f"🎲 Random split with seed {random_seed}: calib indices {calib_idx[:5]}..., test indices {test_idx[:5]}...")

    return calib_data, test_data, calib_idx, test_idx


def run_bci_experiment(calib_data, test_data, score_type='frequency', alpha=0.1, **kwargs):
    """Run BCI (split conformal) on one calibration/test split."""
    logger.info(f"📈 Running BCI (Split Conformal) with {score_type} scores...")

    calib_samples = [item['sample'] for item in calib_data]
    test_samples = [item['sample'] for item in test_data]

    def score_function(samples):
        result = []
        sample_to_data = {item['sample']['prompt']: item for item in calib_data + test_data}

        for sample in samples:
            prompt = sample['prompt']
            if prompt in sample_to_data:
                scores = sample_to_data[prompt]['scores'].get(score_type)
                if scores is not None:
                    result.append(1.0 - scores)
                else:
                    n_claims = len(sample.get('atomic_facts', []))
                    result.append(np.full(n_claims, 0.5))
            else:
                n_claims = len(sample.get('atomic_facts', []))
                result.append(np.full(n_claims, 0.5))
        return result

    basic_conformal = BasicConformal(score_function=score_function, random_state=0)
    basic_conformal.fit_on_calib(calib_samples, alpha=alpha)
    filtered_results, _ = basic_conformal.predict(test_samples)

    coverage = compute_marginal_coverage(filtered_results)
    retention = evaluate_retention(filtered_results, "BCI")

    return {
        'coverage': coverage,
        'retention_rate': retention['overall_retention_rate'],
        'retained_claims': retention['retained_claims'],
        'total_claims': retention['total_claims'],
        'filtered_results': filtered_results
    }


def run_maci_experiment(calib_data: List[Dict], test_data: List[Dict],
                        model_names: List[str],
                        alpha: float,
                        mode: str,
                        subgroup_name: str,
                        n_bins: int = 3,
                        ensemble_train_ratio: float = 0.5,
                        solver: str = 'osqp',
                        **kwargs) -> Dict:
    """Run MACI on one calibration/test split, grouping by ``subgroup_name``."""
    logger.info(f"📊 Running MACI experiment with mode: {mode} for subgroup: '{subgroup_name}'...")

    timing: Dict[str, float] = {}

    if mode == 'subgroup_optimized':
        grouper = get_grouper(subgroup_name)

        as_model = SubgroupOptimizedMACI(
            model_names=model_names,
            grouper=grouper,
            n_bins=n_bins,
            random_state=kwargs.get('random_state', 0),
            solver=solver,
        )
        as_model.fit(calib_data, alpha=alpha, ensemble_train_ratio=ensemble_train_ratio,
                     target_tpr=kwargs.get('target_tpr', 0.95))
        timing_details = as_model.get_timing()
        timing['maci_weight_optimization_s'] = timing_details.get('weight_optimization_s', 0.0)
        timing['maci_calibration_s'] = timing_details.get('calibration_s', 0.0)

        t1 = time.perf_counter()
        filtered_results, _ = as_model.predict(test_data)
        timing['maci_inference_s'] = time.perf_counter() - t1
        budgets = as_model.get_budgets()
        weights = as_model.get_weights()

    else:
        def score_function(data_list: List[Dict]) -> List[np.ndarray]:
            scores_list = []
            for item in data_list:
                valid_scores = [item['scores'][m] for m in model_names if m in item['scores']]
                if valid_scores:
                    scores_list.append(np.mean(valid_scores, axis=0))
                else:
                    scores_list.append(np.array([0.5] * len(item.get('sample', {}).get('atomic_facts', []))))
            return scores_list

        as_model = MACIAdaptiveConformal(score_function=score_function, random_state=kwargs.get('random_state', 0))
        t0 = time.perf_counter()
        as_model.fit_on_calib(calib_data, alpha=alpha)
        timing['maci_calibration_s'] = time.perf_counter() - t0
        t1 = time.perf_counter()
        predicted_samples, _ = as_model.predict(test_data)
        timing['maci_inference_s'] = time.perf_counter() - t1
        # Same {'sample': ...} shape as the subgroup-optimized branch.
        filtered_results = [dict(item, sample=s) for item, s in zip(test_data, predicted_samples)]
        budgets = {'overall': as_model.tau_hat}
        weights = None
    coverage = compute_marginal_coverage(filtered_results)
    retention = evaluate_retention(filtered_results, "MACI")
    return {
        'coverage': coverage,
        'retention_rate': retention['overall_retention_rate'],
        'retained_claims': retention['retained_claims'],
        'total_claims': retention['total_claims'],
        'budgets': budgets,
        'weights': weights,
        'filtered_results': filtered_results,
        'timing': timing
    }


def run_cci_experiment(
    calib_data,
    test_data,
    alpha=0.1,
    boosting_epochs=1000,
    boosting_lr=0.005,
    calib_split_for_boost=0.3,
    random_seed=0,
    adaptive_alpha: bool = False,
    retention_target: float = 0.7
):
    """
    Two-stage CCI:
      - Stage 1 (Boosting): learn beta on a subset of calib_data
      - Stage 2 (CondConf): calibrate CondConf on the remaining calib_data using learned beta
      - Predict on test_data
    """
    logger.info("🎯 Running CCI (Boosting -> CondConf) with internal calib split...")

    try:
        from .conditional import ConditionalConformalBoosting, ConditionalConformalInference
    except ImportError as e:
        logger.error(f"CCI skipped: {e}")
        return {
            "coverage": None,
            "retention_rate": None,
            "retained_claims": 0,
            "total_claims": 0,
            "filtered_results": [],
            "timing": {"cci_skipped": True, "error": str(e)}
        }

    rng = np.random.default_rng(random_seed)
    idx = np.arange(len(calib_data))
    rng.shuffle(idx)
    k = int(len(idx) * calib_split_for_boost)
    idx_boost, idx_conf = idx[:k], idx[k:]
    if len(idx_conf) == 0:
        idx_boost, idx_conf = idx[:-1], idx[-1:]
    calib_boost = [calib_data[i] for i in idx_boost]
    calib_conf = [calib_data[i] for i in idx_conf]
    logger.info(f"  🔧 calib split -> boost:{len(calib_boost)} | conf:{len(calib_conf)} (seed={random_seed})")

    booster = ConditionalConformalBoosting(random_state=random_seed)
    t_boost_0 = time.perf_counter()
    beta = booster.fit(
        calib_boost,
        boosting_epochs=boosting_epochs,
        boosting_lr=boosting_lr
    )
    t_boost_1 = time.perf_counter()

    infer = ConditionalConformalInference(random_state=random_seed)
    t_fit_0 = time.perf_counter()
    infer.fit(calib_conf, alpha=alpha, beta=beta, adaptive_alpha=adaptive_alpha, retention_target=retention_target)
    t_fit_1 = time.perf_counter()
    infer.evaluate_auroc(test_data)
    t_pred_0 = time.perf_counter()
    filtered_results = infer.predict(test_data)
    t_pred_1 = time.perf_counter()

    coverage = compute_marginal_coverage(filtered_results)
    retention = evaluate_retention(filtered_results, "CCI")

    return {
        "coverage": coverage,
        "retention_rate": retention["overall_retention_rate"],
        "retained_claims": retention["retained_claims"],
        "total_claims": retention["total_claims"],
        "filtered_results": filtered_results,
        "beta": beta,
        "calib_sizes": {"boost": len(calib_boost), "conf": len(calib_conf)},
        "split_seed": random_seed,
        "timing": {
            "cci_boost_fit_s": t_boost_1 - t_boost_0,
            "cci_condconf_fit_s": t_fit_1 - t_fit_0,
            "cci_inference_s": t_pred_1 - t_pred_0,
            "cci_adaptive_alpha_enabled": bool(adaptive_alpha)
        }
    }


def _model_set_label(model_names: List[str]) -> str:
    return 'fixed' if list(model_names) == DEFAULT_MODEL_NAMES else 'custom'


def _serializable_results(results: Dict) -> Dict[str, Dict[str, Any]]:
    def convert_to_native_types(obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, defaultdict):
            return dict(obj)
        try:
            json.dumps(obj)
            return obj
        except TypeError:
            return str(obj)

    keys_to_exclude = {'filtered_results', 'beta', 'weights', 'budgets', 'calib_sizes', 'split_seed'}

    serializable_data = {}
    for method_name, method_data in results.items():
        serializable_data[method_name] = {}
        for key, value in method_data.items():
            if key in keys_to_exclude:
                continue
            try:
                serializable_data[method_name][key] = json.loads(json.dumps(value, default=convert_to_native_types))
            except Exception as e:
                logger.warning(f"Could not serialize key '{key}' for method '{method_name}'. Skipping. Error: {e}")
    return serializable_data


def save_aggregated_results_to_json(results: Dict, cfg: Config, dataset_type: str) -> Optional[str]:
    output_dir = cfg.output.resolved_results_dir
    os.makedirs(output_dir, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    groups_str = "_".join(sorted(cfg.experiment.conditional_groups))
    filename = f"results_{dataset_type}_{_model_set_label(cfg.maci.models)}_{groups_str}_{timestamp}.json"
    filepath = os.path.join(output_dir, filename)

    logger.info(f"\n💾 Saving aggregated results to {filepath}...")
    try:
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(_serializable_results(results), f, indent=4, ensure_ascii=False)
        logger.info("✅ Successfully saved results.")
        return filepath
    except Exception as e:
        logger.error(f"❌ Failed to save results to JSON: {e}")
        return None


def _log_aggregated(all_runs_results: Dict, alpha: float) -> None:
    logger.info("\n" + "=" * 100)
    logger.info("📊 AGGREGATED RESULTS (All Runs)")
    logger.info("=" * 100)
    for method_name in sorted(all_runs_results.keys()):
        method_results = all_runs_results[method_name]

        if not method_results['coverage']:
            logger.info(f"\n{method_name}: ❌ NO SUCCESSFUL RUNS")
            continue

        n_runs = len(method_results['coverage'])
        coverage_mean = np.mean(method_results['coverage'])
        coverage_std = np.std(method_results['coverage'])
        retention_mean = np.mean(method_results['retention_rate'])
        retention_std = np.std(method_results['retention_rate'])
        retained_claims_mean = np.mean(method_results['retained_claims'])
        retained_claims_std = np.std(method_results['retained_claims'])
        total_claims_mean = np.mean(method_results['total_claims'])

        logger.info(f"\n{'='*20} {method_name} ({n_runs} runs) {'='*20}")
        logger.info("📈 MARGINAL RESULTS:")
        logger.info(f"  Coverage: {coverage_mean:.4f} ± {coverage_std:.4f}")
        logger.info(f"  Retention Rate: {retention_mean:.3f} ± {retention_std:.3f}")
        logger.info(f"  Claims: {retained_claims_mean:.1f} ± {retained_claims_std:.1f}/{total_claims_mean:.1f}")

        if method_results['subgroup_results']:
            logger.info("\n📊 SUBGROUP RESULTS:")

            subgroup_data = {}
            for run_results in method_results['subgroup_results']:
                for grouper_name, grouper_results in run_results.items():
                    if grouper_name not in subgroup_data:
                        subgroup_data[grouper_name] = {}

                    for group_name, group_result in grouper_results.items():
                        if group_name not in subgroup_data[grouper_name]:
                            subgroup_data[grouper_name][group_name] = {
                                'coverage': [], 'retention_rate': [], 'retained_claims': [],
                                'total_claims': [], 'size': []
                            }

                        bucket = subgroup_data[grouper_name][group_name]
                        bucket['coverage'].append(group_result['coverage'])
                        bucket['retention_rate'].append(group_result['retention_rate'])
                        bucket['retained_claims'].append(group_result['retained_claims'])
                        bucket['total_claims'].append(group_result['total_claims'])
                        bucket['size'].append(group_result['size'])

            for grouper_name, groups in subgroup_data.items():
                logger.info(f"\n  🔍 {grouper_name}:")

                for group_name, group_data in groups.items():
                    if not group_data['coverage']:
                        continue

                    group_coverage_mean = np.mean(group_data['coverage'])
                    group_coverage_std = np.std(group_data['coverage'])
                    group_retention_mean = np.mean(group_data['retention_rate'])
                    group_retention_std = np.std(group_data['retention_rate'])
                    group_retained_claims_mean = np.mean(group_data['retained_claims'])
                    group_retained_claims_std = np.std(group_data['retained_claims'])
                    group_total_claims_mean = np.mean(group_data['total_claims'])
                    group_size_mean = np.mean(group_data['size'])

                    target_coverage = 1 - alpha
                    violation_marker = "⚠️ " if abs(group_coverage_mean - target_coverage) > 0.014 else "✅ "

                    logger.info(f"    {violation_marker}{group_name}:")
                    logger.info(f"      Coverage: {group_coverage_mean:.3f} ± {group_coverage_std:.3f} (target: {target_coverage:.1f})")
                    logger.info(f"      Retention: {group_retention_mean:.3f} ± {group_retention_std:.3f}")
                    logger.info(f"      Claims: {group_retained_claims_mean:.1f} ± {group_retained_claims_std:.1f}/{group_total_claims_mean:.1f}")
                    logger.info(f"      Group size: {group_size_mean:.1f} samples")
                    logger.info(f"      Coverage gap: {group_coverage_mean - target_coverage:+.3f}")

    logger.info("\n" + "=" * 100)


def run_experiments(cfg: Config) -> Dict[str, Any]:
    """Run every configured method for ``cfg.experiment.n_runs`` random splits.

    Returns ``{'results': <JSON-ready aggregate>, 'results_path': <saved file or None>,
    'dataset_type': <str>}``. Progress is logged to the ``maci_conformal`` logger;
    wrap the call in :func:`log_to` to also write a log file.
    """
    cfg.validate()
    e = cfg.experiment
    data_dir = cfg.data.data_dir or default_data_dir()
    set_view_metadata_csv(cfg.data.view_metadata_csv or os.path.join(data_dir, 'wiki_scores', 'wikibio_final.csv'))

    logger.info("=" * 80)
    logger.info(f"📊 Setup: {e.calib_ratio*100:.0f}% calibration + {e.test_ratio*100:.0f}% test, α={e.alpha}, adaptive={cfg.cci.adaptive_alpha}")
    logger.info(f"🔄 Number of runs: {e.n_runs}")
    logger.info(f"🧪 Methods: {', '.join(e.methods)}")
    logger.info(f"🏷️  BCI Score: {cfg.bci.score_type}")
    logger.info(f"📊 Conditional groups: {e.conditional_groups}")
    logger.info(f"🔧 Binning Method: {e.binning_method}")
    logger.info(f"📁 Data directory: {data_dir}")

    data, dataset_type = load_dataset(
        data_dir, scores_dir=cfg.data.scores_dir, dataset_type=cfg.data.dataset,
        limit_samples=cfg.data.limit_samples, model_names=cfg.maci.models,
    )
    logger.info(f"➡️ Using dataset type: {dataset_type}")

    all_runs_results = defaultdict(lambda: defaultdict(list))
    groupers = [get_grouper(name) for name in e.conditional_groups]
    model_names_to_use = list(cfg.maci.models)
    logger.info(f" MACI Models: {', '.join(model_names_to_use)}")

    for run_idx in range(e.n_runs):
        logger.info(f"\n🔄 Run {run_idx + 1}/{e.n_runs}")
        logger.info("-" * 50)

        random_seed = e.random_seed + run_idx
        calib_data, test_data, calib_idx, test_idx = create_splits(
            data, e.calib_ratio, e.test_ratio, random_seed=random_seed
        )

        logger.info(f"📊 Run {run_idx + 1} sizes: {len(calib_data)} calib, {len(test_data)} test (seed: {random_seed})")

        results = {}

        if 'BCI' in e.methods:
            try:
                results['BCI'] = run_bci_experiment(calib_data, test_data, score_type=cfg.bci.score_type, alpha=e.alpha)
            except Exception:
                logger.exception("❌ BCI failed")
                results['BCI'] = None

        if 'CCI' in e.methods:
            try:
                results['CCI'] = run_cci_experiment(
                    calib_data, test_data,
                    alpha=e.alpha,
                    boosting_epochs=cfg.cci.boosting_epochs,
                    boosting_lr=cfg.cci.boosting_lr,
                    calib_split_for_boost=cfg.cci.calib_split_for_boost,
                    adaptive_alpha=cfg.cci.adaptive_alpha,
                    retention_target=cfg.cci.retention_target
                )
            except Exception:
                logger.exception("❌ CCI failed")
                results['CCI'] = None

        if 'MACI' in e.methods:
            results['MACI'] = {
                'coverage': [],
                'retention_rate': [],
                'retained_claims': [],
                'total_claims': [],
                'subgroup_results': {}
            }

            logger.info("--- Starting MACI Experiments ---")
            maci_marginal_results_set = False

            for subgroup_name in e.conditional_groups:
                try:
                    maci_subgroup_result = run_maci_experiment(
                        calib_data, test_data,
                        model_names=model_names_to_use,
                        alpha=e.alpha,
                        mode=cfg.maci.mode,
                        subgroup_name=subgroup_name,
                        n_bins=cfg.maci.n_bins,
                        ensemble_train_ratio=cfg.maci.ensemble_train_ratio,
                        solver=cfg.maci.solver,
                        random_state=random_seed,
                        target_tpr=cfg.maci.target_tpr
                    )

                    if maci_subgroup_result and 'filtered_results' in maci_subgroup_result:
                        flat_filtered_results = []
                        for res in maci_subgroup_result['filtered_results']:
                            flat_item = dict(res.get('sample', {}))
                            flat_item['filtered_claims'] = res.get('sample', {}).get('filtered_claims', [])
                            flat_filtered_results.append(flat_item)
                        maci_subgroup_result['filtered_results'] = flat_filtered_results

                    if not maci_marginal_results_set:
                        results['MACI']['coverage'] = maci_subgroup_result['coverage']
                        results['MACI']['retention_rate'] = maci_subgroup_result['retention_rate']
                        results['MACI']['retained_claims'] = maci_subgroup_result['retained_claims']
                        results['MACI']['total_claims'] = maci_subgroup_result['total_claims']
                        results['MACI']['filtered_results'] = maci_subgroup_result.get('filtered_results', [])
                        results['MACI']['timing'] = maci_subgroup_result.get('timing', {})
                        maci_marginal_results_set = True

                    target_grouper = get_grouper(subgroup_name)
                    try:
                        conditional_results = compute_conditional_coverage(
                            test_data,
                            maci_subgroup_result['filtered_results'],
                            target_grouper,
                            e.alpha,
                            e.binning_method
                        )
                        results['MACI']['subgroup_results'][target_grouper.__class__.__name__] = conditional_results
                    except Exception as exc:
                        logger.error(f"    ❌ MACI subgroup analysis for {target_grouper.__class__.__name__} failed: {exc}")

                except Exception:
                    logger.exception(f"❌ MACI ({subgroup_name}) failed")

            if not maci_marginal_results_set:
                results['MACI']['coverage'] = None

        for method_name, result in results.items():
            if not result or result.get('coverage') is None:
                continue

            all_runs_results[method_name]['coverage'].append(result['coverage'])
            all_runs_results[method_name]['retention_rate'].append(result['retention_rate'])
            all_runs_results[method_name]['retained_claims'].append(result['retained_claims'])
            all_runs_results[method_name]['total_claims'].append(result['total_claims'])

            run_subgroup_results = {}
            if method_name == 'MACI':
                run_subgroup_results = result.get('subgroup_results', {})
            else:
                for grouper in groupers:
                    try:
                        conditional_results = compute_conditional_coverage(
                            test_data,
                            result['filtered_results'],
                            grouper,
                            e.alpha,
                            e.binning_method
                        )
                        run_subgroup_results[grouper.__class__.__name__] = conditional_results
                    except Exception as exc:
                        logger.error(f"    ❌ {grouper.__class__.__name__} failed for {method_name}: {exc}")

            all_runs_results[method_name]['subgroup_results'].append(run_subgroup_results)

        logger.info(f"\n📊 Run {run_idx + 1} Results:")
        for method_name, result in results.items():
            if not result or result.get('coverage') is None:
                logger.info(f"  {method_name}: ❌ FAILED or SKIPPED")
                continue
            logger.info(f"  {method_name}: Coverage={result['coverage']:.4f}, Retention={result['retention_rate']:.3f}, Claims={result['retained_claims']}/{result['total_claims']}")

        if cfg.output.time_profile:
            timing_payload = {
                'dataset_type': dataset_type,
                'model_set': _model_set_label(cfg.maci.models),
                'boosting_epochs': cfg.cci.boosting_epochs,
                'adaptive_alpha': cfg.cci.adaptive_alpha,
                'retention_target': cfg.cci.retention_target,
                'run_idx': run_idx,
                'CCI': (results.get('CCI') or {}).get('timing', {}),
                'MACI': (results.get('MACI') or {}).get('timing') or {},
            }
            timing_dir = cfg.output.resolved_results_dir
            os.makedirs(timing_dir, exist_ok=True)
            tstamp = datetime.now().strftime('%Y%m%d-%H%M%S')
            timing_path = os.path.join(timing_dir, f"time_profile_{dataset_type}_{timing_payload['model_set']}_{tstamp}.json")
            with open(timing_path, 'w', encoding='utf-8') as f:
                json.dump(timing_payload, f, indent=2, ensure_ascii=False)
            logger.info(f"⏱️ Saved timing profile to {timing_path}")

    _log_aggregated(all_runs_results, e.alpha)
    results_path = save_aggregated_results_to_json(all_runs_results, cfg, dataset_type)
    return {
        'results': _serializable_results(all_runs_results),
        'results_path': results_path,
        'dataset_type': dataset_type,
    }
