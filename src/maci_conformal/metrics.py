"""Coverage and retention metrics for filtered claim sets."""

import logging
from typing import Dict, List

import numpy as np

logger = logging.getLogger(__name__)


def evaluate_retention(filtered_dataset: List[Dict], method_name: str = "") -> Dict:
    total_original_claims = 0
    total_retained_claims = 0

    if not filtered_dataset:
        return {'overall_retention_rate': 0.0, 'retained_claims': 0, 'total_claims': 0}

    for item in filtered_dataset:
        sample_dict = item.get('sample', item)
        if not isinstance(sample_dict, dict):
            logger.warning(f"Skipping invalid item in retention evaluation: {type(sample_dict)}")
            continue

        original_claims = sample_dict.get('atomic_facts', [])
        retained_claims = sample_dict.get('filtered_claims', [])

        total_original_claims += len(original_claims)
        total_retained_claims += len(retained_claims)

    if total_original_claims > 0:
        overall_retention_rate = total_retained_claims / total_original_claims
    else:
        overall_retention_rate = 0.0

    return {
        'overall_retention_rate': overall_retention_rate,
        'retained_claims': total_retained_claims,
        'total_claims': total_original_claims
    }


def compute_marginal_coverage(filtered_dataset: List[Dict]):
    """Fraction of samples whose retained claims are all supported."""
    indicators = []
    for item in filtered_dataset:
        sample_dict = item.get('sample', item)
        if not isinstance(sample_dict, dict):
            logger.warning(f"Skipping invalid item in coverage calculation: {type(sample_dict)}")
            continue

        retained = sample_dict.get('filtered_claims', [])

        if len(retained) == 0:
            indicators.append(1.0)
        else:
            has_false = any(not claim.get('is_supported', False) for claim in retained if isinstance(claim, dict))
            indicators.append(0.0 if has_false else 1.0)

    return np.mean(indicators) if indicators else 0.0


def compute_conditional_coverage(test_data, filtered_results, grouper, alpha=0.1, binning_method='quartiles'):
    """Compute conditional coverage for subgroups"""

    combined_data = []
    for orig_sample, filtered_sample in zip(test_data, filtered_results):
        combined_sample = dict(orig_sample['sample'])
        combined_sample['scores'] = orig_sample['scores']
        combined_sample['filtered_claims'] = filtered_sample.get('filtered_claims', [])
        combined_data.append(combined_sample)

    method_mapping = {
        'quantile': 'tertiles',
        'equal_width': 'tertiles',
        'quartiles': 'tertiles'
    }
    method = method_mapping.get(binning_method, 'tertiles')

    values = grouper.compute_values(combined_data)

    if len(values) == 0:
        logger.warning(f"    ⚠️ Warning: {grouper.__class__.__name__} returned no values")
        return {}

    if np.all(values == values[0]):
        logger.warning(f"    ⚠️ Warning: {grouper.__class__.__name__} all values identical ({values[0]:.4f})")

    bins = grouper.create_bins(values, method=method)

    groups = {}
    group_names = ['low', 'medium', 'high'] if len(bins) == 3 else [f'bin_{i}' for i in range(len(bins))]

    for i, (bin_min, bin_max) in enumerate(bins):
        if i == len(bins) - 1:
            mask = (values >= bin_min) & (values <= bin_max)
        else:
            mask = (values >= bin_min) & (values < bin_max)

        indices = np.where(mask)[0].tolist()
        bin_name = group_names[i] if i < len(group_names) else f'bin_{i}'
        groups[bin_name] = indices

    results = {}
    for group_name, indices in groups.items():
        if len(indices) == 0:
            continue

        group_indicators = []
        group_total_claims = 0
        group_retained_claims = 0

        for idx in indices:
            filtered_sample = filtered_results[idx]
            retained = filtered_sample.get('filtered_claims', [])
            original_claims = test_data[idx]['sample'].get('atomic_facts', [])

            has_false = any(not claim.get('is_supported', False) for claim in retained)
            group_indicators.append(0.0 if has_false else 1.0)

            group_total_claims += len(original_claims)
            group_retained_claims += len(retained)

        coverage = np.mean(group_indicators) if group_indicators else 0.0
        retention_rate = group_retained_claims / group_total_claims if group_total_claims > 0 else 0.0
        results[group_name] = {
            'size': len(indices),
            'coverage': coverage,
            'retention_rate': retention_rate,
            'retained_claims': group_retained_claims,
            'total_claims': group_total_claims,
            'target_coverage': 1 - alpha,
        }

    return results
