# MACI: Multi-LLM Adaptive Conformal Inference

This repository packages the code of [MLAI-Yonsei/MACI](https://github.com/MLAI-Yonsei/MACI) as an installable library and command-line tool.

MACI filters the claims in an LLM response so that, with a user-chosen probability `1 - alpha`, none of the claims it keeps is false. Each claim's factuality score is an ensemble of several LLM scorers, and the filter is calibrated separately for each group of prompts.

## Abstract

Ensuring factuality is essential for the safe use of Large Language Models (LLMs) in high-stakes domains such as medicine and law. Conformal inference provides distribution-free guarantees, but existing approaches are either overly conservative, discarding many true-claims, or rely on adaptive error rates and simple linear models that fail to capture complex group structures. To address these challenges, we reformulate conformal inference in a multiplicative filtering setting, modeling factuality as a product of claim-level scores. Our method, Multi-LLM Adaptive Conformal Inference (MACI), leverages ensembles to produce more accurate factuality-scores, which in our experiments led to higher retention, while validity is preserved through group-conditional calibration. Experiments show that MACI consistently achieves user-specified coverage with substantially higher retention and lower time cost than baselines.

## Installation

Requires Python 3.9 or newer and git. The package is not on PyPI yet, so install it from GitHub:

```bash
pip install "git+https://github.com/MLAI-Yonsei/maci-conformal.git"
```

To also run the CCI (conditional conformal) baseline, which needs PyTorch:

```bash
pip install "maci-conformal[cci] @ git+https://github.com/MLAI-Yonsei/maci-conformal.git"
```

These install the latest `main`. To pin a release, add the tag at the end of the URL, for example `...maci-conformal.git@v0.1.0`.

To work on the code:

```bash
git clone https://github.com/MLAI-Yonsei/maci-conformal.git
cd maci-conformal
pip install -e ".[cci,dev]"
```

Check the install with `maci info`. It lists dependency versions, the data folder, and whether CCI is available.

## Quick start: reproduce the benchmark

```bash
maci data download wikibio      # ~36 MB, checked against pinned SHA-256 hashes
maci run                        # BCI, CCI and MACI, 10 random splits
```

Data goes to `~/.cache/maci/data` by default. Set `MACI_DATA_DIR` or pass `--data-dir` to use another folder. `maci data status` shows which files are present and intact.

Results are written to `./maci_outputs/`:

- `maci_outputs/logs/experiment_log_<time>.log`: the full log
- `maci_outputs/results/results_<dataset>_<models>_<groups>_<time>.json`: per-run coverage and retention, overall and per subgroup

Common variations:

```bash
maci run --methods MACI --alpha 0.05 --n-runs 3
maci run --conditional-groups false_claim_risk view_count --target-tpr 0.9
maci run --help                 # every option
```

> **MedLFQA:** only the MedLFQA score files are published. The claim dataset (`med_scores/medlfqa_dataset.pkl`) is not, so `--dataset medlfqa` works only if you supply that file yourself.

## Configuration

Settings come from three places. Later ones override earlier ones:

1. built-in defaults
2. a TOML config file: `--config FILE`, or `./maci.toml` if it exists
3. command-line flags

```bash
maci config init                # writes a commented maci.toml
maci config show --alpha 0.2    # prints the effective settings after merging
maci run --config maci.toml
```

The main settings:

| Section | Key | Default | Meaning |
|---|---|---|---|
| `[data]` | `dataset` | `"auto"` | `wikibio`, `medlfqa`, or `auto` |
| | `data_dir` | `$MACI_DATA_DIR` or `~/.cache/maci/data` | benchmark files |
| | `limit_samples` | `2000` | maximum number of prompts to load |
| `[experiment]` | `methods` | `["BCI", "CCI", "MACI"]` | methods to run |
| | `alpha` | `0.1` | target miscoverage (coverage ≥ 1 − alpha) |
| | `n_runs`, `random_seed` | `10`, `123` | repeated splits; run *i* uses seed `random_seed + i` |
| | `calib_ratio`, `test_ratio` | `0.75`, `0.25` | split sizes |
| | `conditional_groups` | all three | `false_claim_risk`, `medical_content`, `view_count` |
| `[maci]` | `models` | the 3 LLM scorers | ensemble members; each needs a score file |
| | `target_tpr` | `0.8` | true-positive target for the ensemble weights |
| | `n_bins` | `3` | number of subgroups per grouper |
| `[bci]` | `score_type` | `"frequency"` | `frequency`, `selfeval`, `logprob`, `ensemble` |
| `[cci]` | `boosting_epochs` | `100` | |
| | `adaptive_alpha` | `false` | per-sample adaptive alpha |
| `[output]` | `output_dir` | `"maci_outputs"` | logs and results go under here |

Unknown keys are rejected with a suggestion (`Unknown key in [experiment] 'alhpa' (did you mean 'alpha'?)`). Relative paths in a config file are resolved against the folder that contains the file.

## Using MACI on your own data

Each input item is a dict holding the sample and one per-claim score array per scorer. Scores are in `[0, 1]`, higher meaning more likely true:

```python
entry = {
    "sample": {
        "prompt": "Tell me about aspirin.",
        "response": "...",
        "atomic_facts": [
            {"atom": "Aspirin is an NSAID.", "is_supported": True},     # labels only needed for calibration
            {"atom": "Aspirin was discovered in 2001.", "is_supported": False},
        ],
    },
    "scores": {
        "gpt-x": [0.93, 0.12],      # one score per atomic fact
        "llama-y": [0.88, 0.31],
    },
}
```

Calibrate on labelled data, then filter new responses, which need no labels:

```python
from maci_conformal import SubgroupOptimizedMACI, FalseClaimRiskGrouper, compute_marginal_coverage

maci = SubgroupOptimizedMACI(
    model_names=["gpt-x", "llama-y"],
    grouper=FalseClaimRiskGrouper(),   # or your own ConditionalGrouper
    random_state=0,
)
maci.fit(calibration_entries, alpha=0.1)

filtered, retention = maci.predict(new_entries)
kept_claims = filtered[0]["sample"]["filtered_claims"]

print(maci.get_weights())   # learned ensemble weights per subgroup
print(maci.get_budgets())   # calibrated thresholds per subgroup
```

To group prompts differently, subclass `ConditionalGrouper` and implement `compute_values(samples) -> np.ndarray`, which returns one number per sample. MACI splits samples into `n_bins` groups by quantiles of that value and calibrates each group separately.

Other building blocks:

- `MACIAdaptiveConformal(score_function)`: the single-group MACI filter with any score function
- `BasicConformal(score_function)`: split-conformal baseline (keeps claims scoring above the threshold)
- `maci_conformal.conditional`: the CCI baseline (`[cci]` extra)
- `run_experiments(config)` and `load_dataset(...)`: the benchmark runner used by `maci run`

## Reproducing the paper environment

`requirements.txt` pins the exact versions used for the paper (Python 3.9). The original entry point still works from a clone. It now forwards to `maci run`, with the repository's `data/`, `logs/` and `analysis/experiment_results/` as defaults:

```bash
pip install -r requirements.txt
python experiments/run_experiment.py --dataset-type wikibio --conditional-groups false_claim_risk
```

## Development

```bash
pip install -e ".[cci,dev]"
pytest
python -m build && twine check dist/*
```

### Releasing to PyPI

1. One-time setup: on PyPI, add a [trusted publisher](https://docs.pypi.org/trusted-publishers/adding-a-publisher/) for project `maci-conformal`, repository `MLAI-Yonsei/maci-conformal`, workflow `publish.yml`, environment `pypi`. GitHub creates the `pypi` environment on the first run. You can add required reviewers to it in the repository settings.
2. Bump `__version__` in `src/maci_conformal/__init__.py`.
3. Create a GitHub Release with tag `v<version>`, for example `v0.1.0`. The `Publish to PyPI` workflow builds the package, checks that the tag matches the version, and uploads it.

## CCI Attribution

Our implementation of the Conditional Conformal Inference (CCI) baseline is a direct adoption of the work from the [conformal-safety](https://github.com/jjcherian/conformal-safety.git) repository. The installable package uses the published [`conditionalconformal`](https://pypi.org/project/conditionalconformal/) library. For full reproducibility, a local copy of the original modules is kept in the `conditional-conformal/` directory. That code is not the work of the MACI project. For details, see the original repository: [conformal-safety](https://github.com/jjcherian/conformal-safety.git)
