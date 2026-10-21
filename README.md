# MACI: Multi-LLM Adaptive Conformal Inference

This repository packages the code of [MLAI-Yonsei/MACI](https://github.com/MLAI-Yonsei/MACI) as an installable library and command-line tool.

MACI filters the claims in an LLM response so that, with a user-chosen probability `1 - alpha`, none of the claims it keeps is false. Each claim's factuality score is an ensemble of several LLM scorers, and the filter is calibrated separately for each group of prompts.

## Abstract

Ensuring factuality is essential for the safe use of Large Language Models (LLMs) in high-stakes domains such as medicine and law. Conformal inference provides distribution-free guarantees, but existing approaches are either overly conservative, discarding many true-claims, or rely on adaptive error rates and simple linear models that fail to capture complex group structures. To address these challenges, we reformulate conformal inference in a multiplicative filtering setting, modeling factuality as a product of claim-level scores. Our method, Multi-LLM Adaptive Conformal Inference (MACI), leverages ensembles to produce more accurate factuality-scores, which in our experiments led to higher retention, while validity is preserved through group-conditional calibration. Experiments show that MACI consistently achieves user-specified coverage with substantially higher retention and lower time cost than baselines.

## Installation

Requires Python 3.9 or newer.

```bash
pip install maci-conformal
```

To also run the CCI (conditional conformal) baseline, which needs PyTorch:

```bash
pip install "maci-conformal[cci]"
```

To get the latest unreleased code from `main` instead (needs git):

```bash
pip install "git+https://github.com/MLAI-Yonsei/maci-conformal.git"
```

To work on the code:

```bash
git clone https://github.com/MLAI-Yonsei/maci-conformal.git
cd maci-conformal
pip install -e ".[cci,dev]"
```

Check the install with `maci info`. It lists dependency versions, the data folder, and whether CCI is available.

## Two ways to use it

| You want to… | Commands |
|---|---|
| Reproduce the paper's benchmark (no API key needed) | `maci data download` → `maci run` → open `report.html` |
| Filter hallucinated claims out of your own LLM answers | `maci models` → `maci calibrate` → `maci check` |

## Reproduce the benchmark

```bash
maci data download              # WikiBio, ~36 MB, checked against pinned SHA-256 hashes
maci run                        # BCI, CCI and MACI over 10 random splits
```

Data goes to `~/.cache/maci/data` by default. Set `MACI_DATA_DIR` or pass `--data-dir` to use another folder. `maci data status` shows which files are present and intact.

Each run writes one folder, `maci_outputs/<time>/`:

| File | What it holds |
|---|---|
| `report.html` | Open in a browser: coverage and retention per method and subgroup, plus every answer with each claim marked as kept or removed and true or false. Missed hallucinations (false claims a method kept) are red, and a filter shows only those answers. |
| `predictions.jsonl` | One line per answer and method: every claim with its label, score, each LLM's score and whether it was kept. |
| `results.json` | Per-run coverage and retention, overall and per subgroup. |
| `config.toml` | The exact settings of the run. Rerun it with `maci run -c config.toml`. |
| `log.txt` | The full log. |

To browse the same results in the terminal:

```bash
maci inspect maci_outputs/<time> --method MACI --only-missed --show-model-scores
```

Common variations:

```bash
maci run --methods MACI --alpha 0.05 --n-runs 3
maci run --conditional-groups false_claim_risk view_count --target-tpr 0.9
maci run --help                 # every option
```

> **MedLFQA:** only the MedLFQA score files are published. The claim dataset (`med_scores/medlfqa_dataset.pkl`) is not, so `--dataset medlfqa` works only if you supply that file yourself.

## Check your own answers with OpenRouter models

MACI needs per-claim scores from several LLMs. The package gets them through [OpenRouter](https://openrouter.ai), so any model listed there can be a scorer.

**1. Get a key and pick models.** Searching models needs no key:

```bash
export OPENROUTER_API_KEY=sk-or-...     # https://openrouter.ai/keys
maci models llama 70b                   # id, context size, $ per million tokens
maci models --sort price qwen
```

**2. Calibrate.** Each model scores claims whose true/false labels are known, and MACI learns how far it can trust those scores:

```bash
maci calibrate --models meta-llama/llama-3.3-70b-instruct qwen/qwen-2.5-72b-instruct --alpha 0.1
```

This uses the first 500 WikiBio answers by default, downloading them if needed. Before sending anything it shows the number of requests and a cost estimate and asks you to confirm. Two models on 500 answers cost roughly $0.20 at current prices. Replies are cached in `~/.cache/maci/llm`, so rerunning costs nothing. The result is `maci_calibrator.json`, together with a coverage and retention check on held-out answers.

To calibrate on your own labelled data, which is the better choice when your questions are not biographies, pass a `.jsonl` file with one answer per line:

```json
{"question": "What does aspirin do?", "answer": "...", "claims": [{"text": "Aspirin is an NSAID.", "is_supported": true}, {"text": "Aspirin was discovered in 2001.", "is_supported": false}]}
```

```bash
maci calibrate --input my_labelled.jsonl --models <model ids...>
```

**3. Check new answers.** Claims are extracted by an LLM, scored by the calibrated models, and filtered:

```bash
maci check -q "Who was Marie Curie?" -a "Marie Curie won two Nobel Prizes. She was born in Vienna."
maci check --input answers.jsonl --output checked.jsonl      # many at once; claims optional
```

Example output (scores depend on the models):

```
[1] Who was Marie Curie?
  MACI [group=low]: kept 1/2 claims
    ✓  kept     0.912  ?      Marie Curie won two Nobel Prizes.
    -  removed  0.204  ?      Marie Curie was born in Vienna.
  Filtered answer: Marie Curie won two Nobel Prizes.
```

Add `--show-model-scores` to see what each LLM said, or `--json` for machine-readable output. If the input has labels, `check` also reports coverage on them.

**What the guarantee means.** With `alpha = 0.1`, at least 90% of answers keep only true claims, on average over answers like the calibration data. It holds when new answers resemble the calibration answers and are scored the same way. So `check` reuses the models, prompt and temperature stored in the calibrator file, and you should calibrate on data like yours. An answer whose group had no calibration data gets all of its claims removed, and the output says so.

To benchmark new models the same way as the paper's, score the benchmark claims and point `maci run` at the scores:

```bash
maci score --models openai/gpt-4o-mini meta-llama/llama-3.3-70b-instruct --limit 500
maci run --scores-dir maci_scores --models openai/gpt-4o-mini meta-llama/llama-3.3-70b-instruct --limit-samples 500
```

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
| | `scores_dir` | none | extra score files, e.g. from `maci score` |
| | `limit_samples` | `2000` | maximum number of answers to load |
| `[experiment]` | `methods` | `["BCI", "CCI", "MACI"]` | methods to run (CCI is skipped if PyTorch is missing) |
| | `alpha` | `0.1` | target miscoverage (coverage ≥ 1 − alpha) |
| | `n_runs`, `random_seed` | `10`, `123` | repeated splits; run *i* uses seed `random_seed + i` |
| | `calib_ratio`, `test_ratio` | `0.75`, `0.25` | split sizes |
| | `conditional_groups` | all three | `false_claim_risk`, `medical_content`, `view_count` |
| `[maci]` | `models` | the paper's 3 scorers | OpenRouter ids of the ensemble members |
| | `target_tpr` | `0.8` | true-positive target for the ensemble weights |
| | `n_bins` | `3` | calibration groups per grouper |
| `[bci]` | `score_type` | `"frequency"` | `frequency`, `selfeval`, `logprob`, `ensemble` |
| `[cci]` | `boosting_epochs` | `100` | |
| | `adaptive_alpha` | `false` | per-sample adaptive alpha |
| `[llm]` | `base_url` | OpenRouter | any OpenAI-compatible endpoint, e.g. a local vLLM server |
| | `api_key_env` | `"OPENROUTER_API_KEY"` | environment variable that holds the key |
| | `extractor_model` | first scorer | model that splits answers into claims |
| | `max_workers`, `cache` | `8`, `true` | parallel requests; disk cache of replies |
| | `scoring_prompt`, `extraction_prompt` | built-in | custom prompt templates |
| `[output]` | `output_dir` | `"maci_outputs"` | each run gets a folder here |
| | `report`, `save_predictions`, `report_samples` | `true`, `true`, `200` | HTML report and per-claim file |

Typos and wrong types are rejected with a pointer to the fix, for example `Unknown key in [experiment] 'alhpa' (did you mean 'alpha'?)` or `maci.models must be a list`. Relative paths in a config file are resolved against the folder that contains it. True/false flags come in pairs, for example `--time-profile` and `--no-time-profile`.

## Using MACI from Python

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
from maci_conformal import SubgroupOptimizedMACI, FalseClaimRiskGrouper

maci = SubgroupOptimizedMACI(
    model_names=["gpt-x", "llama-y"],
    grouper=FalseClaimRiskGrouper(),   # or your own ConditionalGrouper
    random_state=0,
)
maci.fit(calibration_entries, alpha=0.1)

filtered, retention = maci.predict(new_entries)
item = filtered[0]
item["sample"]["filtered_claims"]   # the claims that were kept
item["sample"]["kept_mask"]         # True/False per claim, in the original order
item["sample"]["claim_scores"]      # ensemble score per claim
item["group"]                       # which calibration group the answer fell into

maci.save("maci_calibrator.json")   # reload with SubgroupOptimizedMACI.load(...)
```

To score claims with OpenRouter from Python, use `LLMClient`, `score_claims` and `extract_claims`, or `score_entries` and `check_answers` from `maci_conformal.pipeline`.

To group prompts differently, subclass `ConditionalGrouper` and implement `compute_values(samples) -> np.ndarray`, which returns one number per sample. MACI splits samples into `n_bins` groups by quantiles of that value and calibrates each group separately.

Other building blocks:

- `MACIAdaptiveConformal(score_function)`: the single-group MACI filter with any score function
- `BasicConformal(score_function)`: split-conformal baseline (keeps claims scoring above the threshold)
- `maci_conformal.conditional`: the CCI baseline (`[cci]` extra)
- `run_experiments(config)` and `load_dataset(...)`: the benchmark runner used by `maci run`

## Reproducing the paper environment

`requirements.txt` pins the exact versions used for the paper (Python 3.9). The original entry point still works from a clone. It forwards to `maci run` and uses the repository's `data/`, `logs/` and `analysis/experiment_results/` unless a config file or flag says otherwise. Results stay identical to the original scripts.

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
3. Create a GitHub Release with tag `v<version>`, for example `v0.2.1`. The `Publish to PyPI` workflow builds the package, checks that the tag matches the version, and uploads it.

## CCI Attribution

Our implementation of the Conditional Conformal Inference (CCI) baseline is a direct adoption of the work from the [conformal-safety](https://github.com/jjcherian/conformal-safety.git) repository. The installable package uses the published [`conditionalconformal`](https://pypi.org/project/conditionalconformal/) library. For full reproducibility, a local copy of the original modules is kept in the `conditional-conformal/` directory. That code is not the work of the MACI project. For details, see the original repository: [conformal-safety](https://github.com/jjcherian/conformal-safety.git)
