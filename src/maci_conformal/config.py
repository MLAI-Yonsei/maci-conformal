"""Experiment configuration: defaults < TOML config file < command-line flags."""

import dataclasses
import difflib
import os
import sys
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from .data import DEFAULT_MODEL_NAMES

METHODS = ('BCI', 'CCI', 'MACI')
DATASETS = ('auto', 'wikibio', 'medlfqa')
GROUPS = ('false_claim_risk', 'medical_content', 'view_count')
BINNING_METHODS = ('quantile', 'equal_width')
BCI_SCORE_TYPES = ('frequency', 'selfeval', 'logprob', 'ensemble')
MACI_MODES = ('subgroup_optimized', 'standard')
SOLVERS = ('auto', 'osqp', 'ecos', 'scs')


@dataclass
class DataConfig:
    dataset: str = 'auto'
    data_dir: Optional[str] = None
    scores_dir: Optional[str] = None
    view_metadata_csv: Optional[str] = None
    limit_samples: int = 2000


@dataclass
class ExperimentConfig:
    methods: List[str] = field(default_factory=lambda: list(METHODS))
    alpha: float = 0.1
    n_runs: int = 10
    random_seed: int = 123
    calib_ratio: float = 0.75
    test_ratio: float = 0.25
    conditional_groups: List[str] = field(default_factory=lambda: list(GROUPS))
    binning_method: str = 'quantile'


@dataclass
class MACIConfig:
    models: List[str] = field(default_factory=lambda: list(DEFAULT_MODEL_NAMES))
    mode: str = 'subgroup_optimized'
    n_bins: int = 3
    ensemble_train_ratio: float = 0.5
    target_tpr: float = 0.8
    solver: str = 'osqp'


@dataclass
class BCIConfig:
    score_type: str = 'frequency'


@dataclass
class CCIConfig:
    boosting_epochs: int = 100
    boosting_lr: float = 0.005
    calib_split_for_boost: float = 0.3
    adaptive_alpha: bool = False
    retention_target: float = 0.4


@dataclass
class OutputConfig:
    output_dir: str = 'maci_outputs'
    log_dir: Optional[str] = None
    results_dir: Optional[str] = None
    time_profile: bool = False

    @property
    def resolved_log_dir(self) -> str:
        return self.log_dir or os.path.join(self.output_dir, 'logs')

    @property
    def resolved_results_dir(self) -> str:
        return self.results_dir or os.path.join(self.output_dir, 'results')


@dataclass
class Config:
    data: DataConfig = field(default_factory=DataConfig)
    experiment: ExperimentConfig = field(default_factory=ExperimentConfig)
    maci: MACIConfig = field(default_factory=MACIConfig)
    bci: BCIConfig = field(default_factory=BCIConfig)
    cci: CCIConfig = field(default_factory=CCIConfig)
    output: OutputConfig = field(default_factory=OutputConfig)

    def to_dict(self) -> Dict[str, Dict[str, Any]]:
        return dataclasses.asdict(self)

    def validate(self) -> 'Config':
        from .groupers import canonical_grouper_name

        errors = []

        def check_choice(where, value, choices):
            if value not in choices:
                errors.append(f"{where} = {value!r}; expected one of {', '.join(choices)}")

        def check_open_unit(where, value):
            if not 0.0 < value < 1.0:
                errors.append(f"{where} = {value!r}; expected a value strictly between 0 and 1")

        e = self.experiment
        e.conditional_groups = [canonical_grouper_name(g) for g in e.conditional_groups]
        for m in e.methods:
            check_choice('experiment.methods', m, METHODS)
        for g in e.conditional_groups:
            check_choice('experiment.conditional_groups', g, GROUPS)
        check_choice('experiment.binning_method', e.binning_method, BINNING_METHODS)
        check_choice('data.dataset', self.data.dataset, DATASETS)
        check_choice('bci.score_type', self.bci.score_type, BCI_SCORE_TYPES)
        check_choice('maci.mode', self.maci.mode, MACI_MODES)
        check_choice('maci.solver', self.maci.solver, SOLVERS)
        check_open_unit('experiment.alpha', e.alpha)
        check_open_unit('maci.ensemble_train_ratio', self.maci.ensemble_train_ratio)
        check_open_unit('maci.target_tpr', self.maci.target_tpr)
        if e.calib_ratio <= 0 or e.test_ratio <= 0 or e.calib_ratio + e.test_ratio > 1.0 + 1e-9:
            errors.append("experiment.calib_ratio and test_ratio must be positive and sum to at most 1")
        if e.n_runs < 1:
            errors.append("experiment.n_runs must be at least 1")
        if not self.maci.models:
            errors.append("maci.models must list at least one model")
        if errors:
            raise ValueError("Invalid configuration:\n  - " + "\n  - ".join(errors))
        return self


_SECTIONS = tuple(f.name for f in dataclasses.fields(Config))
_PATH_KEYS = {('data', 'data_dir'), ('data', 'scores_dir'), ('data', 'view_metadata_csv'),
              ('output', 'output_dir'), ('output', 'log_dir'), ('output', 'results_dir')}


def _unknown(kind: str, name: str, known) -> str:
    hint = difflib.get_close_matches(name, list(known), n=1)
    return f"Unknown {kind} '{name}'" + (f" (did you mean '{hint[0]}'?)" if hint else "")


def config_from_dict(raw: Dict[str, Any], base_dir: Optional[str] = None, base: Optional[Config] = None) -> Config:
    """Apply a nested ``{section: {key: value}}`` mapping on top of ``base`` (defaults if omitted).

    Relative paths are resolved against ``base_dir`` (the config file's folder).
    """
    cfg = base or Config()
    for section, values in raw.items():
        if section not in _SECTIONS:
            raise ValueError(_unknown('config section', section, _SECTIONS))
        if not isinstance(values, dict):
            raise ValueError(f"Config section [{section}] must be a table")
        target = getattr(cfg, section)
        fields = {f.name: f for f in dataclasses.fields(target)}
        for key, value in values.items():
            if key not in fields:
                raise ValueError(_unknown(f'key in [{section}]', key, fields))
            if value == "" and (section, key) in _PATH_KEYS:
                value = None
            if base_dir and isinstance(value, str) and (section, key) in _PATH_KEYS:
                value = os.path.normpath(os.path.join(base_dir, os.path.expanduser(value)))
            setattr(target, key, value)
    return cfg


def _load_toml(path: str) -> Dict[str, Any]:
    if sys.version_info >= (3, 11):
        import tomllib
    else:
        import tomli as tomllib
    with open(path, 'rb') as f:
        return tomllib.load(f)


def load_config(path: Optional[str] = None) -> Config:
    """Load a TOML config file on top of the defaults (or just the defaults if ``path`` is None)."""
    if path is None:
        return Config()
    path = os.path.abspath(os.path.expanduser(path))
    if not os.path.exists(path):
        raise FileNotFoundError(f"Config file not found: {path}. Create one with `maci config init`.")
    return config_from_dict(_load_toml(path), base_dir=os.path.dirname(path))


def _toml_value(value: Any) -> str:
    if isinstance(value, bool):
        return 'true' if value else 'false'
    if isinstance(value, (int, float)):
        return repr(value)
    if isinstance(value, str):
        return '"' + value.replace('\\', '\\\\').replace('"', '\\"') + '"'
    if isinstance(value, (list, tuple)):
        return '[' + ', '.join(_toml_value(v) for v in value) + ']'
    raise TypeError(f"Cannot write {type(value).__name__} to TOML")


def dump_config(cfg: Config) -> str:
    """Render ``cfg`` as TOML. Unset optional values are written as comments."""
    lines: List[str] = []
    for section, values in cfg.to_dict().items():
        lines.append(f"[{section}]")
        for key, value in values.items():
            if value is None:
                lines.append(f"# {key} = \"\"")
            else:
                lines.append(f"{key} = {_toml_value(value)}")
        lines.append("")
    return "\n".join(lines)


CONFIG_TEMPLATE = '''\
# MACI experiment configuration.
# Use it with:  maci run --config maci.toml
# Any command-line flag overrides the value here (see `maci run --help`).
# Relative paths are resolved against the folder this file is in.

[data]
# "wikibio", "medlfqa" or "auto" (picks whichever dataset is present).
dataset = "auto"
# Where the benchmark files live. Empty means $MACI_DATA_DIR, else ~/.cache/maci/data.
# Fetch them with:  maci data download wikibio
data_dir = ""
# Optional folder with per-model score files (<dataset>_scores_<model>.npz).
scores_dir = ""
# CSV with Wikipedia view counts for the view_count grouper.
# Empty means <data_dir>/wiki_scores/wikibio_final.csv.
view_metadata_csv = ""
limit_samples = 2000

[experiment]
# Any of "BCI", "CCI", "MACI". CCI needs: pip install torch conditionalconformal
methods = ["BCI", "CCI", "MACI"]
# Target miscoverage: the method aims for coverage >= 1 - alpha.
alpha = 0.1
n_runs = 10
random_seed = 123
calib_ratio = 0.75
test_ratio = 0.25
# Any of "false_claim_risk", "medical_content", "view_count".
conditional_groups = ["false_claim_risk", "medical_content", "view_count"]
# "quantile" or "equal_width".
binning_method = "quantile"

[maci]
# LLM scorers in the ensemble; each needs a matching score file.
models = ["qwen-2.5-72b-instruct", "deepseek-chat-v3-0324", "llama-3.3-70b-instruct"]
# "subgroup_optimized" (learned per-group weights) or "standard" (plain mean).
mode = "subgroup_optimized"
n_bins = 3
ensemble_train_ratio = 0.5
target_tpr = 0.8
# "osqp", "ecos", "scs" or "auto"; the others are tried if this one fails.
solver = "osqp"

[bci]
# "frequency", "selfeval", "logprob" or "ensemble".
score_type = "frequency"

[cci]
boosting_epochs = 100
boosting_lr = 0.005
calib_split_for_boost = 0.3
adaptive_alpha = false
retention_target = 0.4

[output]
# Logs go to <output_dir>/logs and result JSON to <output_dir>/results
# unless log_dir / results_dir are set.
output_dir = "maci_outputs"
log_dir = ""
results_dir = ""
time_profile = false
'''
