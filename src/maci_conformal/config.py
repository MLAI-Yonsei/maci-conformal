"""Experiment configuration: defaults < TOML config file < command-line flags."""

import dataclasses
import difflib
import json
import os
import sys
import typing
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from .data import DEFAULT_MODEL_NAMES, canonical_model_name

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
class LLMConfig:
    base_url: str = 'https://openrouter.ai/api/v1'
    api_key_env: str = 'OPENROUTER_API_KEY'
    extractor_model: Optional[str] = None
    temperature: float = 0.0
    max_tokens: int = 0
    timeout: float = 120.0
    max_workers: int = 8
    cache: bool = True
    cache_dir: Optional[str] = None
    scoring_prompt: Optional[str] = None
    extraction_prompt: Optional[str] = None


@dataclass
class OutputConfig:
    output_dir: str = 'maci_outputs'
    log_dir: Optional[str] = None
    results_dir: Optional[str] = None
    time_profile: bool = False
    save_predictions: bool = True
    report: bool = True
    report_samples: int = 200

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
    llm: LLMConfig = field(default_factory=LLMConfig)
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

        # Relative paths that were not resolved against a config file are relative to the working folder.
        for section, key in _PATH_KEYS:
            value = getattr(getattr(self, section), key)
            if isinstance(value, str):
                setattr(getattr(self, section), key, os.path.abspath(os.path.expanduser(value)))

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
        if not e.methods:
            errors.append("experiment.methods must list at least one method")
        elif len(set(e.methods)) != len(e.methods):
            errors.append("experiment.methods lists a method twice")
        if 'MACI' in e.methods and self.maci.mode == 'subgroup_optimized' and not e.conditional_groups:
            errors.append("MACI (subgroup_optimized) needs at least one experiment.conditional_groups entry")
        c = self.cci
        if c.boosting_epochs < 0:
            errors.append("cci.boosting_epochs must be 0 or more")
        if c.boosting_lr <= 0:
            errors.append("cci.boosting_lr must be positive")
        check_open_unit('cci.calib_split_for_boost', c.calib_split_for_boost)
        check_open_unit('cci.retention_target', c.retention_target)
        if self.llm.temperature < 0:
            errors.append("llm.temperature must be 0 or more")
        if self.llm.timeout <= 0:
            errors.append("llm.timeout must be positive")
        if self.llm.max_tokens < 0:
            errors.append("llm.max_tokens must be 0 (no limit) or more")
        self.maci.models = [canonical_model_name(m) for m in self.maci.models]
        if not self.maci.models:
            errors.append("maci.models must list at least one model")
        elif len(set(self.maci.models)) != len(self.maci.models):
            errors.append("maci.models lists the same model twice")
        if self.maci.n_bins < 1:
            errors.append("maci.n_bins must be at least 1")
        if self.data.limit_samples < 1:
            errors.append("data.limit_samples must be at least 1")
        if self.llm.max_workers < 1:
            errors.append("llm.max_workers must be at least 1")
        if self.output.report_samples < 0:
            errors.append("output.report_samples must be 0 or more")
        from .llm import check_template
        for key, needed in (('scoring_prompt', ('question', 'claims', 'n')), ('extraction_prompt', ('question', 'answer'))):
            template = getattr(self.llm, key)
            if template:
                try:
                    check_template(template, needed)
                except ValueError as exc:
                    errors.append(f"llm.{key}: {exc}")
        if errors:
            raise ValueError("Invalid configuration:\n  - " + "\n  - ".join(errors))
        return self


_SECTIONS = tuple(f.name for f in dataclasses.fields(Config))
_PATH_KEYS = {('data', 'data_dir'), ('data', 'scores_dir'), ('data', 'view_metadata_csv'),
              ('output', 'output_dir'), ('output', 'log_dir'), ('output', 'results_dir'), ('llm', 'cache_dir')}
# Optional strings where "" in a config file means "not set".
_OPTIONAL_KEYS = _PATH_KEYS | {('llm', 'extractor_model'), ('llm', 'scoring_prompt'), ('llm', 'extraction_prompt')}


def _check_type(where: str, value: Any, annotation: Any) -> Any:
    """Validate a config value against its field annotation; ints are accepted for floats."""
    origin = typing.get_origin(annotation)
    if origin is typing.Union:  # Optional[X]
        if value is None:
            return None
        annotation = [a for a in typing.get_args(annotation) if a is not type(None)][0]
        origin = typing.get_origin(annotation)
    if origin in (list, List):
        if not isinstance(value, list):
            raise ValueError(f"{where} must be a list, e.g. [\"a\", \"b\"]; got {value!r}")
        inner = typing.get_args(annotation)[0]
        return [_check_type(f"{where}[{i}]", v, inner) for i, v in enumerate(value)]
    if annotation is bool:
        if not isinstance(value, bool):
            raise ValueError(f"{where} must be true or false; got {value!r}")
        return value
    if annotation is int:
        if isinstance(value, bool) or not isinstance(value, int):
            raise ValueError(f"{where} must be a whole number; got {value!r}")
        return value
    if annotation is float:
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError(f"{where} must be a number; got {value!r}")
        return float(value)
    if annotation is str:
        if not isinstance(value, str):
            raise ValueError(f"{where} must be a string; got {value!r}")
        return value
    return value


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
            if value == "" and (section, key) in _OPTIONAL_KEYS:
                value = None
            value = _check_type(f"{section}.{key}", value, fields[key].type)
            if base_dir and isinstance(value, str) and (section, key) in _PATH_KEYS:
                value = os.path.normpath(os.path.join(base_dir, os.path.expanduser(value)))
            setattr(target, key, value)
    return cfg


def _load_toml(path: str) -> Dict[str, Any]:
    if sys.version_info >= (3, 11):
        import tomllib
    else:
        try:
            import tomli as tomllib
        except ImportError:
            raise ValueError("Reading config files on Python < 3.11 needs tomli: pip install tomli") from None
    with open(path, 'rb') as f:
        try:
            return tomllib.load(f)
        except tomllib.TOMLDecodeError as e:
            raise ValueError(f"{path} is not valid TOML: {e}") from None


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
        # JSON escapes are valid TOML; JSON leaves DEL unescaped, which TOML forbids.
        return json.dumps(value, ensure_ascii=False).replace('\x7f', '\\u007f')
    if isinstance(value, (list, tuple)):
        return '[' + ', '.join(_toml_value(v) for v in value) + ']'
    raise TypeError(f"Cannot write {type(value).__name__} to TOML")


def dump_config(cfg: Config) -> str:
    """Render ``cfg`` as TOML. Unset optional values are written as comments.

    Paths are written as absolute paths, so the output means the same thing wherever it is saved.
    """
    lines: List[str] = []
    for section, values in cfg.to_dict().items():
        lines.append(f"[{section}]")
        for key, value in values.items():
            if isinstance(value, str) and (section, key) in _PATH_KEYS:
                value = os.path.abspath(value)
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
# Kept for compatibility with the original scripts; evaluation always uses tertiles.
binning_method = "quantile"

[maci]
# LLM scorers in the ensemble, by OpenRouter id (browse them with `maci models`).
# `maci run` needs a score file per model: the paper's three ship with the data;
# score others with `maci score`.
models = ["qwen/qwen-2.5-72b-instruct", "deepseek/deepseek-chat-v3-0324", "meta-llama/llama-3.3-70b-instruct"]
# "subgroup_optimized" (learned per-group weights) or "standard" (plain mean).
mode = "subgroup_optimized"
# Calibration groups per grouper. (Subgroup results are always reported in tertiles.)
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

[llm]
# Used by `maci score`, `maci calibrate` and `maci check`.
# Any OpenAI-compatible endpoint works; the key is read from this environment variable.
base_url = "https://openrouter.ai/api/v1"
api_key_env = "OPENROUTER_API_KEY"
# Model that splits answers into claims for `maci check`. Empty = the first [maci] model.
extractor_model = ""
temperature = 0.0
# 0 = no limit.
max_tokens = 0
timeout = 120.0
# Parallel requests.
max_workers = 8
# Cache replies on disk so that re-running the same request is free.
cache = true
# Empty = ~/.cache/maci/llm
cache_dir = ""
# Custom prompt templates. Empty = built-in. Scoring needs {question}, {claims}, {n};
# extraction needs {question}, {answer}.
scoring_prompt = ""
extraction_prompt = ""

[output]
# Each run writes to <output_dir>/<timestamp>/: results.json, predictions.jsonl
# (every claim, kept or removed), report.html, config.toml and log.txt.
# Setting log_dir or results_dir switches to the original flat layout.
output_dir = "maci_outputs"
log_dir = ""
results_dir = ""
time_profile = false
save_predictions = true
report = true
# How many answers the HTML report shows.
report_samples = 200
'''
