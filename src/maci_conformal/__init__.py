"""Multi-LLM Adaptive Conformal Inference (MACI) for claim-level factuality filtering.

Quick start::

    from maci_conformal import SubgroupOptimizedMACI, FalseClaimRiskGrouper

    maci = SubgroupOptimizedMACI(model_names=["model_a", "model_b"], grouper=FalseClaimRiskGrouper())
    maci.fit(calib_entries, alpha=0.1)
    filtered, retention = maci.predict(new_entries)
    maci.save("maci_calibrator.json")

With OpenRouter models, :mod:`maci_conformal.pipeline` scores claims, calibrates and
checks new answers (the ``maci calibrate`` / ``maci check`` commands).

The CCI baseline lives in :mod:`maci_conformal.conditional` and needs the ``cci`` extra.
"""

__version__ = "0.2.1"

from .adaptive import MACIAdaptiveConformal, SubgroupOptimizedMACI
from .basic import BasicConformal
from .config import Config, load_config
from .data import DEFAULT_MODEL_NAMES, default_data_dir, download_data, load_dataset
from .experiment import run_experiments
from .groupers import (ConditionalGrouper, FalseClaimRiskGrouper, MedicalContentGrouper, ViewCountGrouper,
                       get_available_groupers, get_grouper)
from .llm import LLMClient, LLMError, extract_claims, score_claims
from .metrics import compute_conditional_coverage, compute_marginal_coverage, evaluate_retention
from .pipeline import calibrate, check_answers, score_entries

__all__ = [
    "__version__",
    "MACIAdaptiveConformal",
    "SubgroupOptimizedMACI",
    "BasicConformal",
    "ConditionalGrouper",
    "FalseClaimRiskGrouper",
    "MedicalContentGrouper",
    "ViewCountGrouper",
    "get_available_groupers",
    "get_grouper",
    "compute_marginal_coverage",
    "compute_conditional_coverage",
    "evaluate_retention",
    "Config",
    "load_config",
    "run_experiments",
    "load_dataset",
    "download_data",
    "default_data_dir",
    "DEFAULT_MODEL_NAMES",
    "LLMClient",
    "LLMError",
    "extract_claims",
    "score_claims",
    "score_entries",
    "calibrate",
    "check_answers",
]
