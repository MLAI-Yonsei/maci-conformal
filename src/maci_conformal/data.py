"""Locating, downloading and loading the MACI benchmark data (WikiBio, MedLFQA)."""

import glob
import hashlib
import logging
import os
import pickle
import shutil
import tempfile
import urllib.request
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

logger = logging.getLogger(__name__)

DEFAULT_MODEL_NAMES = ['qwen-2.5-72b-instruct', 'deepseek-chat-v3-0324', 'llama-3.3-70b-instruct']

# Score files are matched by filename tokens; the first token list that matches wins.
# Models not listed here are matched by their lower-cased name.
_MODEL_FILE_TOKENS = {
    'qwen-2.5-72b-instruct': [['qwen-2.5-72b', 'instruct'], ['qwen', 'instruct'], ['qwen']],
    'deepseek-chat-v3-0324': [['deepseek', 'chat', 'v3'], ['deepseek', 'chat'], ['deepseek']],
    'llama-3.3-70b-instruct': [['llama-3.3-70b', 'instruct'], ['llama-3.3', 'instruct'], ['llama']],
}

# Data files are fetched from a fixed commit of the GitHub repository, so every
# download is byte-identical to what the paper used. Update DATA_REF and the
# hashes together if the data in the repository ever changes.
DATA_REPO = "MLAI-Yonsei/maci-conformal"
DATA_REF = "1f436c82750eb3afafcc9287ab6c21174e029aba"
DATA_URL_TEMPLATE = "https://raw.githubusercontent.com/{repo}/{ref}/data/{path}"

DATA_FILES: Dict[str, Dict[str, str]] = {
    'wikibio': {
        'wiki_scores/wikibio_final.csv': '1fe51a7527782df70884f01c6761a4f4315e1caea017607f2e2c9b0e59129dd7',
        'wiki_scores/wikibio_final_dataset.pkl': '482d015cec319b80bf92c4adb8e6d65c20cc40808801561291c7d5bcf76ed551',
        'wiki_scores/wikibio_final_frequencies.npz': 'f657a70d4e4307b91bdd80ee4b03daf5d3362e723ebae54ceefd5c7cc2330a37',
        'wiki_scores/wikibio_final_logprobs.npz': '644865e9037b9139e277f5d4dc2da594314a4414eca3a9bad4dcad5f1c511319',
        'wiki_scores/wikibio_final_self_evals.npz': '77f61a212dabb3fd85724e78c793887ccbe90e0285c4055411888c64dd5d44d4',
        'wiki_scores/wikibio_scores_deepseek-chat-v3-0324.npz': 'b9726f79ffdd9e6c1f89b64c66960cdc3cce5c3b868e750cc99ee28e4a666c50',
        'wiki_scores/wikibio_scores_meta-llama_llama-3.3-70b-instruct.npz': '1c36a3e500d2c08be38fcf07260a0496f498849e8ea57c9cf074f5f10aca855a',
        'wiki_scores/wikibio_scores_qwen_qwen-2.5-72b-instruct.npz': 'c8047a2bcf802c082f6995fae735b634738f351cda58c37409a52376f667ac4a',
    },
    # The MedLFQA claim dataset (med_scores/medlfqa_dataset.pkl) is not published
    # in the repository yet; only the score files are.
    'medlfqa': {
        'med_scores/medlfqa_frequencies.npz': '75c946c6772d18c650b4484bd743032f861a1af4819ded8014cbd5a3b7102857',
        'med_scores/medlfqa_logprobs.npz': '032e80b7aa0c1c73343aec30aca91c128fa9a3fa076333a07a601ba3495b1bd7',
        'med_scores/medlfqa_scores_deepseek_deepseek-chat-v3-0324.npz': '9dee348e1265e4f01029224c070ab7b75d6fdbf51b8f4705828c378430c97e38',
        'med_scores/medlfqa_scores_meta-llama_llama-3.3-70b-instruct.npz': '12eebea7f09ff94aaba4944f6d3ccf3e3ad10e33cd154a6cc274218f2709f1bd',
        'med_scores/medlfqa_scores_qwen_qwen-2.5-72b-instruct.npz': '67fe74653e69c421202cc4da4308f262657349a5bdc10bf65c43f083d92499e8',
        'med_scores/medlfqa_selfevals.npz': '58bc690d686c367019bb016b68723747f5a311b3d56f35249c8ee36a61b77878',
    },
}

DATASET_PATHS = {
    'wikibio': os.path.join('wiki_scores', 'wikibio_final_dataset.pkl'),
    'medlfqa': os.path.join('med_scores', 'medlfqa_dataset.pkl'),
}


def default_data_dir() -> str:
    """``$MACI_DATA_DIR`` if set, else ``$XDG_CACHE_HOME/maci/data`` (``~/.cache/maci/data``)."""
    env = os.environ.get('MACI_DATA_DIR')
    if env:
        return os.path.abspath(os.path.expanduser(env))
    cache_home = os.environ.get('XDG_CACHE_HOME') or os.path.join(os.path.expanduser('~'), '.cache')
    return os.path.join(cache_home, 'maci', 'data')


def _sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def data_status(data_dir: str, datasets: Iterable[str] = ('wikibio', 'medlfqa')) -> Dict[str, Dict[str, str]]:
    """Report each expected file as 'ok', 'missing' or 'checksum mismatch'."""
    status: Dict[str, Dict[str, str]] = {}
    for name in datasets:
        status[name] = {}
        for rel_path, digest in DATA_FILES[name].items():
            path = os.path.join(data_dir, rel_path)
            if not os.path.exists(path):
                status[name][rel_path] = 'missing'
            elif _sha256(path) != digest:
                status[name][rel_path] = 'checksum mismatch'
            else:
                status[name][rel_path] = 'ok'
    return status


def download_data(
    datasets: Sequence[str] = ('wikibio',),
    data_dir: Optional[str] = None,
    force: bool = False,
    ref: str = DATA_REF,
) -> str:
    """Download benchmark files into ``data_dir`` and verify their SHA-256 hashes.

    Files that are already present with the right hash are skipped unless ``force``.
    Returns the data directory used.
    """
    data_dir = os.path.abspath(os.path.expanduser(data_dir or default_data_dir()))
    for name in datasets:
        if name not in DATA_FILES:
            raise ValueError(f"Unknown dataset '{name}'. Available: {', '.join(DATA_FILES)}")
        for rel_path, digest in DATA_FILES[name].items():
            dest = os.path.join(data_dir, rel_path)
            if not force and os.path.exists(dest) and _sha256(dest) == digest:
                logger.info(f"  ✅ {rel_path} (already present)")
                continue
            os.makedirs(os.path.dirname(dest), exist_ok=True)
            url = DATA_URL_TEMPLATE.format(repo=DATA_REPO, ref=ref, path=rel_path)
            logger.info(f"  ⬇️  {url}")
            fd, tmp = tempfile.mkstemp(dir=os.path.dirname(dest), suffix='.part')
            try:
                with os.fdopen(fd, 'wb') as out, urllib.request.urlopen(url, timeout=60) as resp:
                    shutil.copyfileobj(resp, out)
                actual = _sha256(tmp)
                if ref == DATA_REF and actual != digest:
                    raise IOError(f"Checksum mismatch for {rel_path}: expected {digest}, got {actual}")
                os.replace(tmp, dest)
            finally:
                if os.path.exists(tmp):
                    os.remove(tmp)
    return data_dir


def _find_score_file(npz_files: List[str], model_name: str) -> Optional[str]:
    token_options = _MODEL_FILE_TOKENS.get(model_name, [[model_name.lower()]])
    for tokens in token_options:
        for fp in npz_files:
            name = os.path.basename(fp).lower()
            if all(t in name for t in tokens):
                return fp
    return None


def _resolve_dataset_type(data_dir: str, dataset_type: str) -> str:
    if dataset_type != "auto":
        return dataset_type
    for name in ('wikibio', 'medlfqa'):
        if os.path.exists(os.path.join(data_dir, DATASET_PATHS[name])):
            logger.info(f"  🔍 Auto-detected dataset type: {name}")
            return name
    raise FileNotFoundError(
        f"Could not find dataset files in {data_dir}. "
        "Run `maci data download` or pass --data-dir."
    )


def load_dataset(
    data_dir: Optional[str] = None,
    scores_dir: Optional[str] = None,
    dataset_type: str = "auto",
    limit_samples: int = 1000,
    model_names: Sequence[str] = DEFAULT_MODEL_NAMES,
) -> Tuple[List[Dict], str]:
    """Load up to ``limit_samples`` samples and attach per-claim scores.

    Returns ``(entries, dataset_type)``. Each entry is a dict with ``sample``
    (prompt / response / atomic_facts), ``annotations``, ``scores`` (one array
    per score source and per model), ``features_4d`` and ``prompt_features``.
    """
    data_dir = os.path.abspath(os.path.expanduser(data_dir or default_data_dir()))
    logger.info(f"📁 Loading up to {limit_samples} samples with provided scores...")

    dataset_type = _resolve_dataset_type(data_dir, dataset_type)

    if dataset_type == "wikibio":
        base_scores_dir = os.path.join(data_dir, "wiki_scores")
        score_prefix = "wikibio_scores"
        basic_scores = {
            'frequencies': os.path.join(base_scores_dir, "wikibio_final_frequencies.npz"),
            'logprobs': os.path.join(base_scores_dir, "wikibio_final_logprobs.npz"),
            'selfevals': os.path.join(base_scores_dir, "wikibio_final_self_evals.npz")
        }
    elif dataset_type == "medlfqa":
        base_scores_dir = os.path.join(data_dir, "med_scores")
        score_prefix = "medlfqa_scores"
        basic_scores = {
            'frequencies': os.path.join(base_scores_dir, "medlfqa_frequencies.npz"),
            'logprobs': os.path.join(base_scores_dir, "medlfqa_logprobs.npz"),
            'selfevals': os.path.join(base_scores_dir, "medlfqa_selfevals.npz")
        }
    else:
        raise ValueError(f"Unknown dataset type: {dataset_type}")
    dataset_path = os.path.join(data_dir, DATASET_PATHS[dataset_type])

    logger.info(f"  📊 Dataset: {dataset_path}")
    logger.info(f"  🎯 Score prefix: {score_prefix}")

    if not os.path.exists(dataset_path):
        hint = ("Run `maci data download wikibio` or pass --data-dir."
                if dataset_type == 'wikibio' else
                "The MedLFQA claim dataset is not distributed yet; place medlfqa_dataset.pkl there yourself.")
        raise FileNotFoundError(f"Dataset file not found: {dataset_path}. {hint}")

    with open(dataset_path, 'rb') as f:
        dataset = pickle.load(f)

    dataset_1000 = dataset[:limit_samples]

    frequencies = {}
    logprobs = {}
    selfevals = {}

    for score_type, score_path in basic_scores.items():
        try:
            if score_type == 'frequencies':
                frequencies = np.load(score_path, allow_pickle=True)
            elif score_type == 'logprobs':
                logprobs = np.load(score_path, allow_pickle=True)
            elif score_type == 'selfevals':
                selfevals = np.load(score_path, allow_pickle=True)
            logger.info(f"  ✅ Loaded {score_type}: {score_path}")
        except FileNotFoundError:
            logger.warning(f"  ⚠️ {score_type} not found: {score_path}")

    if scores_dir is not None and os.path.isdir(scores_dir):
        score_files_dir = scores_dir
    else:
        score_files_dir = base_scores_dir

    logger.info(f"  🎯 Score files directory: {score_files_dir}")

    all_npz_files = sorted(glob.glob(os.path.join(score_files_dir, f"{score_prefix}_*.npz")))
    score_files = {m: _find_score_file(all_npz_files, m) for m in model_names}

    llm_scores = {}
    for model_name, filename in score_files.items():
        try:
            model_data = np.load(filename, allow_pickle=True)
            model_prompts = model_data['prompts'].tolist()
            model_scores_list = model_data['scores_list'].tolist()
            llm_scores[model_name] = {p: s for p, s in zip(model_prompts, model_scores_list)}
            logger.info(f"  ✅ Loaded {model_name} scores")
        except (FileNotFoundError, TypeError):
            logger.warning(f"  ⚠️ {model_name} scores not found or invalid: {filename}")
            llm_scores[model_name] = {}

    aligned_data = []
    for i, sample in enumerate(dataset_1000):
        prompt = sample['prompt']
        atomic_facts = sample.get('atomic_facts', [])
        n_claims = len(atomic_facts)

        if n_claims == 0:
            continue

        if prompt in selfevals:
            selfeval_vals = selfevals[prompt]
            if hasattr(selfeval_vals, 'ndim') and selfeval_vals.ndim == 1:
                if np.allclose(selfeval_vals, -1):
                    continue
            elif np.allclose(selfeval_vals, -1):
                continue

        annotations = np.array([af.get('is_supported', False) for af in atomic_facts])

        freq_scores = np.zeros(n_claims)
        if dataset_type == 'wikibio':
            key = f'arr_{i}'
            if key in frequencies:
                freq_vals = frequencies[key]
                if hasattr(freq_vals, 'ndim') and freq_vals.ndim == 1:
                    freq_scores = freq_vals[:n_claims]
                else:
                    freq_scores = np.full(n_claims, freq_vals.item() if hasattr(freq_vals, 'item') else freq_vals)
                freq_scores = np.nan_to_num(freq_scores, nan=0.0)
        else:
            if prompt in frequencies:
                freq_vals = frequencies[prompt]
                if hasattr(freq_vals, 'ndim') and freq_vals.ndim == 1:
                    freq_scores = freq_vals[:n_claims]
                else:
                    freq_val = freq_vals.item() if hasattr(freq_vals, 'item') else freq_vals
                    freq_val = 0.0 if np.isnan(freq_val) else freq_val
                    freq_scores = np.full(n_claims, freq_val)
                freq_scores = np.nan_to_num(freq_scores, nan=0.0)

        if dataset_type == 'wikibio':
            key = f'arr_{i}'
            if key in logprobs:
                lp_vals = logprobs[key]
                if hasattr(lp_vals, 'ndim') and lp_vals.ndim == 1:
                    logprob_scores = np.nan_to_num(lp_vals[:n_claims], nan=0.0)
                else:
                    v = lp_vals.item() if hasattr(lp_vals, 'item') else lp_vals
                    v = 0.0 if np.isnan(v) else v
                    logprob_scores = np.full(n_claims, v)
            else:
                logprob_scores = np.zeros(n_claims)
        else:
            if prompt in logprobs:
                logprob_vals = logprobs[prompt]
                if hasattr(logprob_vals, 'ndim') and logprob_vals.ndim == 1:
                    logprob_scores = logprob_vals[:n_claims]
                    logprob_scores = np.nan_to_num(logprob_scores, nan=0.0)
                else:
                    logprob_val = logprob_vals.item() if hasattr(logprob_vals, 'item') else logprob_vals
                    logprob_val = 0.0 if np.isnan(logprob_val) else logprob_val
                    logprob_scores = np.full(n_claims, logprob_val)
            else:
                logprob_scores = np.zeros(n_claims)

        if dataset_type == 'wikibio':
            key = f'arr_{i}'
            if key in selfevals:
                se_vals = selfevals[key]
                if hasattr(se_vals, 'ndim') and se_vals.ndim == 1:
                    selfeval_scores = np.nan_to_num(se_vals[:n_claims], nan=0.0)
                else:
                    v = se_vals.item() if hasattr(se_vals, 'item') else se_vals
                    v = 0.0 if np.isnan(v) else v
                    selfeval_scores = np.full(n_claims, v)
            else:
                selfeval_scores = np.zeros(n_claims)
        else:
            if prompt in selfevals:
                selfeval_vals = selfevals[prompt]
                if hasattr(selfeval_vals, 'ndim') and selfeval_vals.ndim == 1:
                    selfeval_scores = selfeval_vals[:n_claims]
                    selfeval_scores = np.nan_to_num(selfeval_scores, nan=0.0)
                else:
                    selfeval_val = selfeval_vals.item() if hasattr(selfeval_vals, 'item') else selfeval_vals
                    selfeval_val = 0.0 if np.isnan(selfeval_val) else selfeval_val
                    selfeval_scores = np.full(n_claims, selfeval_val)
            else:
                selfeval_scores = np.zeros(n_claims)

        ordinal_scores = np.arange(n_claims)
        if n_claims > 1:
            ordinal_scores = ordinal_scores / (n_claims - 1)
        else:
            ordinal_scores = np.array([0.5])

        scores_dict = {}
        for model_name, model_data in llm_scores.items():
            if prompt in model_data:
                scores_dict[model_name] = np.array(model_data[prompt][:n_claims])
                scores_dict[model_name] = np.clip(scores_dict[model_name], 0.0, 1.0)
            else:
                scores_dict[model_name] = np.full(n_claims, 0.5)

        valid_llm_scores = []
        for model_name in model_names:
            if model_name in scores_dict:
                valid_llm_scores.append(scores_dict[model_name])

        if valid_llm_scores:
            ensemble_mean = np.mean(valid_llm_scores, axis=0)
            ensemble_std = np.std(valid_llm_scores, axis=0)
            lambda_uncertainty = 0.0
            ensemble_scores = ensemble_mean - lambda_uncertainty * ensemble_std
            ensemble_scores = np.clip(ensemble_scores, 0.0, 1.0)
        else:
            ensemble_scores = np.full(n_claims, 0.5)

        features_4d = np.concatenate((
            freq_scores.reshape(-1, 1),
            selfeval_scores.reshape(-1, 1),
            (logprob_scores / (np.max(logprob_scores) + 1e-8)).reshape(-1, 1),
            ordinal_scores.reshape(-1, 1)
        ), axis=1)

        aligned_data.append({
            'sample': sample,
            'annotations': annotations,
            'scores': {
                'frequency': freq_scores,
                'selfeval': selfeval_scores,
                'logprob': logprob_scores,
                'ensemble': ensemble_scores,
                **scores_dict
            },
            'features_4d': features_4d,
            'prompt_features': np.array([1.0, len(sample.get('response', '')), len(prompt)])
        })

    logger.info(f"✅ Loaded {len(aligned_data)} valid samples")
    return aligned_data, dataset_type
