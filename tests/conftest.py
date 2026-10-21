import os
import pickle

import numpy as np
import pytest

MODELS = ['model-a', 'model-b']
WORDS = 'the of and to in is was for on that with as by at from his her it an'.split()


def make_entries(n, rng, models=MODELS, p_true=0.7, offset=0):
    """Synthetic claim sets whose model scores are informative about the labels."""
    entries = []
    for i in range(n):
        k = int(rng.integers(3, 10))
        labels = rng.random(k) < p_true
        facts = [{'atom': f'claim {i}-{j}', 'is_supported': bool(lab)} for j, lab in enumerate(labels)]
        response = ' '.join(rng.choice(WORDS, size=int(rng.integers(20, 400))))
        scores = {m: np.clip(0.45 * labels + 0.55 * rng.random(k), 0.0, 1.0) for m in models}
        entries.append({
            'sample': {'prompt': f'Please write one biographical paragraph about Person {offset + i}.',
                       'response': response, 'atomic_facts': facts},
            'scores': scores,
        })
    return entries


@pytest.fixture
def rng():
    return np.random.default_rng(0)


@pytest.fixture
def fake_wikibio_dir(tmp_path):
    """A tiny data directory laid out like the real WikiBio files."""
    rng = np.random.default_rng(1)
    models = ['qwen-2.5-72b-instruct', 'deepseek-chat-v3-0324', 'llama-3.3-70b-instruct']
    entries = make_entries(120, rng, models=models)
    wiki = tmp_path / 'wiki_scores'
    wiki.mkdir()
    samples = [e['sample'] for e in entries]
    with open(wiki / 'wikibio_final_dataset.pkl', 'wb') as f:
        pickle.dump(samples, f)

    def per_claim(fn):
        return {f'arr_{i}': fn(len(s['atomic_facts'])) for i, s in enumerate(samples)}

    np.savez(wiki / 'wikibio_final_frequencies.npz', **per_claim(lambda k: rng.random(k)))
    np.savez(wiki / 'wikibio_final_logprobs.npz', **per_claim(lambda k: rng.random(k)))
    np.savez(wiki / 'wikibio_final_self_evals.npz', **per_claim(lambda k: rng.random(k)))
    prompts = np.array([s['prompt'] for s in samples], dtype=object)
    for model, filename in zip(models, ['qwen_qwen-2.5-72b-instruct', 'deepseek-chat-v3-0324',
                                        'meta-llama_llama-3.3-70b-instruct']):
        scores_list = np.empty(len(entries), dtype=object)
        for i, e in enumerate(entries):
            scores_list[i] = list(e['scores'][model])
        np.savez(wiki / f'wikibio_scores_{filename}.npz', prompts=prompts, scores_list=scores_list)
    with open(wiki / 'wikibio_final.csv', 'w') as f:
        f.write('Name,Views\n')
        for i in range(len(samples)):
            f.write(f'Person {i},{int(rng.integers(10, 10000))}\n')
    return str(tmp_path)


@pytest.fixture(autouse=True)
def _isolate_env(tmp_path, monkeypatch):
    monkeypatch.setenv('MACI_DATA_DIR', os.path.join(str(tmp_path), 'no-data-here'))
    monkeypatch.chdir(tmp_path)
