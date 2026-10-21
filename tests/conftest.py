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


# ---------------------------------------------------------------------------
# A local stand-in for the OpenRouter API
# ---------------------------------------------------------------------------

MOCK_KEY = 'test-key'
MOCK_MODELS = ['test/good', 'test/prose', 'test/flaky', 'test/percent']


def _mock_scores(model, claims):
    import hashlib
    out = []
    for c in claims:
        jitter = int(hashlib.md5(f"{model}|{c}".encode()).hexdigest()[:4], 16) / 65535.0 - 0.5
        base = 0.15 if 'wrong' in c.lower() else 0.85
        out.append(round(min(1.0, max(0.0, base + (0.3 if model == 'test/prose' else 0.2) * jitter)), 3))
    return out



@pytest.fixture
def mock_openrouter(monkeypatch):
    """Serve /api/v1/models and /api/v1/chat/completions on localhost; yields (base_url, stats)."""
    import json
    import re
    import threading
    from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

    stats = {'chat': 0, 'flaky_seen': set()}
    lock = threading.Lock()

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def _send(self, code, body):
            data = json.dumps(body).encode()
            self.send_response(code)
            self.send_header('Content-Type', 'application/json')
            self.send_header('Content-Length', str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def do_GET(self):
            if self.path.endswith('/models'):
                models = [{'id': m, 'name': m.split('/')[1].title(), 'context_length': 8192,
                           'pricing': {'prompt': '0.000001', 'completion': '0.000002'}} for m in MOCK_MODELS]
                models.append({'id': 'test/free', 'name': 'Free', 'context_length': 4096,
                               'pricing': {'prompt': '0', 'completion': '0'}})
                return self._send(200, {'data': models})
            self._send(404, {'error': {'message': 'not found'}})

        def do_POST(self):
            if self.headers.get('Authorization') != f'Bearer {MOCK_KEY}':
                return self._send(401, {'error': {'message': 'No auth credentials found', 'code': 401}})
            body = json.loads(self.rfile.read(int(self.headers['Content-Length'])))
            model, content = body['model'], body['messages'][0]['content']
            if model not in MOCK_MODELS:
                return self._send(404, {'error': {'message': f'{model} is not a valid model ID', 'code': 404}})
            with lock:
                stats['chat'] += 1
                if model == 'test/flaky' and content not in stats['flaky_seen']:
                    stats['flaky_seen'].add(content)
                    return self._send(429, {'error': {'message': 'Rate limited', 'code': 429}})
            if 'Split the answer below into atomic claims' in content:
                answer = content.split('Answer:\n', 1)[1].split('\n\nReply with JSON', 1)[0]
                claims = [c.strip() + '.' for c in answer.split('.') if c.strip()]
                reply = json.dumps({'claims': claims})
            else:
                claims = re.findall(r'^\d+\. (.*)$', content.split('Claims:\n', 1)[1], flags=re.M)
                scores = _mock_scores(model, claims)
                if model == 'test/percent':
                    scores = [round(100 * s) for s in scores]
                reply = json.dumps({'scores': scores})
                if model == 'test/prose':
                    reply = f"<think>hmm</think>Sure! Here you go:\n```json\n{reply}\n```\nHope that helps."
            self._send(200, {'choices': [{'message': {'role': 'assistant', 'content': reply}}],
                             'usage': {'prompt_tokens': len(content) // 4, 'completion_tokens': 20}})

    server = ThreadingHTTPServer(('127.0.0.1', 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    monkeypatch.setenv('OPENROUTER_API_KEY', MOCK_KEY)
    try:
        yield f"http://127.0.0.1:{server.server_address[1]}/api/v1", stats
    finally:
        server.shutdown()
        server.server_close()


def make_labelled_records(n, rng):
    """Calibration records whose false claims usually contain the word 'wrong' (which the mock scores low)."""
    records = []
    for i in range(n):
        claims = []
        for j in range(int(rng.integers(3, 8))):
            true = bool(rng.random() < 0.7)
            text = f"Person {i} did thing {j}"
            if not true and rng.random() < 0.85:
                text += " which is wrong"
            claims.append({'text': text + '.', 'is_supported': true})
        records.append({'question': f'Tell me about person {i}.',
                        'answer': ' '.join(c['text'] for c in claims) + ' ' + ' '.join(rng.choice(WORDS, size=int(rng.integers(5, 200)))),
                        'claims': claims})
    return records
