"""LLM access through OpenRouter (or any OpenAI-compatible API): claim extraction and claim scoring.

Only the standard library is used. Set ``OPENROUTER_API_KEY`` (get a key at
https://openrouter.ai/keys) and pick models by their OpenRouter id, e.g.
``meta-llama/llama-3.3-70b-instruct``; ``maci models`` lists them.
"""

import hashlib
import json
import logging
import os
import random
import re
import threading
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Callable, Dict, List, Optional, Sequence

import numpy as np

from . import __version__

logger = logging.getLogger(__name__)

OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
_RETRY_STATUS = {408, 409, 425, 429, 500, 502, 503, 504}

EXTRACTION_PROMPT = """Split the answer below into atomic claims.

An atomic claim is one short, self-contained factual statement that can be checked on its own. Replace pronouns with the names they refer to, keep numbers, dates and names exactly as written, and leave out opinions, advice, questions and filler. Do not add facts that are not in the answer.

Question: {question}

Answer:
{answer}

Reply with JSON only, in this form: {{"claims": ["...", "..."]}}"""

SCORING_PROMPT = """You are checking the factual accuracy of claims taken from an answer to a question.

Question: {question}

Claims:
{claims}

For each claim, give the probability (a number from 0 to 1) that it is factually true. Judge each claim on its own using your own knowledge; do not assume the answer is correct. Use values near 0.5 when you are unsure.

Reply with JSON only, with exactly {n} numbers in the same order as the claims: {{"scores": [0.9, 0.1, ...]}}"""


class LLMError(RuntimeError):
    pass


def _default_cache_dir() -> str:
    cache_home = os.environ.get('XDG_CACHE_HOME') or os.path.join(os.path.expanduser('~'), '.cache')
    return os.path.join(cache_home, 'maci', 'llm')


def check_template(template: str, required: Sequence[str]) -> None:
    missing = [name for name in required if '{' + name + '}' not in template]
    if missing:
        raise ValueError(f"Prompt template is missing placeholder(s): {', '.join('{' + m + '}' for m in missing)}")


class LLMClient:
    """Small chat-completions client with retries, a disk cache and a thread pool."""

    def __init__(self, base_url: str = OPENROUTER_BASE_URL, api_key: Optional[str] = None,
                 api_key_env: str = 'OPENROUTER_API_KEY', timeout: float = 120.0, max_retries: int = 5,
                 temperature: float = 0.0, max_tokens: Optional[int] = None, max_workers: int = 8,
                 cache: bool = True, cache_dir: Optional[str] = None):
        self.base_url = base_url.rstrip('/')
        self.api_key = api_key
        self.api_key_env = api_key_env
        self.timeout = timeout
        self.max_retries = max_retries
        self.temperature = temperature
        self.max_tokens = max_tokens or None
        self.max_workers = max(1, int(max_workers))
        self.cache_dir = (cache_dir or _default_cache_dir()) if cache else None
        self.usage = {'requests': 0, 'cached': 0, 'prompt_tokens': 0, 'completion_tokens': 0, 'cost': 0.0}
        self._lock = threading.Lock()

    @classmethod
    def from_config(cls, llm_cfg) -> 'LLMClient':
        return cls(base_url=llm_cfg.base_url, api_key_env=llm_cfg.api_key_env, timeout=llm_cfg.timeout,
                   temperature=llm_cfg.temperature, max_tokens=llm_cfg.max_tokens, max_workers=llm_cfg.max_workers,
                   cache=llm_cfg.cache, cache_dir=llm_cfg.cache_dir)

    @property
    def is_openrouter(self) -> bool:
        return 'openrouter.ai' in self.base_url

    def _key(self, required: bool = True) -> Optional[str]:
        key = self.api_key or os.environ.get(self.api_key_env)
        if not key and required:
            where = " (create one at https://openrouter.ai/keys)" if self.is_openrouter else ""
            raise LLMError(f"No API key: set the {self.api_key_env} environment variable{where}.")
        return key

    def _headers(self, key: Optional[str]) -> Dict[str, str]:
        headers = {'Content-Type': 'application/json', 'User-Agent': f'maci-conformal/{__version__}'}
        if key:
            headers['Authorization'] = f'Bearer {key}'
        if self.is_openrouter:
            headers['HTTP-Referer'] = 'https://github.com/MLAI-Yonsei/maci-conformal'
            headers['X-Title'] = 'maci-conformal'
        return headers

    def _request(self, method: str, path: str, payload: Optional[dict] = None, key_required: bool = True) -> dict:
        url = f"{self.base_url}/{path.lstrip('/')}"
        data = None if payload is None else json.dumps(payload).encode('utf-8')
        headers = self._headers(self._key(required=key_required))
        last_error: Optional[str] = None
        for attempt in range(self.max_retries + 1):
            req = urllib.request.Request(url, data=data, headers=headers, method=method)
            try:
                with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                    body = json.loads(resp.read().decode('utf-8'))
                if isinstance(body, dict) and body.get('error') and not body.get('choices') and not body.get('data'):
                    err = body['error']
                    code = err.get('code') if isinstance(err, dict) else None
                    msg = err.get('message') if isinstance(err, dict) else str(err)
                    if code in _RETRY_STATUS and attempt < self.max_retries:
                        last_error = f"{code}: {msg}"
                        time.sleep(self._backoff(attempt))
                        continue
                    raise LLMError(f"{url} returned an error: {msg}")
                return body
            except urllib.error.HTTPError as e:
                detail = _error_message(e)
                if e.code in _RETRY_STATUS and attempt < self.max_retries:
                    last_error = f"HTTP {e.code}: {detail}"
                    time.sleep(self._backoff(attempt, e.headers.get('Retry-After') if e.headers else None))
                    continue
                hint = {401: " Check your API key.", 402: " Your account is out of credits.",
                        404: " Check the model id with `maci models`."}.get(e.code, "")
                raise LLMError(f"HTTP {e.code} from {url}: {detail}.{hint}") from None
            except (urllib.error.URLError, TimeoutError, ConnectionError, json.JSONDecodeError) as e:
                if attempt < self.max_retries:
                    last_error = str(e)
                    time.sleep(self._backoff(attempt))
                    continue
                raise LLMError(f"Request to {url} failed: {e}") from None
        raise LLMError(f"Request to {url} failed after {self.max_retries + 1} attempts: {last_error}")

    @staticmethod
    def _backoff(attempt: int, retry_after: Optional[str] = None) -> float:
        if retry_after:
            try:
                return min(60.0, float(retry_after))
            except ValueError:
                pass
        return min(60.0, 2.0 ** attempt + random.random())

    def _cache_path(self, model: str, messages: List[dict], tag: str) -> Optional[str]:
        if not self.cache_dir:
            return None
        blob = json.dumps({'base_url': self.base_url, 'model': model, 'messages': messages,
                           'temperature': self.temperature, 'max_tokens': self.max_tokens, 'tag': tag},
                          sort_keys=True, ensure_ascii=False)
        digest = hashlib.sha256(blob.encode('utf-8')).hexdigest()
        return os.path.join(self.cache_dir, digest[:2], digest + '.json')

    def is_cached(self, model: str, messages: List[dict], tag: str = '') -> bool:
        path = self._cache_path(model, messages, tag)
        return bool(path and os.path.exists(path))

    def chat(self, model: str, messages: List[dict], tag: str = '') -> str:
        """Return the assistant message text for ``messages`` (cached on disk when enabled)."""
        path = self._cache_path(model, messages, tag)
        if path and os.path.exists(path):
            try:
                with open(path, 'r', encoding='utf-8') as f:
                    content = json.load(f)['content']
                with self._lock:
                    self.usage['cached'] += 1
                return content
            except (OSError, ValueError, KeyError):
                pass
        payload: Dict[str, Any] = {'model': model, 'messages': messages, 'temperature': self.temperature}
        if self.max_tokens:
            payload['max_tokens'] = self.max_tokens
        body = self._request('POST', 'chat/completions', payload)
        try:
            content = body['choices'][0]['message'].get('content') or ''
        except (KeyError, IndexError, TypeError, AttributeError):
            raise LLMError(f"Unexpected response from {model}: {str(body)[:300]}") from None
        usage = body.get('usage') or {}
        with self._lock:
            self.usage['requests'] += 1
            self.usage['prompt_tokens'] += int(usage.get('prompt_tokens') or 0)
            self.usage['completion_tokens'] += int(usage.get('completion_tokens') or 0)
            self.usage['cost'] += float(usage.get('cost') or 0.0)
        if path:
            os.makedirs(os.path.dirname(path), exist_ok=True)
            tmp = f"{path}.{os.getpid()}.{threading.get_ident()}.tmp"
            with open(tmp, 'w', encoding='utf-8') as f:
                json.dump({'model': model, 'content': content}, f, ensure_ascii=False)
            os.replace(tmp, path)
        return content

    def map(self, fn: Callable[[Any], Any], items: Sequence[Any],
            progress: Optional[Callable[[int, int], None]] = None) -> List[Any]:
        """Apply ``fn`` to every item in a thread pool, keeping the input order."""
        results: List[Any] = [None] * len(items)
        if not items:
            return results
        with ThreadPoolExecutor(max_workers=self.max_workers) as pool:
            futures = {pool.submit(fn, item): i for i, item in enumerate(items)}
            done = 0
            for fut in as_completed(futures):
                results[futures[fut]] = fut.result()
                done += 1
                if progress:
                    progress(done, len(items))
        return results

    def list_models(self) -> List[Dict[str, Any]]:
        """Models the endpoint offers (OpenRouter needs no API key for this)."""
        body = self._request('GET', 'models', key_required=False)
        return list(body.get('data') or [])

    def usage_summary(self) -> str:
        u = self.usage
        text = (f"{u['requests']} API requests ({u['cached']} answered from cache), "
                f"{u['prompt_tokens']:,} prompt + {u['completion_tokens']:,} completion tokens")
        if u['cost']:
            text += f", ${u['cost']:.4f}"
        return text


def _error_message(e: urllib.error.HTTPError) -> str:
    try:
        body = json.loads(e.read().decode('utf-8'))
        err = body.get('error', body)
        return err.get('message', str(err)) if isinstance(err, dict) else str(err)
    except Exception:
        return e.reason if isinstance(e.reason, str) else str(e)


# ---------------------------------------------------------------------------
# Parsing model replies
# ---------------------------------------------------------------------------

_THINK_RE = re.compile(r'<think>.*?</think>', re.DOTALL | re.IGNORECASE)
_FENCE_RE = re.compile(r'```(?:json)?\s*(.*?)```', re.DOTALL | re.IGNORECASE)


def parse_json_reply(text: str) -> Any:
    """Pull the first JSON object or array out of a model reply (tolerates prose and code fences)."""
    text = _THINK_RE.sub('', text or '').strip()
    fence = _FENCE_RE.search(text)
    candidates = [fence.group(1).strip()] if fence else []
    candidates.append(text)
    decoder = json.JSONDecoder()
    for cand in candidates:
        try:
            return json.loads(cand)
        except ValueError:
            pass
        for i, ch in enumerate(cand):
            if ch in '{[':
                try:
                    return decoder.raw_decode(cand[i:])[0]
                except ValueError:
                    continue
    raise ValueError("no JSON found in reply")


def _parse_scores(text: str, n: int) -> np.ndarray:
    obj = parse_json_reply(text)
    if isinstance(obj, dict):
        for key in ('scores', 'probabilities', 'probs', 'p'):
            if key in obj:
                obj = obj[key]
                break
    if not isinstance(obj, list) or len(obj) != n:
        raise ValueError(f"expected a list of {n} scores")
    vals = np.array([float(v['score'] if isinstance(v, dict) else v) for v in obj], dtype=float)
    if not np.all(np.isfinite(vals)):
        raise ValueError("non-numeric score")
    if vals.max(initial=0.0) > 1.0 and vals.max() <= 100.0:
        vals = vals / 100.0  # some models answer in percent
    return np.clip(vals, 0.0, 1.0)


def _parse_claims(text: str) -> List[str]:
    obj = parse_json_reply(text)
    if isinstance(obj, dict):
        obj = obj.get('claims', obj.get('facts'))
    if not isinstance(obj, list):
        raise ValueError("expected a list of claims")
    claims, seen = [], set()
    for c in obj:
        c = (c.get('text') or c.get('claim') or '') if isinstance(c, dict) else str(c)
        c = c.strip()
        if c and c not in seen:
            seen.add(c)
            claims.append(c)
    return claims


# ---------------------------------------------------------------------------
# Claim extraction and scoring
# ---------------------------------------------------------------------------

def scoring_messages(question: str, claims: Sequence[str], template: str = SCORING_PROMPT) -> List[dict]:
    numbered = '\n'.join(f"{i + 1}. {c}" for i, c in enumerate(claims))
    return [{'role': 'user', 'content': template.format(question=question, claims=numbered, n=len(claims))}]


def extraction_messages(question: str, answer: str, template: str = EXTRACTION_PROMPT) -> List[dict]:
    return [{'role': 'user', 'content': template.format(question=question, answer=answer)}]


def score_claims(client: LLMClient, model: str, question: str, claims: Sequence[str],
                 template: str = SCORING_PROMPT, attempts: int = 3) -> np.ndarray:
    """Probability-of-truth per claim from one model. Unparseable replies give NaN (treated as untrusted)."""
    if not claims:
        return np.array([], dtype=float)
    messages = scoring_messages(question, claims, template)
    for attempt in range(attempts):
        reply = client.chat(model, messages, tag='' if attempt == 0 else f'retry{attempt}')
        try:
            return _parse_scores(reply, len(claims))
        except (ValueError, TypeError, KeyError):
            continue
    logger.warning(f"{model}: could not read scores for {len(claims)} claims after {attempts} tries; "
                   "they count as untrusted")
    return np.full(len(claims), np.nan)


def extract_claims(client: LLMClient, model: str, question: str, answer: str,
                   template: str = EXTRACTION_PROMPT, attempts: int = 3) -> List[str]:
    messages = extraction_messages(question, answer, template)
    for attempt in range(attempts):
        reply = client.chat(model, messages, tag='' if attempt == 0 else f'retry{attempt}')
        try:
            return _parse_claims(reply)
        except (ValueError, TypeError, KeyError, AttributeError):
            continue
    raise LLMError(f"{model} did not return a usable claim list after {attempts} tries")


def estimate_cost(models_info: Dict[str, Dict[str, Any]], requests: Sequence[Dict[str, Any]]) -> Optional[float]:
    """Rough USD estimate from OpenRouter prices: ~4 characters per prompt token."""
    total = 0.0
    for r in requests:
        info = models_info.get(r['model'])
        if not info or 'pricing' not in info:
            return None
        price = info['pricing']
        total += float(price.get('prompt') or 0) * r['prompt_chars'] / 4.0
        total += float(price.get('completion') or 0) * r['completion_tokens']
        total += float(price.get('request') or 0)
    return total
