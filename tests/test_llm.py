import json

import numpy as np
import pytest

from conftest import MODELS, make_entries, make_labelled_records
from maci_conformal import FalseClaimRiskGrouper, LLMClient, LLMError, SubgroupOptimizedMACI
from maci_conformal.cli import main
from maci_conformal.llm import parse_json_reply, score_claims


def write_jsonl(path, records):
    path.write_text('\n'.join(json.dumps(r) for r in records) + '\n')
    return str(path)


@pytest.mark.parametrize('reply', [
    '{"scores": [0.1, 0.9]}',
    'Sure!\n```json\n{"scores": [0.1, 0.9]}\n```',
    '<think>let me see {not json}</think> Here: {"scores": [0.1, 0.9]} done',
    '[0.1, 0.9]',
])
def test_parse_json_reply_variants(reply):
    obj = parse_json_reply(reply)
    assert (obj['scores'] if isinstance(obj, dict) else obj) == [0.1, 0.9]


def test_client_retries_caches_and_parses(mock_openrouter, tmp_path):
    base_url, stats = mock_openrouter
    client = LLMClient(base_url=base_url, cache_dir=str(tmp_path / 'cache'), max_retries=3)
    claims = ['The sky is blue.', 'Water is wrong and dry.']
    for model in ('test/flaky', 'test/prose', 'test/percent'):
        scores = score_claims(client, model, 'Facts?', claims)
        assert scores.shape == (2,) and 0 <= scores.min() and scores.max() <= 1
        assert scores[0] > 0.5 > scores[1], model
    calls = stats['chat']
    score_claims(client, 'test/prose', 'Facts?', claims)
    assert stats['chat'] == calls  # served from the disk cache
    assert client.usage['cached'] == 1


def test_client_errors_are_actionable(mock_openrouter, tmp_path, monkeypatch):
    base_url, _ = mock_openrouter
    client = LLMClient(base_url=base_url, cache=False, max_retries=0)
    with pytest.raises(LLMError, match='maci models'):
        client.chat('test/nope', [{'role': 'user', 'content': 'hi'}])
    monkeypatch.delenv('OPENROUTER_API_KEY')
    with pytest.raises(LLMError, match='OPENROUTER_API_KEY'):
        client.chat('test/good', [{'role': 'user', 'content': 'hi'}])


def test_models_command(mock_openrouter, capsys):
    base_url, _ = mock_openrouter
    assert main(['models', 'good', '--base-url', base_url]) == 0
    out = capsys.readouterr().out
    assert 'test/good' in out and 'test/prose' not in out
    assert main(['models', '--free', '--base-url', base_url]) == 0
    assert 'test/free' in capsys.readouterr().out


def test_calibrate_then_check(mock_openrouter, tmp_path, rng, capsys):
    base_url, stats = mock_openrouter
    data = write_jsonl(tmp_path / 'labelled.jsonl', make_labelled_records(300, rng))
    common = ['--base-url', base_url, '--cache-dir', str(tmp_path / 'cache')]

    # Model ids are checked against the endpoint's list, with suggestions.
    assert main(['calibrate', '--input', data, '--models', 'test/god', '--yes', *common]) == 2
    assert 'did you mean test/good' in capsys.readouterr().err
    # Spending money needs a yes when nobody can be asked.
    assert main(['calibrate', '--input', data, '--models', 'test/good', *common]) == 2
    assert '--yes' in capsys.readouterr().err

    cal = str(tmp_path / 'cal.json')
    assert main(['calibrate', '--input', data, '--models', 'test/good', 'test/prose', '--alpha', '0.1',
                 '--output', cal, '--yes', *common]) == 0
    out = capsys.readouterr().out
    assert 'held-out check' in out and 'Estimated cost' in out
    saved = json.load(open(cal))
    assert saved['model']['model_names'] == ['test/good', 'test/prose']
    assert saved['metadata']['alpha'] == 0.1 and '{claims}' in saved['metadata']['llm']['scoring_prompt']
    assert saved['metadata']['evaluation']['holdout_coverage'] >= 0.8

    # Re-calibrating sends no new requests: every reply is cached.
    calls = stats['chat']
    assert main(['calibrate', '--input', data, '--models', 'test/good', 'test/prose', '--output', cal,
                 *common]) == 0
    assert stats['chat'] == calls and 'already cached' in capsys.readouterr().out

    answer = "Marie Curie won two Nobel Prizes. She was born on the Moon which is wrong. She studied in Paris."
    assert main(['check', '--calibrator', cal, '-q', 'Who was Marie Curie?', '-a', answer,
                 '--extractor-model', 'test/good', *common]) == 0
    out = capsys.readouterr().out
    assert 'Filtered answer:' in out and 'kept' in out

    assert main(['check', '--calibrator', cal, '-q', 'Who was Marie Curie?', '-a', answer, '--json',
                 '--extractor-model', 'test/good', *common]) == 0
    (row,) = [json.loads(line) for line in capsys.readouterr().out.splitlines()]
    texts = [c['text'] for c in row['claims']]
    assert len(texts) == 3 and all(set(c['model_scores']) == {'test/good', 'test/prose'} for c in row['claims'])
    moon = row['claims'][texts.index('She was born on the Moon which is wrong.')]
    assert not moon['kept'] and moon['score'] < 0.5
    assert 'Moon' not in row['filtered_answer']

    # Batch mode with labels reports coverage on them.
    batch = write_jsonl(tmp_path / 'batch.jsonl', make_labelled_records(30, np.random.default_rng(9)))
    args = ['check', '--calibrator', cal, '--input', batch, '--output', str(tmp_path / 'checked.jsonl'),
            '--limit', '1', *common]
    assert main(args) == 2  # 60 requests: big batches need --yes when nobody can be asked
    assert '--yes' in capsys.readouterr().err
    assert main([*args, '--yes']) == 0
    out = capsys.readouterr().out
    assert 'On 30 labelled answers: coverage' in out
    assert len((tmp_path / 'checked.jsonl').read_text().splitlines()) == 30


def test_score_then_run_with_new_models(mock_openrouter, fake_wikibio_dir, tmp_path, capsys):
    base_url, _ = mock_openrouter
    scores_dir = tmp_path / 'scores'
    assert main(['score', '--models', 'test/good', 'test/prose', '--data-dir', fake_wikibio_dir, '--limit', '120',
                 '--scores-dir', str(scores_dir), '--base-url', base_url, '--cache-dir', str(tmp_path / 'c'),
                 '--yes']) == 0
    assert sorted(p.name for p in scores_dir.iterdir()) == ['wikibio_scores_test_good.npz',
                                                           'wikibio_scores_test_prose.npz']
    capsys.readouterr()
    assert main(['run', '--data-dir', fake_wikibio_dir, '--scores-dir', str(scores_dir), '--methods', 'MACI',
                 '--models', 'test/good', 'test/prose', '--n-runs', '1', '--limit-samples', '120',
                 '--output-dir', str(tmp_path / 'out')]) == 0
    assert 'have no scores' not in capsys.readouterr().err


def test_calibrator_save_load_predicts_identically(rng, tmp_path):
    calib, test = make_entries(600, rng), make_entries(200, rng, offset=10_000)
    fitted = SubgroupOptimizedMACI(model_names=MODELS, grouper=FalseClaimRiskGrouper(), random_state=7)
    fitted.fit(calib, alpha=0.1)
    fitted.save(str(tmp_path / 'm.json'), metadata={'note': 'x'})
    loaded = SubgroupOptimizedMACI.load(str(tmp_path / 'm.json'))
    assert loaded.metadata == {'note': 'x'} and loaded.alpha == 0.1
    a, _ = fitted.predict(test)
    b, _ = loaded.predict(test)
    assert [x['sample']['kept_mask'] for x in a] == [x['sample']['kept_mask'] for x in b]
    assert [x['group'] for x in a] == [x['group'] for x in b]


def test_uncalibrated_group_abstains(rng):
    calib = make_entries(400, rng)
    maci = SubgroupOptimizedMACI(model_names=MODELS, grouper=FalseClaimRiskGrouper(), random_state=0)
    maci.fit(calib, alpha=0.1)
    del maci.conformal_models['high']
    filtered, rates = maci.predict(make_entries(200, rng, offset=10_000))
    high = [(f, r) for f, r in zip(filtered, rates) if f['group'] == 'high']
    assert high and all(r == 0.0 and f['uncalibrated_group'] for f, r in high)
