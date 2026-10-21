import io
import json
import os

import pytest

from maci_conformal import data as data_mod
from maci_conformal.cli import main
from maci_conformal.config import CONFIG_TEMPLATE, Config, config_from_dict, load_config


def test_template_matches_defaults(tmp_path):
    path = tmp_path / 'maci.toml'
    path.write_text(CONFIG_TEMPLATE)
    loaded = load_config(str(path))
    expected = Config()
    expected.output.output_dir = str(tmp_path / 'maci_outputs')
    assert loaded == expected


def test_unknown_key_suggests_fix():
    with pytest.raises(ValueError, match="did you mean 'alpha'"):
        config_from_dict({'experiment': {'alhpa': 0.2}})
    with pytest.raises(ValueError, match="Unknown config section"):
        config_from_dict({'experiments': {}})


def test_validation_and_group_alias():
    cfg = config_from_dict({'experiment': {'conditional_groups': ['medicalcontent']}}).validate()
    assert cfg.experiment.conditional_groups == ['medical_content']
    with pytest.raises(ValueError, match='experiment.alpha'):
        config_from_dict({'experiment': {'alpha': 1.5}}).validate()
    with pytest.raises(ValueError, match='experiment.methods'):
        config_from_dict({'experiment': {'methods': ['XYZ']}}).validate()


def test_relative_paths_resolve_against_config_file(tmp_path):
    sub = tmp_path / 'cfgs'
    sub.mkdir()
    (sub / 'maci.toml').write_text('[data]\ndata_dir = "../mydata"\n')
    assert load_config(str(sub / 'maci.toml')).data.data_dir == str(tmp_path / 'mydata')


def test_config_init_and_show(tmp_path, capsys):
    assert main(['config', 'init']) == 0
    assert (tmp_path / 'maci.toml').exists()
    assert main(['config', 'init']) == 1  # refuses to overwrite

    capsys.readouterr()
    assert main(['config', 'show', '--alpha', '0.25', '--methods', 'MACI']) == 0
    out = capsys.readouterr().out
    assert 'alpha = 0.25' in out
    assert 'methods = ["MACI"]' in out


def test_run_without_data_gives_actionable_error(capsys):
    assert main(['run', '--methods', 'MACI', '--n-runs', '1']) == 2
    assert 'maci data download' in capsys.readouterr().err


def test_default_data_dir_respects_env(monkeypatch, tmp_path):
    monkeypatch.setenv('MACI_DATA_DIR', str(tmp_path / 'd'))
    assert data_mod.default_data_dir() == str(tmp_path / 'd')
    monkeypatch.delenv('MACI_DATA_DIR')
    monkeypatch.setenv('XDG_CACHE_HOME', str(tmp_path / 'cache'))
    assert data_mod.default_data_dir() == str(tmp_path / 'cache' / 'maci' / 'data')


def test_download_verifies_checksums(monkeypatch, tmp_path):
    import hashlib
    payload = b'hello maci'
    files = {'wikibio': {'wiki_scores/a.bin': hashlib.sha256(payload).hexdigest()}}
    monkeypatch.setattr(data_mod, 'DATA_FILES', files)
    calls = []

    def fake_urlopen(url, timeout=None):
        calls.append(url)
        return io.BytesIO(payload)

    monkeypatch.setattr(data_mod.urllib.request, 'urlopen', fake_urlopen)
    out = data_mod.download_data(['wikibio'], data_dir=str(tmp_path / 'dl'))
    assert (tmp_path / 'dl' / 'wiki_scores' / 'a.bin').read_bytes() == payload
    assert data_mod.data_status(out, ['wikibio']) == {'wikibio': {'wiki_scores/a.bin': 'ok'}}

    data_mod.download_data(['wikibio'], data_dir=out)
    assert len(calls) == 1  # already present, skipped

    files['wikibio']['wiki_scores/a.bin'] = '0' * 64
    with pytest.raises(IOError, match='Checksum mismatch'):
        data_mod.download_data(['wikibio'], data_dir=out, force=True)
    assert not [p for p in os.listdir(tmp_path / 'dl' / 'wiki_scores') if p.endswith('.part')]


def _methods_available():
    methods = ['BCI', 'MACI']
    try:
        import maci_conformal.conditional  # noqa: F401
        methods.append('CCI')
    except ImportError:
        pass
    return methods


def test_end_to_end_run(fake_wikibio_dir, tmp_path, capsys):
    methods = _methods_available()
    code = main(['run', '--data-dir', fake_wikibio_dir, '--dataset', 'wikibio', '--n-runs', '2',
                 '--boosting-epochs', '2', '--output-dir', str(tmp_path / 'out'), '--methods', *methods])
    assert code == 0
    (run_dir,) = list((tmp_path / 'out').iterdir())
    for name in ('results.json', 'predictions.jsonl', 'report.html', 'config.toml', 'log.txt'):
        assert (run_dir / name).exists(), name
    results = json.loads((run_dir / 'results.json').read_text())
    assert set(results) == set(methods)
    assert len(results['MACI']['coverage']) == 2
    assert set(results['MACI']['subgroup_results'][0]) == {
        'FalseClaimRiskGrouper', 'MedicalContentGrouper', 'ViewCountGrouper'}

    rows = [json.loads(line) for line in (run_dir / 'predictions.jsonl').read_text().splitlines()]
    maci_rows = [r for r in rows if r['method'] == 'MACI' and r['grouper'] == 'false_claim_risk' and r['run'] == 1]
    assert maci_rows and all(r['group'] in ('low', 'medium', 'high') for r in maci_rows)
    # The saved per-claim flags reproduce the reported coverage exactly.
    covered = sum(r['covered'] for r in rows if r['method'] == 'MACI' and r['grouper'] == 'false_claim_risk'
                  and r['run'] == 1) / len(maci_rows)
    assert abs(covered - results['MACI']['coverage'][0]) < 1e-12
    for r in maci_rows:
        assert r['n_kept'] == sum(c['kept'] for c in r['claims'])
        assert set(r['claims'][0]['model_scores']) == set(load_config().maci.models)

    html_text = (run_dir / 'report.html').read_text()
    assert 'MACI benchmark report' in html_text and 'details class=s' in html_text
    # The saved config reproduces the run's settings.
    saved = load_config(str(run_dir / 'config.toml'))
    assert saved.experiment.methods == methods and saved.data.data_dir == fake_wikibio_dir

    capsys.readouterr()
    assert main(['inspect', str(run_dir), '--method', 'MACI', '--limit', '2']) == 0
    out = capsys.readouterr().out
    assert 'false removed' in out and '[1]' in out


def test_legacy_flat_layout(fake_wikibio_dir, tmp_path):
    code = main(['run', '--data-dir', fake_wikibio_dir, '--n-runs', '1', '--methods', 'BCI', 'MACI',
                 '--log-dir', str(tmp_path / 'logs'), '--results-dir', str(tmp_path / 'res'), '--time-profile'])
    assert code == 0
    assert len(list((tmp_path / 'res').glob('results_wikibio_fixed_*.json'))) == 1
    assert len(list((tmp_path / 'res').glob('predictions_wikibio_fixed_*.jsonl'))) == 1
    assert len(list((tmp_path / 'res').glob('time_profile_*_run1.json'))) == 1
    assert len(list((tmp_path / 'logs').glob('experiment_log_*.log'))) == 1


def test_legacy_defaults_do_not_override_config(tmp_path, fake_wikibio_dir):
    from maci_conformal.cli import _resolve_config, build_parser
    (tmp_path / 'mine.toml').write_text(f'[data]\ndata_dir = "{fake_wikibio_dir}"\n[output]\noutput_dir = "o"\n')
    args = build_parser().parse_args(['run', '-c', str(tmp_path / 'mine.toml')])
    defaults = {('data', 'data_dir'): '/repo/data', ('output', 'log_dir'): '/repo/logs',
                ('output', 'results_dir'): '/repo/res', ('experiment', 'alpha'): 0.3}
    cfg = _resolve_config(args, defaults)
    assert cfg.data.data_dir == fake_wikibio_dir
    assert cfg.output.log_dir is None and cfg.output.results_dir is None
    assert cfg.experiment.alpha == 0.3  # not set anywhere else, so the default applies


def test_run_fails_loudly(fake_wikibio_dir, tmp_path, capsys):
    # A model without a score file is an error, not a silent 0.5.
    assert main(['run', '--data-dir', fake_wikibio_dir, '--models', 'nobody/none', '--methods', 'MACI',
                 '--output-dir', str(tmp_path / 'o')]) == 2
    assert 'maci score' in capsys.readouterr().err
    # A scores folder that does not exist is an error.
    assert main(['run', '--data-dir', fake_wikibio_dir, '--scores-dir', str(tmp_path / 'typo'),
                 '--output-dir', str(tmp_path / 'o')]) == 2
    # An explicit view-count CSV that does not exist is an error.
    assert main(['run', '--data-dir', fake_wikibio_dir, '--view-metadata-csv', str(tmp_path / 'nope.csv'),
                 '--output-dir', str(tmp_path / 'o')]) == 2


def test_data_commands_without_dataset_argument(tmp_path, capsys):
    # argparse nargs='*' with choices fails on an empty list before Python 3.12.
    assert main(['data', 'status', '--data-dir', str(tmp_path / 'empty')]) == 1
    out = capsys.readouterr().out
    assert 'wikibio' in out and 'medlfqa' not in out
    assert main(['data', 'status', 'nonsense']) == 2


def test_boolean_flags_switch_both_ways(tmp_path):
    from maci_conformal.cli import _resolve_config, build_parser
    (tmp_path / 'maci.toml').write_text('[output]\ntime_profile = true\n[cci]\nadaptive_alpha = true\n')
    args = build_parser().parse_args(['run', '--no-time-profile', '--no-adaptive-alpha'])
    cfg = _resolve_config(args)
    assert cfg.output.time_profile is False and cfg.cci.adaptive_alpha is False


def test_config_types_are_checked():
    for raw, msg in [({'maci': {'models': 'qwen/qwen-2.5-72b-instruct'}}, 'must be a list'),
                     ({'experiment': {'alpha': '0.1'}}, 'must be a number'),
                     ({'experiment': {'n_runs': 2.5}}, 'whole number'),
                     ({'output': {'time_profile': 'no'}}, 'true or false'),
                     ({'data': {'data_dir': 5}}, 'must be a string')]:
        with pytest.raises(ValueError, match=msg):
            config_from_dict(raw)
    for raw, msg in [({'maci': {'n_bins': 0}}, 'n_bins'), ({'data': {'limit_samples': -5}}, 'limit_samples'),
                     ({'experiment': {'methods': []}}, 'at least one method'),
                     ({'experiment': {'conditional_groups': []}}, 'conditional_groups'),
                     ({'cci': {'boosting_lr': -1.0}}, 'boosting_lr')]:
        with pytest.raises(ValueError, match=msg):
            config_from_dict(raw).validate()


def test_config_show_round_trips(tmp_path, capsys):
    from maci_conformal.config import dump_config
    cfg = config_from_dict({'llm': {'scoring_prompt': 'Q: {question}\n"{claims}"\tn={n}\x7f'},
                            'experiment': {'alpha': 0.2}}).validate()
    path = tmp_path / 'elsewhere' / 'shown.toml'
    path.parent.mkdir()
    path.write_text(dump_config(cfg))
    assert load_config(str(path)) == cfg
