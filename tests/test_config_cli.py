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


def test_end_to_end_run(fake_wikibio_dir, tmp_path):
    methods = ['BCI', 'MACI']
    try:
        import maci_conformal.conditional  # noqa: F401
        methods.append('CCI')
    except ImportError:
        pass
    code = main(['run', '--data-dir', fake_wikibio_dir, '--dataset', 'wikibio', '--n-runs', '2',
                 '--boosting-epochs', '2', '--output-dir', str(tmp_path / 'out'), '--methods', *methods])
    assert code == 0
    results_dir = tmp_path / 'out' / 'results'
    (result_file,) = list(results_dir.glob('results_wikibio_fixed_*.json'))
    results = json.loads(result_file.read_text())
    assert set(results) == set(methods)
    assert len(results['MACI']['coverage']) == 2
    assert set(results['MACI']['subgroup_results'][0]) == {
        'FalseClaimRiskGrouper', 'MedicalContentGrouper', 'ViewCountGrouper'}
    assert list((tmp_path / 'out' / 'logs').glob('experiment_log_*.log'))
