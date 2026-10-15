from pathlib import Path
import json
import os
import shutil
import subprocess
import sys

import numpy as np
import pandas as pd
import pytest
import torch

import run_experiments as cli
from src.experiments import runner as workflow
from src.data.preparation import prepare_data
from src.features.causal import add_features
from src.features.window_wavelet import wavelet_inputs
from src.models.neural.lstm_model import StockLSTM
from src.models.fuzzy.anfis import ANFIS
from src.models.neural.checkpoints import load_model, export_weights


@pytest.fixture
def raw():
    rng = np.random.default_rng(9)
    close = 100 + np.cumsum(rng.normal(.05, 1, 220))
    return pd.DataFrame({"Open": close-.2, "High": close+1, "Low": close-1,
                         "Close": close, "Volume": rng.integers(1000, 10000, len(close))},
                        index=pd.bdate_range("2020-01-01", periods=len(close), name="Date"))


def test_cli_smoke_and_model_selection():
    args = cli.parser().parse_args(["--models", "wavelet-lstm-pso", "--smoke", "--device", "cpu"])
    config = cli.read_config(args)
    assert config["MODELS"] == ["Wavelet_LSTM_PSO"]
    assert config["RUN_WAVELET"] and config["SMOKE_TEST"]
    assert config["EPOCHS"] == 2 and config["PSO_PARTICLES"] == 2
    assert config["SEEDS"] == [42]


def test_cli_full_configuration_is_not_smoke():
    config = cli.read_config(cli.parser().parse_args(["--models", "all"]))
    assert set(config["MODELS"]) == set(cli.MODEL_NAMES.values())
    assert config["EPOCHS"] == 60 and config["PSO_ITERATIONS"] == 5
    assert len(config["SEEDS"]) == 3 and not config["SMOKE_TEST"]


@pytest.mark.parametrize("model", list(cli.MODEL_NAMES.values()))
def test_run_only_selected_model(model, tmp_path, monkeypatch):
    config = cli.read_config(cli.parser().parse_args(["--smoke"]))
    arrays = {f"X_{s}": np.ones((3, 60, 4), dtype=np.float32) for s in ["train", "val", "test"]}
    arrays.update({f"y_{s}": np.ones(3, dtype=np.float32) for s in ["train", "val", "test"]})
    meta = {"feature_columns": ["Close", "High", "Low", "Volume"], "lookback": 60}
    calls = []
    params = {"hidden_dim": 16, "num_layers": 2, "dropout": .2, "learning_rate": .001}
    def search(data, *args):
        assert set(data) == {"X_train", "y_train", "X_val", "y_val"}
        calls.append("pso")
        return {"params": params, "search_seconds": 1.0}
    def fit(data, *args, **kwargs):
        assert set(data) == {"X_train", "y_train", "X_val", "y_val"}
        calls.append("fit")
        return None, pd.DataFrame({"epoch": [1], "val_mse": [.1]}), {"train_seconds": .1, "best_val_mse": .1}
    monkeypatch.setattr(workflow, "pso_search", search)
    monkeypatch.setattr(workflow, "fit_lstm", fit)
    monkeypatch.setattr(workflow, "fit_anfis", fit)
    config["MODELS"] = [model]
    records, _ = workflow.run_experiments(arrays, meta, config, torch.device("cpu"), tmp_path)
    assert [r["method"] for r in records] == [model]
    assert calls == (["pso", "fit"] if model.endswith("PSO") else ["fit"])
    assert (tmp_path / "experiments.json").exists()


def test_local_preprocessing_preserves_past_context(raw, tmp_path):
    config = cli.read_config(cli.parser().parse_args(["--smoke"]))
    data, meta = prepare_data(raw, config, tmp_path)
    frame = add_features(raw)
    pd.testing.assert_frame_equal(add_features(raw.iloc[:130]), frame.loc[:raw.index[129]])
    end = meta["train_end_row"]
    np.testing.assert_allclose(meta["feature_scaler"]["data_max_"], frame.iloc[:end].max().to_numpy())
    assert data["previous_val"][0] == frame.Close.iloc[end - 1]
    changed = raw.copy()
    changed.loc[frame.index[end]:, ["Open", "High", "Low", "Close"]] *= 2
    other = tmp_path / "changed"
    other.mkdir()
    changed_data, _ = prepare_data(changed, config, other)
    np.testing.assert_array_equal(data["X_train"], changed_data["X_train"])
    np.testing.assert_array_equal(data["X_val"][0], changed_data["X_val"][0])
    wave = wavelet_inputs(data, meta, config)
    changed_wave = wavelet_inputs(changed_data, meta, config)
    np.testing.assert_array_equal(wave["X_val"][0], changed_wave["X_val"][0])
    np.testing.assert_array_equal(wave["y_test"], data["y_test"])


def test_portable_checkpoint_preserves_predictions(tmp_path):
    # Exercise the real independent notebook loader against a local export.
    import importlib.util
    path = Path(__file__).resolve().parents[1] / "notebooks/_kaggle_workflow.py"
    spec = importlib.util.spec_from_file_location("standalone_checkpoint_test", path)
    notebook = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(notebook)
    args = {"input_dim": 4, "hidden_dim": 16, "num_layers": 2, "dropout": .2}
    model = StockLSTM(**args).eval()
    checkpoint = tmp_path / "best.pt"
    torch.save({"format_version": 1, "model_args": args,
                "state_dict": export_weights(model.state_dict())}, checkpoint)
    inputs = torch.randn(3, 60, 4)
    expected = model(inputs).detach()
    torch.testing.assert_close(load_model(checkpoint, "cpu")(inputs), expected, rtol=0, atol=0)
    torch.testing.assert_close(notebook.load_model(checkpoint, "cpu")(inputs), expected, rtol=0, atol=0)
    # Earlier notebook/local runs use head.* keys directly.
    old_model = notebook.StockLSTM(**args).eval()
    torch.save({"format_version": 1, "model_args": args,
                "state_dict": old_model.state_dict()}, checkpoint)
    torch.testing.assert_close(load_model(checkpoint, "cpu")(inputs), old_model(inputs), rtol=0, atol=0)


def test_local_cli_without_notebooks(raw, tmp_path):
    """Run all five real models using only src, CLI, config and a CSV."""
    project = Path(__file__).resolve().parents[1]
    shutil.copytree(project / "src", tmp_path / "src", ignore=shutil.ignore_patterns("__pycache__"))
    shutil.copy2(project / "run_experiments.py", tmp_path)
    (tmp_path / "configs").mkdir()
    shutil.copy2(project / "configs/experiment.local.yaml", tmp_path / "configs")
    raw.to_csv(tmp_path / "prices.csv")
    env = dict(os.environ, PYTHONPATH=str(tmp_path), PYTHONUTF8="1")
    # Use runpy so even a future accidental notebook import is explicitly blocked.
    launcher = '''
import importlib.abc, runpy, sys
class NoNotebooks(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] == 'notebooks':
            raise RuntimeError('Local CLI must not import notebooks')
sys.meta_path.insert(0, NoNotebooks())
runpy.run_path('run_experiments.py', run_name='__main__')
'''
    def run(*args):
        result = subprocess.run([sys.executable, "-X", "utf8", "-c", launcher, *args],
                                cwd=tmp_path, env=env, capture_output=True, text=True,
                                encoding="utf-8", timeout=120)
        assert result.returncode == 0, result.stdout + result.stderr

    run("--models", "all", "--smoke", "--csv", "prices.csv", "--backtest")
    root = Path((tmp_path / "reports/experiments/latest_run.txt").read_text(encoding="utf-8"))
    records = json.loads((root / "experiments.json").read_text(encoding="utf-8"))
    assert {r["method"] for r in records} == set(cli.MODEL_NAMES.values())
    for record in records:
        expected = ANFIS if record["method"] == "ANFIS" else StockLSTM
        assert isinstance(load_model(root / record["checkpoint"], "cpu"), expected)
    before = pd.read_csv(root / "evaluation/metrics_by_seed.csv")
    assert set(before.method) == {"Naive", *cli.MODEL_NAMES.values()}
    assert (root / "evaluation/backtest.csv").exists()
    run("--evaluate-run", str(root), "--backtest")
    recheck = next((root / "evaluation").glob("recheck_*"))
    pd.testing.assert_frame_equal(before, pd.read_csv(recheck / "metrics_by_seed.csv"))
    pd.testing.assert_frame_equal(pd.read_csv(root / "evaluation/predictions.csv"),
                                  pd.read_csv(recheck / "predictions.csv"))
