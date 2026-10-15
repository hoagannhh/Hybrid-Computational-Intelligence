"""Numerical stability, actual learning and portable ANFIS checkpoint tests."""
import importlib.util
from pathlib import Path

import numpy as np
import pytest
import torch

import run_experiments as cli
from src.models.fuzzy.anfis import ANFIS
from src.models.checkpoints import load_model
from src.training.anfis import fit_anfis


@pytest.fixture
def notebook():
    path = Path(__file__).resolve().parents[1] / "notebooks/_kaggle_workflow.py"
    spec = importlib.util.spec_from_file_location("independent_anfis_tests", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_anfis_matches_normalized_bell_rules():
    torch.manual_seed(9)
    model = ANFIS(4, num_rules=3).double()
    x = torch.rand(8, 4, dtype=torch.float64)
    # Independent direct formula on a moderate domain where no underflow occurs.
    width = torch.log1p(torch.exp(model.mf.raw_width)) + 1e-4
    slope = torch.log1p(torch.exp(model.mf.raw_slope)) + .5
    distance = ((x.unsqueeze(-1) - model.mf.centers) / width).abs().clamp_min(1e-6)
    memberships = 1 / (1 + distance ** (2 * slope))
    firing = memberships.prod(dim=1)
    weights = firing / firing.sum(dim=1, keepdim=True)
    expected = (weights * (x @ model.linear_weights.t() + model.biases)).sum(dim=1)
    torch.testing.assert_close(model(x), expected)


@pytest.mark.parametrize("value", [0., 0.5, 1e6, -1e6])
def test_anfis_finite_forward_and_gradients(value):
    torch.set_num_threads(1)
    torch.manual_seed(12)
    model = ANFIS(64, 7)
    x = torch.full((4, 64), value)
    output = model(x)
    assert torch.isfinite(output).all()
    output.square().mean().backward()
    for parameter in model.parameters():
        assert parameter.grad is not None
        assert torch.isfinite(parameter.grad).all()
    log_firing = model.mf.log_membership(x).sum(dim=1)
    torch.testing.assert_close(torch.softmax(log_firing, dim=1).sum(dim=1), torch.ones(4))


def test_anfis_window_uses_only_last_observation():
    model = ANFIS(4, 3)
    window = torch.randn(6, 20, 4)
    before = model(window)
    torch.testing.assert_close(before, model(window[:, -1]), rtol=0, atol=0)
    window[:, :-1] = torch.randn_like(window[:, :-1]) * 100
    torch.testing.assert_close(model(window), before, rtol=0, atol=0)
    with pytest.raises(ValueError, match="expects"):
        model(torch.ones(6, 5))


@pytest.mark.parametrize("backend", ["local", "notebook"])
def test_anfis_learns_and_checkpoint_roundtrip(backend, notebook, tmp_path):
    torch.set_num_threads(1)
    rng = np.random.default_rng(3)
    X = rng.uniform(0, 1, (160, 6, 4)).astype(np.float32)
    y = (.1 + .4 * X[:, -1, 0] + .2 * X[:, -1, 1]).astype(np.float32)
    data = {"X_train": X[:120], "y_train": y[:120], "X_val": X[120:], "y_val": y[120:],
            "X_test": np.full_like(X[:3], np.nan), "y_test": np.full(3, np.nan)}
    params = {"num_rules": 4, "learning_rate": .02}
    trainer = fit_anfis if backend == "local" else notebook.fit_anfis
    checkpoint = tmp_path / "anfis.pt"
    model, history, result = trainer(data, params, {"BATCH_SIZE": 24}, torch.device("cpu"),
                                     42, 30, 30, checkpoint, quiet=True)
    assert result["best_val_mse"] < history.iloc[0].val_mse * .1
    assert result["best_val_mse"] < .001
    assert np.isfinite(history[["train_mse", "val_mse"]].to_numpy()).all()
    model.eval()
    batch = torch.from_numpy(data["X_val"])
    expected = model(batch)
    assert float(((expected - torch.from_numpy(data["y_val"])) ** 2).mean().detach()) == pytest.approx(result["best_val_mse"])
    for loader in [load_model, notebook.load_model]:
        restored = loader(checkpoint, "cpu")
        assert restored.num_rules == 4
        torch.testing.assert_close(restored(batch), expected, rtol=0, atol=0)
    saved = torch.load(checkpoint, weights_only=True)
    assert saved["format_version"] == 2 and saved["model_type"] == "ANFIS"


def test_anfis_cli_configuration():
    config = cli.read_config(cli.parser().parse_args([
        "--models", "anfis", "--anfis-rules", "7", "--anfis-lr", ".005", "--smoke"]))
    assert config["MODELS"] == ["ANFIS"] and not config["RUN_WAVELET"]
    assert config["ANFIS_NUM_RULES"] == 7 and config["ANFIS_LEARNING_RATE"] == .005


@pytest.mark.parametrize("args", [["--anfis-rules", "0"], ["--anfis-lr", "0"], ["--anfis-lr", "nan"]])
def test_invalid_anfis_configuration(args):
    with pytest.raises(ValueError, match="ANFIS_"):
        cli.read_config(cli.parser().parse_args(["--models", "anfis", *args]))


def test_notebook_anfis_only_does_not_train_lstm(notebook, tmp_path, monkeypatch):
    config = cli.read_config(cli.parser().parse_args(["--models", "anfis", "--smoke"]))
    calls = []

    def forbidden(*args, **kwargs):
        pytest.fail("ANFIS-only run must not invoke LSTM, Wavelet or PSO")

    def fit(data, params, *args, **kwargs):
        import pandas as pd
        assert set(data) == {"X_train", "y_train", "X_val", "y_val"}
        assert "num_rules" in params and "hidden_dim" not in params
        calls.append("ANFIS")
        return None, pd.DataFrame({"epoch": [1]}), {"train_seconds": 0.}

    for name in ["fit_lstm", "pso_search", "wavelet_inputs"]:
        monkeypatch.setattr(notebook, name, forbidden)
    monkeypatch.setattr(notebook, "fit_anfis", fit)
    arrays = {key: np.ones(3) for key in ["X_train", "y_train", "X_val", "y_val", "X_test"]}
    records, variants = notebook.run_experiments(arrays, {}, config, "cpu", tmp_path, selected_models=["ANFIS"])
    assert calls == ["ANFIS"] and list(variants) == ["raw"]
    assert records[0]["method"] == "ANFIS"
