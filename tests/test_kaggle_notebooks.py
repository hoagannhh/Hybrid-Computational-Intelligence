"""Execute generated notebook cells offline, including checkpoint round trips."""
import importlib.util
import json
import sys
from types import SimpleNamespace
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
NOTEBOOKS = ROOT / "notebooks"
spec = importlib.util.spec_from_file_location("kaggle_workflow", NOTEBOOKS / "_kaggle_workflow.py")
workflow = importlib.util.module_from_spec(spec)
spec.loader.exec_module(workflow)


@pytest.fixture
def raw():
    rng = np.random.default_rng(9)
    close = 100 + np.cumsum(rng.normal(.05, 1, 320))
    return pd.DataFrame({"Open": close-.2, "High": close+1, "Low": close-1,
                         "Close": close, "Volume": rng.integers(1000, 10000, len(close))},
                        index=pd.bdate_range("2020-01-01", periods=len(close), name="Date"))


def test_features_use_only_past(raw):
    full = workflow.add_features(raw)
    prefix = workflow.add_features(raw.iloc[:150])
    pd.testing.assert_frame_equal(prefix, full.loc[prefix.index])
    assert full.index[0] == raw.index[19]


def test_split_scaling_and_window_alignment(raw, tmp_path):
    config = {"TRAIN_RATIO": .7, "VAL_RATIO": .15, "TEST_RATIO": .15, "LOOKBACK": 60, "HORIZON": 1}
    data, meta = workflow.prepare_data(raw, config, tmp_path)
    frame = workflow.add_features(raw)
    end = meta["train_end_row"]
    column = meta["feature_columns"].index("Close")
    np.testing.assert_allclose(meta["feature_scaler"]["data_max_"], frame.iloc[:end].max().to_numpy())
    np.testing.assert_allclose(data["previous_val"][0], frame.Close.iloc[end-1])
    assert data["dates_val"][0] == frame.index[end].strftime("%Y-%m-%d")
    assert len(data["y_val"]) == meta["val_end_row"] - end
    np.testing.assert_allclose(workflow.inverse_target(data["y_test"], meta), data["actual_test"], rtol=1e-6)
    changed = raw.copy()
    changed.loc[frame.index[end]:, ["Open", "High", "Low", "Close"]] *= 2
    other = tmp_path / "changed"
    other.mkdir()
    changed_data, _ = workflow.prepare_data(changed, config, other)
    np.testing.assert_array_equal(data["X_train"], changed_data["X_train"])
    np.testing.assert_array_equal(data["X_val"][0], changed_data["X_val"][0])
    wave_config = {"WAVELET": "db4", "WAVELET_LEVEL": 3}
    wave = workflow.wavelet_inputs(data, meta, wave_config)
    changed_wave = workflow.wavelet_inputs(changed_data, meta, wave_config)
    np.testing.assert_array_equal(wave["X_val"][0], changed_wave["X_val"][0])
    np.testing.assert_array_equal(wave["y_test"], data["y_test"])
    assert np.isfinite(wave["X_train"][:, :, column]).all()


def test_backtest_includes_initial_loss():
    stats, equity = workflow.trading_metrics(np.array([90., 81.]), np.array([100., 90.]),
                                            np.array([110., 100.]), 0)
    assert equity[0] == 1
    assert stats["Max_drawdown_percent"] == pytest.approx(-19.)


def test_apple_csv_currency_and_reverse_dates(tmp_path):
    path = tmp_path / "HistoricalQuotes.csv"
    path.write_text("Date, Close/Last, Volume, Open, High, Low\n02/28/2020, $273.36, 106721200, $257.26, $278.41, $256.37\n02/27/2020, $273.52, 80151380, $281.1, $286, $272.96\n", encoding="utf-8")
    config = {"COLUMN_MAP": {"Close/Last": "Close"}, "DATE_FORMAT": "%m/%d/%Y", "START": None, "END": None}
    frame, info = workflow.csv_frame(path, config)
    frame = workflow.validate_raw(frame, config)
    assert len(frame) == 2 and frame.index.is_monotonic_increasing
    assert frame.Close.iloc[-1] == pytest.approx(273.36)
    assert frame.Volume.iloc[-1] == 106721200
    assert info["missing_open"] is False


def test_filter_ticker_before_date_validation(raw, tmp_path):
    a = raw.reset_index()
    a["Name"] = "000001"
    b = a.copy()
    b["Name"] = "000002"
    b["Date"] = "invalid-date-in-other-ticker"
    path = tmp_path / "stocks.csv"
    pd.concat([a, b]).to_csv(path, index=False)
    config = {"TICKER": "000001", "TICKER_COLUMN": "Name"}
    frame, _ = workflow.csv_frame(path, config)
    np.testing.assert_allclose(frame.Close.to_numpy(), raw.Close.to_numpy(), rtol=1e-12)
    assert len(frame) == len(raw)
    with pytest.raises(ValueError, match="Không có ticker"):
        workflow.csv_frame(path, dict(config, TICKER="AAPL"))


def test_tech_daily_policy_no_fabricated_open(tmp_path):
    path = tmp_path / "AAPL.csv"
    path.write_text("Date,stock_close,stock_high,stock_low,stock_volume,stock_ma_20\n2020-01-02 05:00:00+00:00,100,105,90,1000,999\n2020-01-02 23:00:00+00:00,101,106,91,1100,999\n2020-01-03 05:00:00+00:00,102,106,91,1200,999\n", encoding="utf-8")
    config = {"COLUMN_MAP": {"stock_close": "Close", "stock_high": "High", "stock_low": "Low", "stock_volume": "Volume"},
              "DATE_FORMAT": "%Y-%m-%d %H:%M:%S%z"}
    with pytest.raises(ValueError, match="nhiều timestamp"):
        workflow.csv_frame(path, config)
    frame, info = workflow.csv_frame(path, dict(config, DAILY_POLICY="last_snapshot"))
    frame = workflow.validate_raw(frame, {})
    assert len(frame) == 2 and "Open" not in frame and "stock_ma_20" not in frame
    assert frame.Volume.iloc[0] == 1100  # Do not sum snapshots of daily volume.
    assert frame.Close.iloc[0] == 101 and info["discarded_same_day_snapshots"] == 1


def test_missing_open_pipeline(raw, tmp_path):
    config = {"TRAIN_RATIO": .7, "VAL_RATIO": .15, "TEST_RATIO": .15, "LOOKBACK": 60, "HORIZON": 1}
    arrays, meta = workflow.prepare_data(workflow.validate_raw(raw.drop(columns="Open"), {}), config, tmp_path)
    assert meta["input_dim"] == 17
    model = workflow.StockLSTM(input_dim=17, hidden_dim=16)
    assert model(torch.from_numpy(arrays["X_train"][:2])).shape == (2,)


@pytest.mark.parametrize("value,expected", [
    ("https://www.kaggle.com/tarunpaparaju/apple-aapl-historical-stock-data?utm_source=chatgpt.com", "tarunpaparaju/apple-aapl-historical-stock-data"),
    ("https://www.kaggle.com/datasets/sinankr/tech-stocks-dataset", "sinankr/tech-stocks-dataset"),
    ("sinankr/tech-stocks-dataset/versions/1", "sinankr/tech-stocks-dataset/versions/1"),
])
def test_kaggle_handles(value, expected):
    assert workflow.kaggle_handle(value) == expected


def test_kaggle_single_file_download_contract(raw, tmp_path, monkeypatch):
    path = tmp_path / "AAPL.csv"
    raw.to_csv(path)
    calls = []
    def download(handle, **kwargs):
        calls.append((handle, kwargs))
        return str(path)
    monkeypatch.setitem(sys.modules, "kagglehub", SimpleNamespace(dataset_download=download))
    config = {"SOURCE": "kaggle", "TICKER": "AAPL", "KAGGLE_DATASET": "https://www.kaggle.com/datasets/example/stocks",
              "KAGGLE_FILE": "AAPL.csv", "CSV_PRICE_BASIS": "unknown", "START": None, "END": None}
    actual = workflow.load_raw(config, tmp_path)
    assert len(actual) == len(raw)
    assert calls == [("example/stocks", {"path": "AAPL.csv"})]


def test_all_notebooks_execute_and_reload(raw, tmp_path, monkeypatch):
    # Executing notebook cells must never require project modules, even when
    # those modules are installed/cached in the test process.
    import builtins
    original_import = builtins.__import__

    def isolated_import(name, *args, **kwargs):
        if name.split(".")[0] in {"src", "notebooks", "run_experiments"}:
            raise AssertionError(f"Standalone notebook imported project code: {name}")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", isolated_import)
    torch.set_num_threads(1)
    monkeypatch.setattr(plt, "show", lambda: None)
    csv = tmp_path / "prices.csv"
    raw.to_csv(csv)
    comparison_root = None
    original_metrics = None
    for path in sorted(NOTEBOOKS.glob("0[1-5]_*.ipynb")):
        notebook = json.loads(path.read_text(encoding="utf-8"))
        assert notebook["nbformat"] == 4
        namespace = {"display": lambda *args, **kwargs: None}
        for i, cell in enumerate(notebook["cells"]):
            if cell["cell_type"] != "code":
                continue
            assert cell["execution_count"] is None and cell["outputs"] == []
            compiled = compile("".join(cell["source"]), f"{path.name}:cell{i}", "exec")
            exec(compiled, namespace)
            # Notebook 01 was saved by an editor without its config tag.
            if "config" in cell["metadata"].get("tags", []) or (path.name.startswith("01_") and i == 2):
                if path.name.startswith("05"):
                    namespace["RUN_PATH"] = str(comparison_root)
                    namespace["EVAL_OUTPUT_ROOT"] = str(tmp_path / "evaluation")
                else:
                    namespace["CONFIG"].update(
                        SOURCE="csv", CSV_PATH=str(csv), START="2020-01-01", END="2022-01-01",
                        OUTPUT_ROOT=str(tmp_path / path.stem), REQUIRE_GPU=False, SMOKE_TEST=True,
                        SEEDS=[42], EPOCHS=2, PATIENCE=2, HIDDEN_DIM=16, PSO_PARTICLES=2,
                        PSO_ITERATIONS=1, SEARCH_EPOCHS=1, SEARCH_PATIENCE=1,
                        RUN_WAVELET=path.name.startswith(("02", "04")))
        if path.name.startswith("04"):
            comparison_root = namespace["RUN_DIR"]
            original_metrics = namespace["metrics"]
            assert len(namespace["experiments"]) == 5
            for record in namespace["experiments"]:
                assert (comparison_root / record["checkpoint"]).is_file()
            assert set(original_metrics.method) == {"Naive", "LSTM", "LSTM_PSO", "Wavelet_LSTM", "Wavelet_LSTM_PSO", "ANFIS"}
        if path.name.startswith("03"):
            assert [r["method"] for r in namespace["experiments"]] == ["LSTM"]
        if path.name.startswith("05"):
            pd.testing.assert_frame_equal(original_metrics, namespace["metrics"])
            assert (namespace["OUT"] / "backtest.csv").is_file()


@pytest.mark.parametrize("models,extras", [
    (["LSTM", "LSTM_PSO", "Wavelet_LSTM", "Wavelet_LSTM_PSO", "ANFIS"], True),
    (["ANFIS"], False),
])
def test_complete_notebook_standalone(raw, tmp_path, monkeypatch, models, extras):
    import builtins
    import zipfile
    original_import = builtins.__import__

    def isolated_import(name, *args, **kwargs):
        if name.split(".")[0] in {"src", "notebooks", "run_experiments"}:
            raise AssertionError(f"Notebook imported project code: {name}")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", isolated_import)
    monkeypatch.setattr(plt, "show", lambda: None)
    torch.set_num_threads(1)
    csv = tmp_path / "prices.csv"
    raw.to_csv(csv)
    namespace = {"display": lambda *args, **kwargs: None}

    def execute(path, configure):
        for i, cell in enumerate(json.loads(path.read_text(encoding="utf-8"))["cells"]):
            if cell["cell_type"] != "code":
                continue
            assert cell["execution_count"] is None and cell["outputs"] == []
            exec(compile("".join(cell["source"]), f"{path.name}:cell{i}", "exec"), namespace)
            if "config" in cell["metadata"].get("tags", []):
                configure()

    def configure_complete():
        namespace["CONFIG"].update(
            SOURCE="csv", CSV_PATH=str(csv), START=None, END=None,
            OUTPUT_ROOT=str(tmp_path / "runs"), REQUIRE_GPU=False, SMOKE_TEST=True,
            MODELS=models, RUN_BACKTEST=extras, CREATE_ZIP=extras)

    execute(NOTEBOOKS / "06_complete_kaggle_pipeline.ipynb", configure_complete)
    root = namespace["RUN_DIR"]
    records = namespace["experiments"]
    assert len(records) == len(models)
    assert {r["method"] for r in records} == set(models)
    metrics = namespace["metrics"].copy()
    assert set(metrics.method) == set(models) | {"Naive"}
    assert np.isfinite(metrics[["RMSE", "MAE", "MAPE_percent", "R2", "DA_percent"]]).all().all()
    assert len(list((tmp_path / "runs").glob("*/experiments.json"))) == 1
    for record in records:
        checkpoint = root / record["checkpoint"]
        assert checkpoint.is_file()
        assert len(pd.read_csv(checkpoint.parent / "history.csv")) == 2
        assert (root / "learning_curves" / f"{record['run_id']}.png").is_file()
    assert (root / "learning_curves" / "validation_curves.png").is_file()
    quality = json.loads((root / "eda" / "data_quality.json").read_text())
    assert quality["eda_scope"] == "train_only"
    assert pd.Timestamp(quality["eda_last_date"]) == pd.Timestamp(str(namespace["data"]["dates_train"][-1]))
    assert len(namespace["eda_train"]) < len(raw)
    assert (root / "evaluation" / "backtest.csv").exists() == extras
    assert (root / "prepared_wavelet.npz").exists() == extras
    if extras:
        with zipfile.ZipFile(namespace["ARCHIVE_PATH"]) as archive:
            entries = set(archive.namelist())
            assert f"{root.name}/experiments.json" in entries
            assert f"{root.name}/evaluation/backtest.csv" in entries
            for record in records:
                assert f"{root.name}/{record['checkpoint']}" in entries
            restored = tmp_path / "unpacked"
            archive.extractall(restored)
        reload_root = restored / root.name
    else:
        assert namespace["ARCHIVE_PATH"] is None
        reload_root = root

    # A fresh session can evaluate the run (including after unzipping) using 05.
    namespace = {"display": lambda *args, **kwargs: None}
    def configure_reload():
        namespace.update(RUN_PATH=str(reload_root), EVAL_OUTPUT_ROOT=str(tmp_path / "recheck"))
    execute(NOTEBOOKS / "05_backtesting_and_ablation_study.ipynb", configure_reload)
    pd.testing.assert_frame_equal(metrics, namespace["metrics"])
