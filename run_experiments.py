"""Run local experiments implemented in src; no notebook dependency.

python run_experiments.py --models all --smoke
python run_experiments.py --models lstm lstm-pso
python run_experiments.py --evaluate-run reports/experiments/<run> --backtest
"""
import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import numpy as np
import torch
import yaml

from src.data.sources import load_raw
from src.data.preparation import prepare_data
from src.evaluation.experiment import evaluate_experiments, plot_results, backtest_predictions
from src.evaluation.learning_curves import plot_loss_curves
from src.experiments.runner import run_experiments
from src.models.hybrid.registry import MODEL_NAMES, MODEL_SPECS
from src.utils.experiment import new_run, select_device, resolve_run_directory

ROOT = Path(__file__).resolve().parent


def parser():
    p = argparse.ArgumentParser(description="Local LSTM / PSO / Wavelet / ANFIS experiments implemented in src.")
    p.add_argument("--config", type=Path, default=ROOT / "configs/experiment.local.yaml")
    p.add_argument("--models", nargs="+", choices=[*MODEL_NAMES, "all"], help="Default: models in YAML.")
    p.add_argument("--device", choices=["cpu", "cuda", "auto"])
    p.add_argument("--threads", type=int, help="CPU threads; default 1 for this small dataset.")
    p.add_argument("--smoke", action="store_true", help="2 epochs, 1 seed, small PSO search; not a final experiment.")
    p.add_argument("--epochs", type=int)
    p.add_argument("--seeds", nargs="+", type=int)
    p.add_argument("--particles", type=int)
    p.add_argument("--iterations", type=int)
    p.add_argument("--search-epochs", type=int)
    p.add_argument("--anfis-rules", type=int, help="Number of fuzzy rules (default 5).")
    p.add_argument("--anfis-lr", type=float, help="ANFIS learning rate (default 0.001).")
    p.add_argument("--csv", type=Path, help="Use a local CSV (schema still comes from YAML).")
    p.add_argument("--ticker")
    p.add_argument("--start")
    p.add_argument("--end")
    p.add_argument("--output-root", type=Path)
    p.add_argument("--backtest", action="store_true", help="Also export the illustrative close-to-close backtest.")
    saved_run = p.add_mutually_exclusive_group()
    saved_run.add_argument("--evaluate-run", type=Path, help="Run directory or latest_run.txt; evaluate without retraining or fetching data.")
    saved_run.add_argument("--plot-loss-run", type=Path, help="Run directory or latest_run.txt; plot saved histories without training.")
    return p


def read_config(args):
    with args.config.open(encoding="utf-8-sig") as stream:
        config = yaml.safe_load(stream)
    config.setdefault("ANFIS_NUM_RULES", 5)
    config.setdefault("ANFIS_LEARNING_RATE", 0.001)
    for arg, key in [("device", "DEVICE"), ("threads", "CPU_THREADS"), ("epochs", "EPOCHS"),
                     ("seeds", "SEEDS"), ("particles", "PSO_PARTICLES"), ("iterations", "PSO_ITERATIONS"),
                     ("search_epochs", "SEARCH_EPOCHS"), ("anfis_rules", "ANFIS_NUM_RULES"),
                     ("anfis_lr", "ANFIS_LEARNING_RATE"), ("ticker", "TICKER"), ("start", "START"), ("end", "END")]:
        value = getattr(args, arg)
        if value is not None:
            config[key] = value
    if args.csv:
        config.update(SOURCE="csv", CSV_PATH=str(args.csv))
    if args.output_root:
        config["OUTPUT_ROOT"] = str(args.output_root)
    if args.models:
        config["MODELS"] = list(MODEL_NAMES.values()) if "all" in args.models else list(dict.fromkeys(MODEL_NAMES[m] for m in args.models))
    if not config["MODELS"] or not set(config["MODELS"]).issubset(MODEL_SPECS):
        raise ValueError(f"MODELS must select from {list(MODEL_SPECS)}")
    config["RUN_WAVELET"] = any(MODEL_SPECS[m].wavelet for m in config["MODELS"])
    config["REQUIRE_GPU"] = config["DEVICE"] == "cuda"
    if not config["SEEDS"] or len(set(config["SEEDS"])) != len(config["SEEDS"]):
        raise ValueError("SEEDS must be nonempty and unique.")
    if args.smoke or config.get("SMOKE_TEST", False):
        config.update(SMOKE_TEST=True, SEEDS=[config["SEEDS"][0]], EPOCHS=2, PATIENCE=2,
                      PSO_PARTICLES=2, PSO_ITERATIONS=1, SEARCH_EPOCHS=1, SEARCH_PATIENCE=1)
    for key in ["CPU_THREADS", "EPOCHS", "PATIENCE", "PSO_PARTICLES", "SEARCH_EPOCHS", "SEARCH_PATIENCE",
                "BATCH_SIZE", "LOOKBACK", "HIDDEN_DIM", "NUM_LAYERS", "ANFIS_NUM_RULES"]:
        if not isinstance(config[key], int) or config[key] < 1:
            raise ValueError(f"{key} must be a positive integer.")
    if config["PSO_ITERATIONS"] < 0:
        raise ValueError("PSO_ITERATIONS must be nonnegative.")
    if not np.isfinite(config["ANFIS_LEARNING_RATE"]) or config["ANFIS_LEARNING_RATE"] <= 0:
        raise ValueError("ANFIS_LEARNING_RATE must be finite and positive.")
    return config


def choose_device(name):
    if name == "cpu":
        print("Device: cpu")
        return torch.device("cpu")
    if name not in {"cuda", "auto"}:
        raise ValueError("DEVICE must be cpu, cuda or auto.")
    return select_device(require_gpu=name == "cuda")


def evaluate_run(root, device, backtest=False, output=None):
    experiments = json.loads((root / "experiments.json").read_text(encoding="utf-8"))
    meta = json.loads((root / "preprocessing.json").read_text(encoding="utf-8"))
    config = json.loads((root / "config.json").read_text(encoding="utf-8"))
    variants = {}
    for variant in {"raw"} | {record["variant"] for record in experiments}:
        name = "prepared_wavelet.npz" if variant == "wavelet" else "prepared_data.npz"
        with np.load(root / name, allow_pickle=False) as archive:
            variants[variant] = {key: archive[key] for key in archive.files}
    if output is None:
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ")
        output = root / "evaluation" / f"recheck_{stamp}"
    metrics, predictions = evaluate_experiments(experiments, variants, meta, root, output, device)
    plot_results(metrics, predictions, output, show=False)
    if backtest:
        backtest_predictions(predictions, config["COMMISSION"], output, show=False)
    print("Evaluation:", output.resolve())
    return metrics


def main(argv=None):
    for stream in [sys.stdout, sys.stderr]:
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8")
    args = parser().parse_args(argv)
    if args.plot_loss_run:
        plot_loss_curves(resolve_run_directory(args.plot_loss_run))
        return
    if args.evaluate_run:
        root = resolve_run_directory(args.evaluate_run)
        config = json.loads((root / "config.json").read_text(encoding="utf-8"))
        threads = args.threads or config.get("CPU_THREADS", 1)
        if threads < 1:
            raise ValueError("--threads must be positive.")
        torch.set_num_threads(threads)
        evaluate_run(root, choose_device(args.device or "cpu"), args.backtest)
        return
    config = read_config(args)
    torch.set_num_threads(config["CPU_THREADS"])
    device = choose_device(config["DEVICE"])
    if config["SOURCE"] == "csv" and config.get("CSV_PATH") and not Path(config["CSV_PATH"]).is_file():
        raise FileNotFoundError(f"CSV not found: {config['CSV_PATH']}. Pass --csv PATH or update the YAML.")
    print("Models:", config["MODELS"], "Seeds:", config["SEEDS"])
    print("SMOKE TEST" if config["SMOKE_TEST"] else "FULL CONFIGURED EXPERIMENT", "— max epochs:", config["EPOCHS"])
    root = new_run(config, "local_smoke" if config["SMOKE_TEST"] else "local_experiment")
    raw = load_raw(config, root)
    arrays, meta = prepare_data(raw, config, root)
    run_experiments(arrays, meta, config, device, root, selected_models=config["MODELS"])
    plot_loss_curves(root)
    evaluate_run(root, device, args.backtest, output=root / "evaluation")
    (Path(config["OUTPUT_ROOT"]) / "latest_run.txt").write_text(str(root.resolve()), encoding="utf-8")
    print("Run complete:", root.resolve())
    print("Use this whole run directory with notebook 05, or --evaluate-run.")


if __name__ == "__main__":
    main()
