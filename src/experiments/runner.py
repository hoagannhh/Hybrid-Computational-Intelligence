"""Compose registered LSTM/ANFIS variants and train requested experiments."""
import numpy as np
from src.features.window_wavelet import wavelet_inputs
from src.models.hybrid.registry import MODEL_SPECS
from src.models.optimizers.pso_search import pso_search
from src.training.lstm import fit_lstm
from src.training.anfis import fit_anfis
from src.utils.experiment import save_json


def run_experiments(data, meta, config, device, root, selected_models=None):
    allowed = set(MODEL_SPECS)
    if selected_models is None:
        selected_models = config["MODELS"]
    selected = set(selected_models)
    if not selected or not selected.issubset(allowed):
        raise ValueError(f"Chọn ít nhất một model trong {sorted(allowed)}.")
    variants = {"raw": data}
    if any(MODEL_SPECS[name].wavelet for name in selected):
        variants["wavelet"] = wavelet_inputs(data, meta, config)
        np.savez_compressed(root / "prepared_wavelet.npz", **variants["wavelet"])
    fixed = {"hidden_dim": config["HIDDEN_DIM"], "num_layers": config["NUM_LAYERS"],
             "dropout": config["DROPOUT"], "learning_rate": config["LEARNING_RATE"]}
    experiments = []
    # Hoàn thành lựa chọn tham số và train trước khi đánh giá test.
    for variant, arrays in variants.items():
        specs = [spec for spec in MODEL_SPECS.values()
                 if spec.label in selected and spec.wavelet == (variant == "wavelet")]
        if not specs:
            continue
        variant_root = root / variant
        variant_root.mkdir()
        train_val = {k: arrays[k] for k in ["X_train", "y_train", "X_val", "y_val"]}
        search = None
        if any(spec.pso for spec in specs):
            search = pso_search(train_val, config, device, variant_root)
        for spec in specs:
            params = search["params"] if spec.pso else fixed
            trainer = fit_lstm
            if spec.model_type == "ANFIS":
                params = {"num_rules": config.get("ANFIS_NUM_RULES", 5),
                          "learning_rate": config.get("ANFIS_LEARNING_RATE", 0.001)}
                trainer = fit_anfis
            search_seconds = search["search_seconds"] if spec.pso else 0.0
            label = spec.label
            for seed in config["SEEDS"]:
                run_id = f"{label}_seed{seed}"
                out = variant_root / run_id
                out.mkdir()
                print("Training:", run_id, params)
                model, history, result = trainer(train_val, params, config, device, seed,
                                                  config["EPOCHS"], config["PATIENCE"], out / "best.pt")
                history.to_csv(out / "history.csv", index=False)
                record = {"run_id": run_id, "method": label, "variant": variant, "seed": seed,
                          "model_type": spec.model_type,
                          "checkpoint": str((out / "best.pt").relative_to(root).as_posix()),
                          "params": params, "search_seconds_shared": search_seconds, **result}
                experiments.append(record)
                save_json(root / "experiments.json", experiments)
                del model
    return experiments, variants
