"""Shared supervised training for local LSTM and ANFIS experiments."""
import time
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from src.models.checkpoints import build_model, export_weights
from src.utils.experiment import seed_everything


def make_loader(X, y, batch_size, shuffle, seed=42, cuda=False):
    generator = torch.Generator().manual_seed(seed)
    return DataLoader(TensorDataset(torch.from_numpy(X), torch.from_numpy(y)),
                      batch_size=batch_size, shuffle=shuffle, num_workers=0,
                      pin_memory=cuda, generator=generator)


def predict_scaled(model, X, device, batch_size=256):
    model.eval()
    values = []
    with torch.no_grad():
        for start in range(0, len(X), batch_size):
            batch = torch.from_numpy(X[start:start+batch_size]).to(device)
            values.append(model(batch).cpu().numpy())
    return np.concatenate(values)


def fit_model(data, params, config, device, seed, epochs, patience, checkpoint=None, quiet=False, model_type="LSTM"):
    seed_everything(seed)
    keys = ["num_rules"] if model_type == "ANFIS" else ["hidden_dim", "num_layers", "dropout"]
    model_args = {k: params[k] for k in keys}
    model_args["input_dim"] = data["X_train"].shape[-1]
    model = build_model(model_type, model_args).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=params["learning_rate"])
    train = make_loader(data["X_train"], data["y_train"], config["BATCH_SIZE"], True, seed, device.type == "cuda")
    val = make_loader(data["X_val"], data["y_val"], config["BATCH_SIZE"], False, seed, device.type == "cuda")
    best, stale, best_weights, best_epoch, history = float("inf"), 0, None, None, []
    started = time.perf_counter()
    for epoch in range(1, epochs + 1):
        model.train()
        total = 0.0
        for X, y in train:
            X, y = X.to(device, non_blocking=True), y.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            loss = nn.functional.mse_loss(model(X), y)
            if not torch.isfinite(loss):
                raise FloatingPointError("Train loss NaN/Inf.")
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0, error_if_nonfinite=True)
            optimizer.step()
            total += loss.item() * len(y)
        model.eval()
        val_sum = 0.0
        with torch.no_grad():
            for X, y in val:
                X, y = X.to(device), y.to(device)
                val_sum += nn.functional.mse_loss(model(X), y, reduction="sum").item()
        val_loss = val_sum / len(val.dataset)
        if not np.isfinite(val_loss):
            raise FloatingPointError("Validation loss NaN/Inf.")
        history.append({"epoch": epoch, "train_mse": total / len(train.dataset), "val_mse": val_loss})
        if val_loss < best - 1e-9:
            best, stale, best_epoch = val_loss, 0, epoch
            best_weights = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            if checkpoint is not None:
                torch.save({"format_version": 2 if model_type == "ANFIS" else 1,
                            "model_type": model_type, "model_args": model_args, "params": params,
                            "seed": seed, "best_epoch": epoch, "val_mse": best,
                            "state_dict": export_weights(best_weights)}, checkpoint)
        else:
            stale += 1
        if not quiet and (epoch == 1 or epoch % 10 == 0 or epoch == epochs):
            print(f"epoch={epoch:03d} train={history[-1]['train_mse']:.6f} val={val_loss:.6f}")
        if stale >= patience:
            break
    if best_weights is None:
        raise RuntimeError("Không tạo được checkpoint hợp lệ.")
    model.load_state_dict(best_weights)
    elapsed = time.perf_counter() - started
    return model, pd.DataFrame(history), {"best_val_mse": best, "best_epoch": best_epoch, "train_seconds": elapsed}
