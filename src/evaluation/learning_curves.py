"""Plot saved epoch histories without loading models, data or scalers."""
import json
import math
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def plot_loss_curves(root):
    root = Path(root).resolve()
    records = json.loads((root / "experiments.json").read_text(encoding="utf-8"))
    if not records:
        raise ValueError("Run chưa có model đã train để vẽ loss.")
    histories = []
    for record in records:
        path = (root / record["checkpoint"]).parent / "history.csv"
        if not path.resolve().is_relative_to(root):
            raise ValueError("Đường dẫn history phải nằm trong thư mục run.")
        history = pd.read_csv(path)
        columns = ["epoch", "train_mse", "val_mse"]
        if not set(columns).issubset(history.columns) or history.empty:
            raise ValueError(f"History cần các cột {columns} và ít nhất một epoch: {path}")
        values = history[columns].to_numpy(dtype=float)
        if (not np.isfinite(values).all() or (values[:, 1:] < 0).any()
                or (values[:, 0] < 1).any() or (np.diff(values[:, 0]) <= 0).any()):
            raise ValueError(f"History chứa loss/epoch không hợp lệ: {path}")
        best_epoch = record.get("best_epoch", history.loc[history.val_mse.idxmin(), "epoch"])
        if best_epoch not in history.epoch.to_numpy():
            raise ValueError(f"Không tìm thấy best_epoch={best_epoch} trong {path}")
        histories.append((record, history, best_epoch))

    output = root / "learning_curves"
    output.mkdir(exist_ok=True)
    files = []
    for record, history, best_epoch in histories:
        fig, ax = plt.subplots(figsize=(9, 5))
        ax.plot(history.epoch, history.train_mse, label="Train", color="#2563eb")
        ax.plot(history.epoch, history.val_mse, label="Validation", color="#e76f51")
        ax.axvline(best_epoch, color="#16a34a", linestyle="--", alpha=.8,
                   label=f"Best checkpoint: epoch {int(best_epoch)}")
        ax.set(title=record["run_id"], xlabel="Epoch", ylabel="MSE (scaled target)")
        ax.grid(alpha=.2)
        ax.legend()
        fig.tight_layout()
        filename = re.sub(r"[^A-Za-z0-9._-]", "_", record["run_id"]) + ".png"
        path = output / filename
        fig.savefig(path, dpi=160)
        plt.close(fig)
        files.append(path)

    methods = list(dict.fromkeys(record["method"] for record, _, _ in histories))
    columns = min(2, len(methods))
    rows = math.ceil(len(methods) / columns)
    fig, axes = plt.subplots(rows, columns, figsize=(7 * columns, 4 * rows), squeeze=False)
    for ax, method in zip(axes.flat, methods):
        for record, history, best_epoch in histories:
            if record["method"] != method:
                continue
            line, = ax.plot(history.epoch, history.val_mse, label=f"Seed {record['seed']}")
            best_loss = history.loc[history.epoch == best_epoch, "val_mse"].iloc[0]
            ax.scatter([best_epoch], [best_loss], color=line.get_color(), s=24, zorder=3)
        ax.set(title=method, xlabel="Epoch", ylabel="Validation MSE (scaled target)")
        ax.grid(alpha=.2)
        ax.legend()
    for ax in list(axes.flat)[len(methods):]:
        ax.set_visible(False)
    fig.suptitle("Validation loss by model and seed — dots mark saved best epochs")
    fig.tight_layout(rect=(0, 0, 1, .97))
    summary = output / "validation_curves.png"
    fig.savefig(summary, dpi=160)
    plt.close(fig)
    files.append(summary)
    print(f"Loss plots: {len(files)} files in {output}")
    return files
