"""Evaluate saved experiments, plot predictions and run illustrative backtests."""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from src.data.preparation import inverse_target
from src.models.checkpoints import load_model
from src.training.supervised import predict_scaled


def regression_metrics(actual, predicted, previous):
    actual, predicted, previous = map(lambda x: np.asarray(x, dtype=float), [actual, predicted, previous])
    if not (actual.shape == predicted.shape == previous.shape) or actual.ndim != 1:
        raise ValueError("actual, predicted, previous phải là vector cùng shape.")
    if not all(np.isfinite(x).all() for x in [actual, predicted, previous]):
        raise ValueError("Dự đoán chứa NaN/Inf.")
    error = actual - predicted
    denom = np.sum((actual - actual.mean()) ** 2)
    return {"RMSE": float(np.sqrt(np.mean(error ** 2))), "MAE": float(np.mean(np.abs(error))),
            "MAPE_percent": float(np.mean(np.abs(error / actual)) * 100),
            "R2": float(1 - np.sum(error ** 2) / denom) if denom > 0 else None,
            "DA_percent": float(np.mean(np.sign(actual - previous) == np.sign(predicted - previous)) * 100)}


def evaluate_experiments(experiments, variants, meta, source_root, output, device):
    output.mkdir(parents=True, exist_ok=True)
    rows, predictions = [], []
    raw = variants["raw"]
    naive = raw["previous_test"]
    rows.append({"run_id": "Naive", "method": "Naive", "seed": -1,
                 **regression_metrics(raw["actual_test"], naive, naive)})
    predictions.append(pd.DataFrame({"date": raw["dates_test"], "actual": raw["actual_test"],
                                    "previous": naive, "predicted": naive, "run_id": "Naive"}))
    for record in experiments:
        arrays = variants[record["variant"]]
        model = load_model(source_root / record["checkpoint"], device)
        pred = inverse_target(predict_scaled(model, arrays["X_test"], device), meta)
        rows.append({"run_id": record["run_id"], "method": record["method"], "seed": record["seed"],
                     **regression_metrics(arrays["actual_test"], pred, arrays["previous_test"]),
                     "train_seconds": record["train_seconds"],
                     "search_seconds_shared": record["search_seconds_shared"]})
        predictions.append(pd.DataFrame({"date": arrays["dates_test"], "actual": arrays["actual_test"],
                                         "previous": arrays["previous_test"], "predicted": pred,
                                         "run_id": record["run_id"]}))
        del model
    metrics = pd.DataFrame(rows)
    pred_frame = pd.concat(predictions, ignore_index=True)
    metrics.to_csv(output / "metrics_by_seed.csv", index=False)
    pred_frame.to_csv(output / "predictions.csv", index=False)
    columns = ["RMSE", "MAE", "MAPE_percent", "R2", "DA_percent"]
    summary = metrics.groupby("method", sort=False)[columns].agg(["mean", "std"])
    summary.to_csv(output / "metrics_summary.csv")
    print(metrics.to_string(index=False))
    print("\nMean/std qua seeds (std trống nếu chỉ có một lần chạy):\n", summary)
    return metrics, pred_frame


def plot_results(metrics, predictions, output, show=True):
    fig, axes = plt.subplots(2, 1, figsize=(13, 9))
    first = predictions[predictions.run_id == "Naive"]
    axes[0].plot(pd.to_datetime(first.date), first.actual, label="Actual", color="black")
    # Một seed cố định đầu tiên cho mỗi method; không chọn seed theo điểm test.
    for method, group in metrics[metrics.method != "Naive"].groupby("method", sort=False):
        run_id = group.iloc[0].run_id
        subset = predictions[predictions.run_id == run_id]
        axes[0].plot(pd.to_datetime(subset.date), subset.predicted, label=run_id, alpha=.8)
    axes[0].set_title("Giá thực tế và dự đoán trên test — seed đầu tiên")
    axes[0].legend()
    means = metrics.groupby("method", sort=False).RMSE.mean()
    means.plot.bar(ax=axes[1], rot=15, title="Test RMSE trung bình qua seeds")
    fig.tight_layout()
    fig.savefig(output / "comparison.png", dpi=160)
    if show:
        plt.show()
    plt.close(fig)


def trading_metrics(actual, previous, predicted, fee, buy_hold=False):
    position = np.ones(len(actual)) if buy_hold else (predicted > previous).astype(float)
    turnover = np.abs(np.diff(np.r_[0, position]))
    returns = position * (actual / previous - 1) - fee * turnover
    returns[-1] -= fee * position[-1]  # Thanh lý vị thế cuối kỳ.
    equity = np.r_[1.0, np.cumprod(1 + returns)]
    drawdown = equity / np.maximum.accumulate(equity) - 1
    std = returns.std()
    return {"Return_percent": float((equity[-1] - 1) * 100),
            "Sharpe": float(np.sqrt(252) * returns.mean() / std) if std > 0 else 0.0,
            "Max_drawdown_percent": float(drawdown.min() * 100),
            "Turnover_units": float(turnover.sum() + position[-1])}, equity


def backtest_predictions(predictions, fee, output, show=True):
    rows = []
    fig, ax = plt.subplots(figsize=(12, 5))
    for run_id, group in predictions.groupby("run_id", sort=False):
        stats, equity = trading_metrics(group.actual.to_numpy(), group.previous.to_numpy(),
                                        group.predicted.to_numpy(), fee)
        rows.append({"run_id": run_id, **stats})
        ax.plot(equity, label=run_id, alpha=.7)
    first = predictions[predictions.run_id == "Naive"]
    stats, equity = trading_metrics(first.actual.to_numpy(), first.previous.to_numpy(),
                                    first.predicted.to_numpy(), fee, buy_hold=True)
    rows.append({"run_id": "Buy_and_hold", **stats})
    ax.plot(equity, label="Buy_and_hold", color="black", linewidth=2)
    ax.set(title="Backtest minh họa close-to-close, có phí", xlabel="Phiên test", ylabel="Vốn / vốn ban đầu")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(output / "backtest.png", dpi=160)
    if show:
        plt.show()
    plt.close(fig)
    table = pd.DataFrame(rows)
    table.to_csv(output / "backtest.csv", index=False)
    print(table.to_string(index=False))
    return table
