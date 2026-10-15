"""Chronological splits, train-only scaling and past-context windows."""
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from src.features.causal import add_features
from src.utils.experiment import save_json


def scaler_state(scaler):
    return {name: getattr(scaler, name).tolist() for name in
            ["scale_", "min_", "data_min_", "data_max_", "data_range_"]}


def inverse_target(values, meta):
    s = meta["target_scaler"]
    return (np.asarray(values).reshape(-1) - s["min_"][0]) / s["scale_"][0]


def prepare_data(raw, config, root):
    if config["HORIZON"] != 1:
        raise ValueError("Pipeline local chỉ hỗ trợ dự báo một phiên tiếp theo (HORIZON=1).")
    ratios = [config["TRAIN_RATIO"], config["VAL_RATIO"], config["TEST_RATIO"]]
    if any(r <= 0 for r in ratios) or not np.isclose(sum(ratios), 1):
        raise ValueError("Tỷ lệ chia tập phải dương và tổng bằng 1.")
    frame = add_features(raw)
    n = len(frame)
    train_end, val_end = int(n * ratios[0]), int(n * sum(ratios[:2]))
    lookback = config["LOOKBACK"]
    if not (0 < lookback < train_end < val_end < n):
        raise ValueError("Không đủ dữ liệu cho lookback và ba tập; tăng khoảng ngày hoặc giảm LOOKBACK.")
    fs, ts = MinMaxScaler(), MinMaxScaler()
    fs.fit(frame.iloc[:train_end])
    ts.fit(frame.iloc[:train_end][["Close"]])
    features = fs.transform(frame).astype(np.float32)
    target = ts.transform(frame[["Close"]]).ravel().astype(np.float32)
    target_rows = np.arange(lookback, n)
    X = np.stack([features[i-lookback:i] for i in target_rows])
    y = target[target_rows]
    masks = {"train": target_rows < train_end,
             "val": (target_rows >= train_end) & (target_rows < val_end),
             "test": target_rows >= val_end}
    data, split_rows = {}, []
    for split, mask in masks.items():
        rows = target_rows[mask]
        data[f"X_{split}"] = X[mask]
        data[f"y_{split}"] = y[mask]
        data[f"actual_{split}"] = frame.Close.to_numpy()[rows]
        data[f"previous_{split}"] = frame.Close.to_numpy()[rows-1]
        data[f"dates_{split}"] = frame.index[rows].strftime("%Y-%m-%d").to_numpy(dtype=str)
        split_rows.append({"split": split, "samples": len(rows),
                           "first_target": data[f"dates_{split}"][0],
                           "last_target": data[f"dates_{split}"][-1]})
    meta = {"feature_columns": list(frame.columns), "input_dim": len(frame.columns),
            "lookback": lookback, "horizon": 1, "target": "Close",
            "feature_scaler": scaler_state(fs), "target_scaler": scaler_state(ts),
            "train_end_row": train_end, "val_end_row": val_end,
            "dropped_warmup_rows": len(raw)-len(frame), "splits": split_rows}
    assert data["dates_train"][-1] < data["dates_val"][0] < data["dates_test"][0]
    for key, value in data.items():
        if not key.startswith("dates"):
            assert np.isfinite(value).all(), key
    np.savez_compressed(root / "prepared_data.npz", **data)
    save_json(root / "preprocessing.json", meta)
    frame.to_csv(root / "features.csv")
    pd.DataFrame(split_rows).to_csv(root / "splits.csv", index=False)
    print(pd.DataFrame(split_rows).to_string(index=False))
    print("Input shape:", data["X_train"].shape, "— scaler fit trên train; val/test giữ context quá khứ.")
    return data, meta
