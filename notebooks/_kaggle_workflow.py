"""Source embedded into the standalone notebooks by build_notebooks.py.

Upload the generated .ipynb only; this file is not needed on Kaggle.
"""

# %% imports
import hashlib
import importlib.metadata
import json
import random
import re
import time
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urlsplit

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import MinMaxScaler
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


def save_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False), encoding="utf-8")


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def select_device(require_gpu=False):
    if require_gpu and not torch.cuda.is_available():
        raise RuntimeError("Chưa có GPU. Bật Accelerator GPU trên Kaggle rồi chạy lại; hoặc đặt REQUIRE_GPU=False để thử CPU.")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device, torch.cuda.get_device_name(0) if device.type == "cuda" else "CPU")
    return device


def new_run(config, label):
    base = Path(config["OUTPUT_ROOT"])
    base.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ")
    symbol = re.sub(r"[^A-Za-z0-9._-]", "_", config['TICKER'].replace('^', ''))
    root = base / f"{symbol}_{label}_{stamp}"
    root.mkdir(exist_ok=False)
    save_json(root / "config.json", config)
    versions = {}
    for name in ["numpy", "pandas", "scikit-learn", "torch", "matplotlib", "yfinance", "PyWavelets", "kagglehub"]:
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = None
    save_json(root / "environment.json", {"packages": versions, "cuda_runtime": torch.version.cuda,
              "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None})
    print("Output:", root)
    return root


# %% data
OHLCV = ["Open", "High", "Low", "Close", "Volume"]
REQUIRED_PRICES = ["High", "Low", "Close", "Volume"]


def kaggle_handle(value):
    value = value.strip()
    if "://" in value:
        url = urlsplit(value)
        if url.scheme != "https" or url.hostname not in {"www.kaggle.com", "kaggle.com"}:
            raise ValueError("KAGGLE_DATASET cần URL https://www.kaggle.com/... hoặc owner/dataset.")
        value = url.path.strip("/")
        if value.startswith("datasets/"):
            value = value[len("datasets/"):]
    if not re.fullmatch(r"[\w-]+/[\w-]+(?:/versions/[1-9][0-9]*)?", value):
        raise ValueError("KAGGLE_DATASET cần owner/dataset hoặc owner/dataset/versions/N.")
    return value


def csv_frame(path, config):
    # Read as strings to preserve ticker codes such as 000001 and currency formatting.
    table = pd.read_csv(path, dtype=str, sep=config.get("CSV_SEP", ","),
                        encoding=config.get("CSV_ENCODING", "utf-8"))
    table.columns = table.columns.str.strip()
    mapping = config.get("COLUMN_MAP", {})
    missing = set(mapping) - set(table.columns)
    if missing:
        raise ValueError(f"COLUMN_MAP có cột không tồn tại: {sorted(missing)}; CSV có: {list(table.columns)}")
    table = table.rename(columns=mapping)
    # Standard names are case insensitive. Other names require an explicit mapping.
    aliases = {c.casefold(): c for c in OHLCV + ["Date", "Ticker"]}
    aliases.update({"symbol": "Ticker", "datetime": "Date", "timestamp": "Date"})
    table = table.rename(columns={c: aliases.get(c.casefold(), c) for c in table.columns})
    if table.columns.duplicated().any():
        raise ValueError("Tên cột trùng sau mapping; chỉ chọn một nguồn cho mỗi cột chuẩn.")
    ticker_col = config.get("TICKER_COLUMN") or ("Ticker" if "Ticker" in table else None)
    if ticker_col:
        ticker_col = aliases.get(mapping.get(ticker_col, ticker_col).casefold(), mapping.get(ticker_col, ticker_col))
        if ticker_col not in table:
            raise ValueError(f"Không có cột ticker {ticker_col}.")
        available = table[ticker_col].dropna().str.strip().unique()
        table = table.loc[table[ticker_col].str.strip().eq(config["TICKER"])].copy()
        if table.empty:
            raise ValueError(f"Không có ticker {config['TICKER']}. Ví dụ mã hiện có: {available[:15].tolist()}")
    date_col = config.get("DATE_COLUMN", "Date")
    date_col = aliases.get(mapping.get(date_col, date_col).casefold(), mapping.get(date_col, date_col))
    if date_col not in table:
        raise ValueError(f"Không có cột ngày {date_col}. Dùng COLUMN_MAP hoặc DATE_COLUMN; CSV có {list(table.columns)}")
    if not set(REQUIRED_PRICES).issubset(table):
        raise ValueError(f"Cần tối thiểu {REQUIRED_PRICES}; Open là tùy chọn. Dùng COLUMN_MAP để đổi tên cột.")
    dates = pd.to_datetime(table[date_col].str.strip(), format=config.get("DATE_FORMAT"), errors="raise", utc=True)
    columns = [c for c in OHLCV if c in table]
    frame = table[columns].copy()
    for column in columns:
        values = frame[column].str.strip()
        for token in config.get("CSV_STRIP_TOKENS", ["$"]):
            values = values.str.replace(token, "", regex=False)
        thousands, decimal = config.get("CSV_THOUSANDS", ","), config.get("CSV_DECIMAL", ".")
        if thousands and thousands == decimal:
            raise ValueError("CSV_THOUSANDS và CSV_DECIMAL phải khác nhau.")
        if thousands:
            values = values.str.replace(thousands, "", regex=False)
        if decimal != ".":
            values = values.str.replace(decimal, ".", regex=False)
        frame[column] = pd.to_numeric(values.str.strip(), errors="raise")
    frame.index = pd.DatetimeIndex(dates).tz_convert(None)
    if frame.index.isna().any() or not np.isfinite(frame.to_numpy()).all():
        raise ValueError("Dữ liệu đã chọn chứa ngày thiếu hoặc giá/volume NaN/Inf.")
    # Daily snapshots: never sum repeated daily volume or invent a missing Open.
    frame = frame.sort_index(kind="stable")
    if frame.index.duplicated().any():
        raise ValueError("Timestamp trùng sau lọc ticker; cần xử lý nguồn dữ liệu trước.")
    policy = config.get("DAILY_POLICY", "strict")
    days = frame.index.normalize()
    extra_rows = int(days.duplicated().sum())
    if policy not in {"strict", "last_snapshot"}:
        raise ValueError("DAILY_POLICY phải là strict hoặc last_snapshot.")
    if extra_rows and policy == "strict":
        raise ValueError("Có nhiều timestamp trong một ngày. Chỉ dùng last_snapshot nếu các dòng là snapshot dữ liệu ngày; dữ liệu intraday cần tổng hợp OHLCV riêng.")
    frame.index = days
    if extra_rows:
        frame = frame.loc[~frame.index.duplicated(keep="last")].copy()
        print(f"Daily snapshots: bỏ {extra_rows} dòng lặp ngày, giữ snapshot cuối; không cộng volume.")
    info = {"column_map": mapping, "ticker_column": ticker_col, "date_column": date_col,
            "daily_policy": policy, "discarded_same_day_snapshots": extra_rows,
            "selected_columns": columns, "missing_open": "Open" not in frame}
    return frame, info


def validate_raw(frame, config):
    df = frame.copy()
    if isinstance(df.columns, pd.MultiIndex):
        if len(df.columns.get_level_values(-1).unique()) != 1:
            raise ValueError("Chỉ hỗ trợ một ticker trong một lần chạy.")
        df.columns = df.columns.get_level_values(0)
    if not set(REQUIRED_PRICES).issubset(df.columns):
        raise ValueError(f"CSV cần các cột {REQUIRED_PRICES}; Open là tùy chọn.")
    df.index = pd.to_datetime(df.index, errors="raise")
    if df.index.tz is not None:
        df.index = df.index.tz_localize(None)
    if df.index.isna().any() or df.index.duplicated().any():
        raise ValueError("Ngày bị thiếu hoặc trùng; kiểm tra dữ liệu gốc.")
    columns = [c for c in OHLCV if c in df]
    df = df.sort_index().loc[:, columns].apply(pd.to_numeric, errors="raise")
    if config.get("START"):
        df = df.loc[df.index >= config["START"]]
    if config.get("END"):
        df = df.loc[df.index < config["END"]]
    df = df.copy()
    if df.empty or not np.isfinite(df.to_numpy()).all():
        raise ValueError("Dữ liệu rỗng hoặc chứa NaN/Inf. Kiểm tra ngày, ticker và CSV.")
    price_cols = [c for c in columns if c != "Volume"]
    if (df[price_cols] <= 0).any().any() or (df.Volume < 0).any():
        raise ValueError("Giá phải dương, volume không âm.")
    tolerance = 1e-6 * df.Close
    if ((df.High + tolerance < df[price_cols].max(axis=1)) |
            (df.Low - tolerance > df[price_cols].min(axis=1))).any():
        raise ValueError("Quan hệ OHLC không hợp lệ.")
    df.index.name = "Date"
    return df


def load_raw(config, root):
    symbol = re.sub(r"[^A-Za-z0-9._-]", "_", config['TICKER'].replace('^', ''))
    filename = f"{symbol}_{config.get('START') or 'start'}_{config.get('END') or 'end'}.csv"
    if config["SOURCE"] in {"csv", "kaggle"}:
        if config["SOURCE"] == "kaggle":
            import kagglehub
            handle = kaggle_handle(config.get("KAGGLE_DATASET", ""))
            file = config.get("KAGGLE_FILE", "")
            if not file or file.startswith(("/", "\\")) or ".." in Path(file).parts:
                raise ValueError("Đặt KAGGLE_FILE là đường dẫn tương đối của một CSV trong dataset.")
            path = Path(kagglehub.dataset_download(handle, path=file))
        elif config.get("CSV_PATH"):
            if "://" in config["CSV_PATH"]:
                raise ValueError("CSV_PATH cần đường dẫn file; link Kaggle dùng SOURCE='kaggle' và KAGGLE_DATASET.")
            path = Path(config["CSV_PATH"])
        else:
            lookup = config.get("KAGGLE_FILE") or filename
            matches = list(Path("/kaggle/input").rglob(lookup))
            if len(matches) != 1:
                raise ValueError(f"Tìm thấy {len(matches)} file {filename}. Hãy đặt CSV_PATH chính xác trong config.")
            path = matches[0]
        frame, csv_info = csv_frame(path, config)
        source = {"source": config["SOURCE"], "path": str(path), "price_basis": config["CSV_PRICE_BASIS"],
                  "kaggle_dataset": config.get("KAGGLE_DATASET", ""), "dataset_note": config.get("DATASET_NOTE", ""), **csv_info}
    elif config["SOURCE"] == "yfinance":
        import yfinance as yf
        frame = None
        failure = None
        for attempt in range(3):
            try:
                frame = yf.download(config["TICKER"], start=config["START"], end=config["END"],
                                    auto_adjust=True, progress=False, threads=False, timeout=30)
                if frame is not None and not frame.empty:
                    break
            except Exception as exc:
                failure = str(exc)
            if attempt < 2:
                time.sleep(2 * (attempt + 1))
        if frame is None or frame.empty:
            raise RuntimeError(f"Yahoo download thất bại ({failure}). Bật Internet hoặc Add Input CSV rồi đặt SOURCE='csv'.")
        source = {"source": "yfinance", "auto_adjust": True, "price_basis": "adjusted"}
    else:
        raise ValueError("SOURCE phải là 'yfinance', 'csv' hoặc 'kaggle'.")
    df = validate_raw(frame, config)
    raw_file = root / filename
    df.to_csv(raw_file)
    source.update(ticker=config["TICKER"], rows=len(df), first_date=str(df.index[0].date()),
                  last_date=str(df.index[-1].date()), sha256=hashlib.sha256(raw_file.read_bytes()).hexdigest())
    save_json(root / "data_source.json", source)
    print("RAW:", source)
    return df


# %% features
def add_features(raw):
    # Tất cả phép tính chỉ dùng giá hiện tại/quá khứ; không backward fill.
    df = raw.copy()
    close = df.Close
    df["SMA_10"] = close.rolling(10).mean()
    df["SMA_20"] = close.rolling(20).mean()
    df["EMA_12"] = close.ewm(span=12, adjust=False).mean()
    df["EMA_26"] = close.ewm(span=26, adjust=False).mean()
    delta = close.diff()
    gain = delta.clip(lower=0).rolling(14).mean()
    loss = (-delta.clip(upper=0)).rolling(14).mean()
    df["RSI_14"] = 100 - 100 / (1 + gain / (loss + 1e-9))
    df["MACD"] = df.EMA_12 - df.EMA_26
    df["MACD_signal"] = df.MACD.ewm(span=9, adjust=False).mean()
    df["MACD_hist"] = df.MACD - df.MACD_signal
    std = close.rolling(20).std()
    df["BB_upper"] = df.SMA_20 + 2 * std
    df["BB_lower"] = df.SMA_20 - 2 * std
    df["BB_width"] = (df.BB_upper - df.BB_lower) / df.SMA_20
    tr = pd.concat([df.High - df.Low, (df.High - close.shift()).abs(),
                    (df.Low - close.shift()).abs()], axis=1).max(axis=1)
    df["ATR_14"] = tr.rolling(14).mean()
    df["OBV"] = (np.sign(close.diff()).fillna(0) * df.Volume).cumsum()
    df = df.dropna().copy()
    if not np.isfinite(df.to_numpy()).all():
        raise ValueError("Đặc trưng chứa NaN/Inf.")
    return df


def scaler_state(scaler):
    return {name: getattr(scaler, name).tolist() for name in
            ["scale_", "min_", "data_min_", "data_max_", "data_range_"]}


def inverse_target(values, meta):
    s = meta["target_scaler"]
    return (np.asarray(values).reshape(-1) - s["min_"][0]) / s["scale_"][0]


def prepare_data(raw, config, root):
    if config["HORIZON"] != 1:
        raise ValueError("Notebook này chỉ hỗ trợ dự báo một phiên tiếp theo (HORIZON=1).")
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


def wavelet_inputs(data, meta, config):
    import pywt
    result = dict(data)
    close_index = meta["feature_columns"].index("Close")
    name, level = config["WAVELET"], config["WAVELET_LEVEL"]
    maximum = pywt.dwt_max_level(meta["lookback"], pywt.Wavelet(name).dec_len)
    if not 1 <= level <= maximum:
        raise ValueError(f"WAVELET_LEVEL cần trong [1, {maximum}] cho lookback hiện tại.")
    for split in ["train", "val", "test"]:
        windows = data[f"X_{split}"].copy()
        for window in windows:
            series = window[:, close_index]
            coeffs = pywt.wavedec(series, name, level=level, mode="symmetric")
            sigma = np.median(np.abs(coeffs[-1] - np.median(coeffs[-1]))) / 0.6745
            threshold = sigma * np.sqrt(2 * np.log(len(series)))
            if threshold > 0:
                coeffs[1:] = [pywt.threshold(c, threshold, mode="soft") for c in coeffs[1:]]
            window[:, close_index] = pywt.waverec(coeffs, name, mode="symmetric")[:len(series)]
        assert np.isfinite(windows).all()
        result[f"X_{split}"] = windows
    # Target thật và các đặc trưng khác giữ nguyên; không lọc toàn chuỗi trước khi split.
    return result


# %% model
class StockLSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim=64, num_layers=2, dropout=0.2):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers=num_layers, batch_first=True,
                            dropout=dropout if num_layers > 1 else 0.0)
        self.head = nn.Sequential(nn.Linear(hidden_dim, 32), nn.ReLU(), nn.Dropout(dropout), nn.Linear(32, 1))

    def forward(self, x):
        outputs, _ = self.lstm(x)
        return self.head(outputs[:, -1]).squeeze(-1)


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


# %% training
def fit_lstm(data, params, config, device, seed, epochs, patience, checkpoint=None, quiet=False):
    seed_everything(seed)
    model_args = {k: params[k] for k in ["hidden_dim", "num_layers", "dropout"]}
    model_args["input_dim"] = data["X_train"].shape[-1]
    model = StockLSTM(**model_args).to(device)
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
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
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
                torch.save({"format_version": 1, "model_args": model_args, "params": params,
                            "seed": seed, "best_epoch": epoch, "val_mse": best,
                            "state_dict": best_weights}, checkpoint)
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


def decode_particle(position, config):
    return {"hidden_dim": int(np.clip(round(position[0] / 16) * 16, 16, 128)),
            "dropout": float(np.clip(position[1], 0, 0.5)),
            "learning_rate": float(10 ** np.clip(position[2], -4, -2)),
            "num_layers": config["NUM_LAYERS"]}


def pso_search(data, config, device, output):
    # Chỉ truyền train/val vào tìm kiếm. Test không tham gia fitness hoặc early stopping.
    rng = np.random.default_rng(config["SEARCH_SEED"])
    count = config["PSO_PARTICLES"]
    low, high = np.array([16, 0, -4.0]), np.array([128, .5, -2.0])
    particles = rng.uniform(low, high, (count, 3))
    velocity = np.zeros_like(particles)
    personal, scores = particles.copy(), np.full(count, np.inf)
    global_best, global_score = None, float("inf")
    cache, records = {}, []
    start = time.perf_counter()
    for iteration in range(config["PSO_ITERATIONS"] + 1):
        if iteration:
            velocity = (.5 * velocity + 1.5 * rng.random(particles.shape) * (personal - particles)
                        + 1.5 * rng.random(particles.shape) * (global_best - particles))
            particles = np.clip(particles + velocity, low, high)
        for i, particle in enumerate(particles):
            params = decode_particle(particle, config)
            key = json.dumps(params, sort_keys=True)
            cached = key in cache
            if not cached:
                model, _, result = fit_lstm(data, params, config, device, config["SEARCH_SEED"],
                                            config["SEARCH_EPOCHS"], config["SEARCH_PATIENCE"], quiet=True)
                cache[key] = result["best_val_mse"]
                del model
            score = cache[key]
            if score < scores[i]:
                scores[i], personal[i] = score, particle.copy()
            if score < global_score:
                global_score, global_best = score, particle.copy()
            records.append({"iteration": iteration, "particle": i, **params, "val_mse": score,
                            "best_val_mse": global_score, "cached": cached})
        pd.DataFrame(records).to_csv(output / "pso_history.csv", index=False)
        print(f"PSO {iteration}/{config['PSO_ITERATIONS']}: best val MSE={global_score:.6f}")
    result = {"params": decode_particle(global_best, config), "val_mse": global_score,
              "search_seconds": time.perf_counter() - start, "unique_candidates": len(cache)}
    save_json(output / "pso_best.json", result)
    return result


def run_experiments(data, meta, config, device, root, include_pso=True):
    variants = {"raw": data}
    if config["RUN_WAVELET"]:
        variants["wavelet"] = wavelet_inputs(data, meta, config)
        np.savez_compressed(root / "prepared_wavelet.npz", **variants["wavelet"])
    fixed = {"hidden_dim": config["HIDDEN_DIM"], "num_layers": config["NUM_LAYERS"],
             "dropout": config["DROPOUT"], "learning_rate": config["LEARNING_RATE"]}
    experiments = []
    # Hoàn thành lựa chọn tham số và train trước khi đánh giá test.
    for variant, arrays in variants.items():
        variant_root = root / variant
        variant_root.mkdir()
        train_val = {k: arrays[k] for k in ["X_train", "y_train", "X_val", "y_val"]}
        choices = [("LSTM", fixed, 0.0)]
        if include_pso:
            search = pso_search(train_val, config, device, variant_root)
            choices.append(("LSTM_PSO", search["params"], search["search_seconds"]))
        for method, params, search_seconds in choices:
            label = ("Wavelet_" if variant == "wavelet" else "") + method
            for seed in config["SEEDS"]:
                run_id = f"{label}_seed{seed}"
                out = variant_root / run_id
                out.mkdir()
                print("Training:", run_id, params)
                model, history, result = fit_lstm(train_val, params, config, device, seed,
                                                  config["EPOCHS"], config["PATIENCE"], out / "best.pt")
                history.to_csv(out / "history.csv", index=False)
                record = {"run_id": run_id, "method": label, "variant": variant, "seed": seed,
                          "checkpoint": str((out / "best.pt").relative_to(root).as_posix()),
                          "params": params, "search_seconds_shared": search_seconds, **result}
                experiments.append(record)
                save_json(root / "experiments.json", experiments)
                del model
    return experiments, variants


# %% evaluation
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


def load_model(path, device):
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    model = StockLSTM(**checkpoint["model_args"]).to(device)
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    return model


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


def plot_results(metrics, predictions, output):
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


def backtest_predictions(predictions, fee, output):
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
    plt.show()
    plt.close(fig)
    table = pd.DataFrame(rows)
    table.to_csv(output / "backtest.csv", index=False)
    print(table.to_string(index=False))
    return table
