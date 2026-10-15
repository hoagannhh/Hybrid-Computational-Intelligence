"""Validated daily stock data from local CSV, Yahoo Finance or Kaggle."""
import hashlib
import re
import time
from pathlib import Path
from urllib.parse import urlsplit
import numpy as np
import pandas as pd
from src.utils.experiment import save_json


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
            raise ValueError("SOURCE=csv cần CSV_PATH. Dùng --csv PATH hoặc đặt đường dẫn trong YAML.")
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
            raise RuntimeError(f"Yahoo download thất bại ({failure}). Kiểm tra Internet hoặc dùng SOURCE='csv' và CSV_PATH.")
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
