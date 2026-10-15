"""Technical features computed using only current and past observations."""
import numpy as np
import pandas as pd


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
