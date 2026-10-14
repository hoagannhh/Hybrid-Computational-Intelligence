import pandas as pd
import numpy as np

def calculate_technical_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """
    Tính toán các chỉ báo phân tích kỹ thuật phổ biến từ chuỗi giá OHLCV.
    """
    data = df.copy()
    close = data["Close"]
    high = data["High"]
    low = data["Low"]
    volume = data["Volume"]

    # 1. Simple & Exponential Moving Averages (SMA & EMA)
    data["SMA_10"] = close.rolling(window=10).mean()
    data["SMA_20"] = close.rolling(window=20).mean()
    data["EMA_12"] = close.ewm(span=12, adjust=False).mean()
    data["EMA_26"] = close.ewm(span=26, adjust=False).mean()

    # 2. Relative Strength Index (RSI)
    delta = close.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / (loss + 1e-9)
    data["RSI_14"] = 100 - (100 / (1 + rs))

    # 3. MACD (Moving Average Convergence Divergence)
    data["MACD"] = data["EMA_12"] - data["EMA_26"]
    data["MACD_signal"] = data["MACD"].ewm(span=9, adjust=False).mean()
    data["MACD_hist"] = data["MACD"] - data["MACD_signal"]

    # 4. Bollinger Bands
    rolling_mean = close.rolling(window=20).mean()
    rolling_std = close.rolling(window=20).std()
    data["BB_upper"] = rolling_mean + (rolling_std * 2)
    data["BB_lower"] = rolling_mean - (rolling_std * 2)
    data["BB_width"] = (data["BB_upper"] - data["BB_lower"]) / (rolling_mean + 1e-9)

    # 5. Average True Range (ATR)
    tr1 = high - low
    tr2 = (high - close.shift()).abs()
    tr3 = (low - close.shift()).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    data["ATR_14"] = tr.rolling(window=14).mean()

    # 6. On-Balance Volume (OBV)
    data["OBV"] = (np.sign(close.diff()) * volume).fillna(0).cumsum()

    # Xử lý các giá trị NaN sinh ra do rolling window
    data.bfill(inplace=True)
    data.ffill(inplace=True)
    return data
