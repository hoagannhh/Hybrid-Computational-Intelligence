import pytest
import numpy as np
import pandas as pd
import torch
from src.features.technical_indicators import calculate_technical_indicators
from src.data.preprocessor import TimeSeriesPreprocessor
from src.models.neural.lstm_model import StockLSTM
from src.evaluation.metrics import calculate_metrics

def test_technical_indicators():
    dates = pd.date_range("2023-01-01", periods=50)
    data = {
        "Open": np.random.uniform(100, 110, 50),
        "High": np.random.uniform(110, 120, 50),
        "Low": np.random.uniform(90, 100, 50),
        "Close": np.random.uniform(100, 110, 50),
        "Volume": np.random.uniform(1000, 5000, 50)
    }
    df = pd.DataFrame(data, index=dates)
    res = calculate_technical_indicators(df)
    assert "RSI_14" in res.columns
    assert "MACD" in res.columns
    assert "BB_upper" in res.columns

def test_lstm_forward():
    batch_size = 8
    seq_len = 30
    input_dim = 10
    model = StockLSTM(input_dim=input_dim, hidden_dim=32)
    x = torch.randn(batch_size, seq_len, input_dim)
    out = model(x)
    assert out.shape == (batch_size,)

def test_metrics():
    y_true = np.array([10.0, 12.0, 15.0, 14.0])
    y_pred = np.array([10.5, 11.5, 14.8, 14.2])
    m = calculate_metrics(y_true, y_pred)
    assert "RMSE" in m
    assert "MAE" in m
    assert "Directional_Accuracy(%)" in m
