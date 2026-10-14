import numpy as np
import pandas as pd
from typing import Dict, Any

class TradingBacktester:
    """
    Giả lập chiến lược giao dịch dựa trên tín hiệu dự đoán từ mô hình lai.
    So sánh hiệu suất chiến lược với chiến lược cơ sở Mua & Nắm giữ (Buy and Hold).
    """
    def __init__(self, initial_capital: float = 100000.0, commission: float = 0.001):
        self.initial_capital = initial_capital
        self.commission = commission

    def run_backtest(self, actual_prices: np.ndarray, predicted_prices: np.ndarray) -> Dict[str, Any]:
        """
        Chiến lược cơ bản:
        - Mua (Long) nếu dự đoán ngày mai giá tăng (y_pred[t] > y_actual[t-1])
        - Đóng vị thế / Bán ra nếu dự đoán giá giảm
        """
        n = len(actual_prices)
        if n < 2:
            raise ValueError("Cần ít nhất 2 phiên giao dịch để backtest.")

        signals = np.zeros(n)
        # Sinh tín hiệu mua/bán
        for i in range(1, n):
            if predicted_prices[i] > actual_prices[i - 1]:
                signals[i] = 1 # Mua / Giữ
            else:
                signals[i] = 0 # Tiền mặt

        # Tính lợi nhuận theo từng ngày
        daily_returns = np.diff(actual_prices) / actual_prices[:-1]
        strategy_returns = signals[1:] * daily_returns - (np.abs(np.diff(signals)) * self.commission)

        # Lợi nhuận tích lũy
        cum_strategy = np.cumprod(1 + strategy_returns)
        cum_benchmark = np.cumprod(1 + daily_returns)

        total_strategy_return = (cum_strategy[-1] - 1.0) * 100.0
        total_benchmark_return = (cum_benchmark[-1] - 1.0) * 100.0

        # Sharpe Ratio (Giả sử 252 ngày giao dịch/năm, risk-free rate = 0%)
        mean_ret = np.mean(strategy_returns)
        std_ret = np.std(strategy_returns) + 1e-9
        annualized_sharpe = np.sqrt(252) * (mean_ret / std_ret)

        # Maximum Drawdown (MDD)
        peak = np.maximum.accumulate(cum_strategy)
        drawdown = (cum_strategy - peak) / peak
        max_drawdown = float(np.min(drawdown)) * 100.0

        return {
            "Total_Strategy_Return(%)": float(total_strategy_return),
            "Benchmark_BuyHold_Return(%)": float(total_benchmark_return),
            "Annualized_Sharpe_Ratio": float(annualized_sharpe),
            "Max_Drawdown(%)": float(max_drawdown),
            "Cumulative_Strategy": cum_strategy.tolist(),
            "Cumulative_Benchmark": cum_benchmark.tolist()
        }
