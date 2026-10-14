import os
import argparse
import numpy as np
import torch
from src.utils.helpers import load_config, get_device
from src.utils.logger import setup_logger
from src.data.data_fetcher import fetch_stock_data
from src.features.technical_indicators import calculate_technical_indicators
from src.data.preprocessor import TimeSeriesPreprocessor
from src.models.neural.lstm_model import StockLSTM
from src.evaluation.backtester import TradingBacktester

logger = setup_logger("backtest_main")

def main():
    parser = argparse.ArgumentParser(description="Chạy thử nghiệm giao dịch (Backtesting)")
    parser.add_argument("--config", type=str, default="configs/config.yaml", help="Đường dẫn file cấu hình")
    parser.add_argument("--model_path", type=str, default="saved_models/best_model.pt", help="Đường dẫn checkpoint")
    args = parser.parse_args()

    config = load_config(args.config)
    device = get_device(config["project"].get("device", "cpu"))

    logger.info("=== BẮT ĐẦU QUÁ TRÌNH BACKTEST CHIẾN LƯỢC GIAO DỊCH ===")

    # 1. Tải và tiền xử lý dữ liệu
    df_raw = fetch_stock_data(
        ticker=config["data"]["ticker"],
        start_date=config["data"]["start_date"],
        end_date=config["data"]["end_date"],
        source=config["data"].get("source", "yfinance")
    )
    df_features = calculate_technical_indicators(df_raw) if config["features"].get("use_technical_indicators", True) else df_raw.copy()

    preprocessor = TimeSeriesPreprocessor(
        seq_length=config["data"].get("seq_length", 60),
        horizon=config["data"].get("prediction_horizon", 1)
    )
    cleaned_df = preprocessor.clean_data(df_features)
    train_df, val_df, test_df = preprocessor.split_data(cleaned_df)

    data_pipeline = preprocessor.fit_transform_pipeline(train_df, val_df, test_df)
    X_test, y_test = data_pipeline["X_test"], data_pipeline["y_test"]
    target_scaler = data_pipeline["target_scaler"]

    # 2. Dự đoán
    model = StockLSTM(input_dim=X_test.shape[2], hidden_dim=config["models"]["neural"].get("hidden_dim", 64)).to(device)
    model.load_state_dict(torch.load(args.model_path, map_location=device))
    model.eval()

    with torch.no_grad():
        preds_scaled = model(torch.tensor(X_test, dtype=torch.float32).to(device)).cpu().numpy()

    actual_prices = target_scaler.inverse_transform(y_test.reshape(-1, 1)).flatten()
    predicted_prices = target_scaler.inverse_transform(preds_scaled.reshape(-1, 1)).flatten()

    # 3. Giả lập backtesting
    backtester = TradingBacktester(
        initial_capital=config["evaluation"]["backtest"].get("initial_capital", 100000.0),
        commission=config["evaluation"]["backtest"].get("commission_rate", 0.001)
    )

    results = backtester.run_backtest(actual_prices, predicted_prices)
    logger.info("Kết quả giao dịch giả lập:")
    logger.info(f" - Lợi nhuận chiến lược mô hình lai: {results['Total_Strategy_Return(%)']:.2f}%")
    logger.info(f" - Lợi nhuận cơ sở Buy & Hold: {results['Benchmark_BuyHold_Return(%)']:.2f}%")
    logger.info(f" - Tỷ số Sharpe (Annualized Sharpe Ratio): {results['Annualized_Sharpe_Ratio']:.2f}")
    logger.info(f" - Mức sụt giảm tối đa (Max Drawdown): {results['Max_Drawdown(%)']:.2f}%")

if __name__ == "__main__":
    main()
