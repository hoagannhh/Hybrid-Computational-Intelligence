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
from src.evaluation.metrics import calculate_metrics
from src.utils.visualizer import plot_predictions

logger = setup_logger("evaluate_main")

def main():
    parser = argparse.ArgumentParser(description="Đánh giá mô hình trên tập Test")
    parser.add_argument("--config", type=str, default="configs/config.yaml", help="Đường dẫn file cấu hình")
    parser.add_argument("--model_path", type=str, default="saved_models/best_model.pt", help="Đường dẫn checkpoint")
    args = parser.parse_args()

    config = load_config(args.config)
    device = get_device(config["project"].get("device", "cpu"))

    logger.info("=== BẮT ĐẦU ĐÁNH GIÁ MÔ HÌNH TRÊN TẬP KIỂM THỬ (TEST SET) ===")

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

    # 2. Khởi tạo mô hình và tải trọng số
    input_dim = X_test.shape[2]
    model = StockLSTM(
        input_dim=input_dim,
        hidden_dim=config["models"]["neural"].get("hidden_dim", 64),
        dropout=config["models"]["neural"].get("dropout", 0.2)
    ).to(device)

    if not os.path.exists(args.model_path):
        raise FileNotFoundError(f"Không tìm thấy file checkpoint: {args.model_path}")

    model.load_state_dict(torch.load(args.model_path, map_location=device))
    model.eval()

    # 3. Dự đoán trên tập test
    with torch.no_grad():
        X_test_tensor = torch.tensor(X_test, dtype=torch.float32).to(device)
        y_pred_scaled = model(X_test_tensor).cpu().numpy()

    # Khôi phục về thang giá trị thực ban đầu (Inverse transform)
    y_test_real = target_scaler.inverse_transform(y_test.reshape(-1, 1)).flatten()
    y_pred_real = target_scaler.inverse_transform(y_pred_scaled.reshape(-1, 1)).flatten()

    # 4. Tính toán các chỉ số đánh giá
    metrics = calculate_metrics(y_test_real, y_pred_real)
    logger.info("Kết quả đánh giá trên tập kiểm thử:")
    for k, v in metrics.items():
        logger.info(f" - {k}: {v:.4f}")

    # 5. Lưu biểu đồ so sánh
    plot_predictions(
        actual=y_test_real,
        predicted=y_pred_real,
        title=f"Dự đoán giá cổ phiếu {config['data']['ticker']} (Tập Test)",
        save_path="reports/figures/test_prediction_comparison.png"
    )
    logger.info("Đã lưu biểu đồ kết quả tại reports/figures/test_prediction_comparison.png")

if __name__ == "__main__":
    main()
