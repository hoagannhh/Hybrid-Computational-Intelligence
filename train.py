import os
import argparse
import torch
import torch.nn as nn
from src.utils.helpers import set_seed, load_config, get_device
from src.utils.logger import setup_logger
from src.data.data_fetcher import fetch_stock_data
from src.features.technical_indicators import calculate_technical_indicators
from src.data.preprocessor import TimeSeriesPreprocessor
from src.data.data_loader import get_data_loaders
from src.models.hybrid.wavelet_lstm_pso import WaveletLSTMHybrid
from src.training.trainer import ModelTrainer
from src.utils.visualizer import plot_training_loss

logger = setup_logger("train_main")

def main():
    parser = argparse.ArgumentParser(description="Huấn luyện mô hình Hybrid Computational Intelligence")
    parser.add_argument("--config", type=str, default="configs/config.yaml", help="Đường dẫn file cấu hình")
    parser.add_argument("--optimize_pso", action="store_true", help="Kích hoạt giải thuật PSO tìm siêu tham số")
    args = parser.parse_args()

    config = load_config(args.config)
    set_seed(config["project"].get("seed", 42))
    device = get_device(config["project"].get("device", "cpu"))

    logger.info("=== BẮT ĐẦU PIPELINE HUẤN LUYỆN HYBRID MODEL ===")
    
    # 1. Thu thập dữ liệu
    df_raw = fetch_stock_data(
        ticker=config["data"]["ticker"],
        start_date=config["data"]["start_date"],
        end_date=config["data"]["end_date"],
        source=config["data"].get("source", "yfinance")
    )

    # 2. Trích xuất đặc trưng kỹ thuật
    if config["features"].get("use_technical_indicators", True):
        logger.info("Đang tính toán các chỉ báo phân tích kỹ thuật...")
        df_features = calculate_technical_indicators(df_raw)
    else:
        df_features = df_raw.copy()

    # 3. Tiền xử lý & Tạo chuỗi thời gian
    preprocessor = TimeSeriesPreprocessor(
        seq_length=config["data"].get("seq_length", 60),
        horizon=config["data"].get("prediction_horizon", 1)
    )
    cleaned_df = preprocessor.clean_data(df_features)
    train_df, val_df, test_df = preprocessor.split_data(
        cleaned_df,
        train_ratio=config["data"].get("train_ratio", 0.7),
        val_ratio=config["data"].get("val_ratio", 0.15),
        test_ratio=config["data"].get("test_ratio", 0.15)
    )

    data_pipeline = preprocessor.fit_transform_pipeline(
        train_df, val_df, test_df,
        target_col=config["data"].get("target_col", "Close")
    )

    X_train, y_train = data_pipeline["X_train"], data_pipeline["y_train"]
    X_val, y_val = data_pipeline["X_val"], data_pipeline["y_val"]
    input_dim = X_train.shape[2]

    # 4. Khởi tạo mô hình Lai (Wavelet - LSTM - PSO)
    hybrid_system = WaveletLSTMHybrid(config)
    
    hidden_dim = config["models"]["neural"].get("hidden_dim", 64)
    dropout = config["models"]["neural"].get("dropout", 0.2)
    lr = config["models"]["neural"].get("learning_rate", 0.001)

    if args.optimize_pso:
        logger.info("Kích hoạt tối ưu bầy đàn (PSO) để tinh chỉnh siêu tham số...")
        best_params = hybrid_system.optimize_hyperparameters(X_train, y_train, X_val, y_val, input_dim=input_dim)
        hidden_dim = best_params["hidden_dim"]
        dropout = best_params["dropout"]
        lr = best_params["learning_rate"]

    model = hybrid_system.build_model(input_dim=input_dim, hidden_dim=hidden_dim, dropout=dropout)

    # 5. Huấn luyện
    train_loader, val_loader, _ = get_data_loaders(
        X_train, y_train, X_val, y_val, data_pipeline["X_test"], data_pipeline["y_test"],
        batch_size=config["models"]["neural"].get("batch_size", 32)
    )

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()
    trainer = ModelTrainer(model=model, optimizer=optimizer, criterion=criterion, device=device)

    history = trainer.fit(
        train_loader=train_loader,
        val_loader=val_loader,
        epochs=config["models"]["neural"].get("epochs", 50),
        patience=config["models"]["neural"].get("early_stopping_patience", 10)
    )

    plot_training_loss(history)
    logger.info("=== HOÀN TẤT HUẤN LUYỆN THÀNH CÔNG ===")

if __name__ == "__main__":
    main()
