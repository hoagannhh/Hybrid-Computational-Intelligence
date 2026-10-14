# Hybrid Computational Intelligence for Stock Market Prediction

## Chạy trên Kaggle GPU

Upload [`notebooks/04_hybrid_model_experiment.ipynb`](notebooks/04_hybrid_model_experiment.ipynb)
để chạy độc lập toàn bộ quy trình LSTM so với LSTM–PSO, có tùy chọn Wavelet.
Xem [hướng dẫn notebook](notebooks/README.md) để cấu hình GPU, CSV offline và đánh giá lại checkpoint.
Luồng notebook có preprocessing/checkpoint riêng; hướng dẫn CLI bên dưới mô tả mã CLI hiện có,
không thay thế các kiểm tra và cách lưu kết quả trong notebook.

> **Đề tài:** Ứng dụng Trí tuệ tính toán lai (Hybrid Computational Intelligence - HCI) trong dự đoán thị trường chứng khoán.

Dự án này kết hợp các trụ cột cốt lõi của **Trí tuệ tính toán (Computational Intelligence)**:
1. **Deep Learning / Neural Networks (Mạng nơ-ron sâu):** Mô hình hóa các quan hệ phi tuyến tính phức tạp và phụ thuộc thời gian (LSTM, BiLSTM, GRU, Transformer, CNN-LSTM).
2. **Fuzzy Systems / Neuro-Fuzzy (Hệ mờ & Nơ-ron mờ):** Xử lý tính bất định, mơ hồ và nhiễu loạn của thị trường tài chính (ANFIS - Adaptive Neuro-Fuzzy Inference System).
3. **Evolutionary Computation / Metaheuristics (Giải thuật tiến hóa & Bầy đàn):** Tối ưu hóa siêu tham số (Hyperparameter Tuning), chọn đặc trưng (Feature Selection) và tối ưu hóa trọng số (PSO - Particle Swarm Optimization, GA - Genetic Algorithm, GWO - Grey Wolf Optimizer).
4. **Signal Processing & Decomposition (Xử lý tín hiệu):** Khử nhiễu và phân rã đa độ phân giải chuỗi tài chính phi dừng (Wavelet Transform DWT, EMD/CEEMDAN).

---

## 📂 Cấu trúc thư mục dự án

```text
Hybrid-Computational-Intelligence/
│
├── configs/                          # Quản lý cấu hình thử nghiệm tập trung (YAML)
│   └── config.yaml                   # Thiết lập siêu tham số dữ liệu, mô hình, optimizer
│
├── data/                             # Quản lý dữ liệu chuỗi thời gian
│   ├── raw/                          # Dữ liệu gốc OHLCV từ yfinance / vnstock
│   ├── processed/                    # Dữ liệu đã làm sạch, xử lý missing values & chuẩn hóa
│   └── features/                     # Dữ liệu đặc trưng kỹ thuật & phân rã tín hiệu
│
├── notebooks/                        # Jupyter Notebooks phục vụ EDA và báo cáo nghiên cứu
│   ├── README.md
│   ├── 01_eda_and_data_collection.ipynb
│   ├── 02_feature_engineering_and_decomposition.ipynb
│   ├── 03_baseline_models.ipynb
│   ├── 04_hybrid_model_experiment.ipynb
│   └── 05_backtesting_and_ablation_study.ipynb
│
├── saved_models/                     # Lưu trữ trọng số mô hình tốt nhất (.pt / .pkl)
│
├── reports/                          # Xuất báo cáo, bảng biểu phục vụ khóa luận / bài báo
│   ├── figures/                      # Biểu đồ so sánh giá, biểu đồ phân rã Wavelet, Loss curve
│   └── tables/                       # Bảng so sánh chỉ số đánh giá (RMSE, MAE, Sharpe, MDD)
│
├── src/                              # Mã nguồn chính của dự án (Modular Clean Architecture)
│   ├── __init__.py
│   ├── data/                         # Xử lý & Tải dữ liệu
│   │   ├── __init__.py
│   │   ├── data_fetcher.py           # Tải giá chứng khoán (Yahoo Finance / vnstock / CSV)
│   │   ├── preprocessor.py           # Chuẩn hóa (MinMax), chống rò rỉ dữ liệu, Sliding Window
│   │   └── data_loader.py            # PyTorch Dataset & DataLoader
│   │
│   ├── features/                     # Kỹ nghệ đặc trưng & Xử lý tín hiệu
│   │   ├── __init__.py
│   │   ├── technical_indicators.py   # RSI, MACD, Bollinger Bands, ATR, EMA, SMA
│   │   └── signal_decomposition.py   # Phân rã & Khử nhiễu Wavelet (DWT), EMD
│   │
│   ├── models/                       # Kiến trúc các mô hình Trí tuệ tính toán
│   │   ├── __init__.py
│   │   ├── baselines/                # Mô hình đối chuẩn: ARIMA, SVR, Random Forest, XGBoost
│   │   │   ├── __init__.py
│   │   │   └── classic_models.py
│   │   ├── neural/                   # Mạng nơ-ron học sâu (Deep Learning)
│   │   │   ├── __init__.py
│   │   │   └── lstm_model.py         # LSTM, BiLSTM, Attention
│   │   ├── fuzzy/                    # Hệ suy diễn nơ-ron mờ (Neuro-Fuzzy)
│   │   │   ├── __init__.py
│   │   │   └── anfis.py              # Takagi-Sugeno ANFIS tích hợp PyTorch
│   │   ├── optimizers/               # Thuật toán tối ưu bầy đàn & tiến hóa (Metaheuristics)
│   │   │   ├── __init__.py
│   │   │   └── metaheuristics.py     # PSO, GA, GWO
│   │   └── hybrid/                   # Mô hình lai (Wavelet-LSTM-PSO, ANFIS-GA,...)
│   │       ├── __init__.py
│   │       └── wavelet_lstm_pso.py
│   │
│   ├── training/                     # Quản lý huấn luyện
│   │   ├── __init__.py
│   │   ├── trainer.py                # Vòng lặp Train/Val, Early Stopping, lưu Checkpoint
│   │   └── loss_functions.py         # MSE, Huber Loss, Directional Penalty Loss
│   │
│   ├── evaluation/                   # Đánh giá & Giả lập giao dịch
│   │   ├── __init__.py
│   │   ├── metrics.py                # RMSE, MAE, MAPE, R2, Directional Accuracy (DA)
│   │   └── backtester.py             # Chiến lược giao dịch giả lập, Sharpe Ratio, Max Drawdown
│   │
│   └── utils/                        # Tiện ích bổ trợ
│       ├── __init__.py
│       ├── helpers.py                # Seed reproducibility, nạp config, chọn CUDA/CPU
│       ├── logger.py                 # Logging chuẩn mực
│       └── visualizer.py             # Vẽ đồ thị kết quả
│
├── tests/                            # Unit tests kiểm tra tính toàn vẹn của mã nguồn
│   ├── __init__.py
│   └── test_pipeline.py
│
├── .gitignore                        # Bỏ qua file nhị phân, dữ liệu lớn, venv, cache
├── requirements.txt                  # Danh mục các thư viện phụ thuộc
├── train.py                          # Script thực thi huấn luyện mô hình
├── evaluate.py                       # Script đánh giá mô hình trên tập Test
└── backtest.py                       # Script kiểm thử chiến lược giao dịch thực tế
```

---

## 🚀 Hướng dẫn cài đặt & Thực thi

### 1. Tạo môi trường ảo và cài đặt thư viện
```bash
# Tạo môi trường ảo
python -m venv venv

# Kích hoạt trên Windows
venv\Scripts\activate

# Cài đặt các gói cần thiết
pip install -r requirements.txt
```

### 2. Huấn luyện mô hình lai
Chạy huấn luyện mô hình kết hợp Wavelet Denoising + LSTM:
```bash
python train.py --config configs/config.yaml
```

Nếu muốn kích hoạt giải thuật bầy đàn **PSO** để tự động tìm siêu tham số tối ưu (Hyperparameter Tuning):
```bash
python train.py --config configs/config.yaml --optimize_pso
```

### 3. Đánh giá mô hình trên tập Test
```bash
python evaluate.py --config configs/config.yaml --model_path saved_models/best_model.pt
```
Kết quả biểu đồ đối chiếu thực tế và dự đoán sẽ được lưu tự động tại `reports/figures/test_prediction_comparison.png`.

### 4. Kiểm thử chiến lược giao dịch (Backtesting)
```bash
python backtest.py --config configs/config.yaml --model_path saved_models/best_model.pt
```
Script sẽ tính toán lợi nhuận tích lũy (Cumulative Returns), tỷ số Sharpe, và mức độ sụt giảm tối đa (Maximum Drawdown) so với chiến lược Buy & Hold.

### 5. Chạy Unit Tests
```bash
pytest tests/
```
