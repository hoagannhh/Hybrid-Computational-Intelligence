# Hybrid Computational Intelligence for Stock Market Prediction


```text
run_experiments.py                    # CLI, config, chọn thiết bị, gọi pipeline
src/
  data/sources.py                    # CSV/Yahoo/Kaggle, mapping cột, kiểm tra dữ liệu
  data/preparation.py                # Split theo thời gian, scaler train, tạo cửa sổ
  features/causal.py                 # Chỉ báo chỉ dùng hiện tại/quá khứ
  features/window_wavelet.py         # Wavelet trong từng cửa sổ đầu vào
  models/neural/lstm_model.py        # Lớp StockLSTM dùng cho local
  models/fuzzy/anfis.py              # ANFIS: hàm thuộc bell, luật Takagi–Sugeno bậc một
  models/checkpoints.py              # LSTM v1 / ANFIS v2; tự khôi phục đúng kiến trúc
  models/hybrid/registry.py          # Đăng ký bốn tổ hợp LSTM/Wavelet/PSO và ANFIS
  models/optimizers/pso_search.py     # PSO tìm siêu tham số trên toàn train/val
  training/supervised.py             # Vòng lặp chung, early stopping, dự đoán theo batch
  training/lstm.py, training/anfis.py # Điểm vào train cho từng kiến trúc
  experiments/runner.py              # Ghép các bước và chạy model/seed đã chọn
  evaluation/experiment.py           # Metrics, biểu đồ, backtest minh họa
  utils/experiment.py                # Seed, thư mục run, metadata
```

`--models all` chạy LSTM, LSTM–PSO, Wavelet–LSTM, Wavelet–LSTM–PSO và ANFIS; Naive được đánh giá tự động.
Wavelet là bước biến đổi đầu vào, PSO là bước chọn siêu tham số nên các tổ hợp dùng chung lớp LSTM.
Random Forest, SVR, XGBoost trong `src/models` chưa tích hợp vào CLI này.

ANFIS là nhánh Neural + Fuzzy độc lập, sử dụng vector đặc trưng ở phiên cuối trong cửa sổ quá khứ.
LSTM sử dụng cả cửa sổ; hai model có cùng ngày target/test, scaler, split, seed và ngân sách train cuối.
ANFIS học cả hàm thuộc bell và hệ số tuyến tính của luật bằng Adam (không dùng bước bình phương tối thiểu
của thuật toán học ANFIS cổ điển). Độ rộng/độ dốc luôn dương, chuẩn hóa luật trong miền log để tránh
underflow. Số luật mặc định 5; cấu hình `ANFIS_NUM_RULES`, `ANFIS_LEARNING_RATE` trong YAML hoặc
`--anfis-rules`, `--anfis-lr` trên CLI. Đặt cấu hình trước khi đánh giá test. Chưa có ANFIS–PSO hay LSTM–ANFIS.

Các script `train.py`, `evaluate.py`, `backtest.py` và module cũ được giữ để dùng lại các lần chạy trước,
với config `configs/config*.yaml`. Thí nghiệm mới nên dùng `run_experiments.py` và
`configs/experiment.local.yaml`; không dùng checkpoint của hai CLI thay lẫn nhau.
Các run v1 đã tạo bằng `run_experiments.py` trước khi tách code vẫn đọc lại được bằng CLI mới.

## Dữ liệu trung gian của CLI cũ (`train.py`)

Chạy `python train.py` sẽ tự lưu dữ liệu sau bước tiền xử lý, trước khi huấn luyện.
Mỗi lần chạy tạo một thư mục `<ticker>_<timestamp UTC>` riêng trong cả hai nơi:

- `data/features/<run_id>/features.csv`: bảng đặc trưng dùng trong lần chạy.
- `data/processed/<run_id>/cleaned.csv`: bảng sau làm sạch.
- `data/processed/<run_id>/train.csv`, `val.csv`, `test.csv`: các tập chia theo thời gian, còn giữ đơn vị gốc, chưa chuẩn hóa.
- `data/processed/<run_id>/sequences.npz`: các mảng `X_train`, `y_train`, `X_val`, `y_val`, `X_test`, `y_test` đã chuẩn hóa và tạo cửa sổ trượt.
- `data/processed/<run_id>/scalers.pkl`: feature scaler và target scaler đã fit trên tập train.
- `data/processed/<run_id>/metadata.json`: cấu hình, thứ tự cột đặc trưng, kích thước mảng và mốc thời gian các tập.

Ví dụ đọc mảng đã lưu: `arrays = numpy.load("data/processed/<run_id>/sequences.npz")`,
sau đó dùng `arrays["X_train"]`. Chỉ nạp file pickle do bạn tạo hoặc từ nguồn tin cậy.
Các file này được `.gitignore` bỏ qua. CLI vẫn tính lại dữ liệu mỗi lần chạy;
chưa tự động dùng các file đã lưu làm cache. Luồng notebook có cơ chế lưu riêng.


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

## Cài đặt và các lệnh CLI cũ

### 1. Tạo môi trường ảo và cài đặt thư viện
```bash
# Tạo môi trường ảo
py -3.10 -m venv .venv

# Kích hoạt trên Windows
.venv\Scripts\activate

# Cài đặt các gói cần thiết
pip install -r requirements.txt
```

### 2. Huấn luyện mô hình lai
CLI cũ train LSTM; hàm Wavelet trong lớp hybrid chưa được nối vào luồng `train.py`.
Để chạy Wavelet thực sự, dùng `run_experiments.py --models wavelet-lstm` ở trên.
Lệnh cũ:
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
