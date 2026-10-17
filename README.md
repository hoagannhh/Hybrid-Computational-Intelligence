# Hybrid Computational Intelligence for Stock Market Prediction

## Chạy local: train → validation → test

Chạy các lệnh PowerShell dưới đây từ thư mục gốc project. Pipeline hiện tại dùng
`run_experiments.py` và `configs/experiment.local.yaml`; mã xử lý nằm trong `src/`.
Notebook Kaggle chạy độc lập; dùng [notebook tổng hợp 06](notebooks/06_complete_kaggle_pipeline.ipynb)
để chạy EDA, train năm model, loss, test và backtest trong một file, rồi tải ZIP kết quả.
Xem [hướng dẫn notebook](notebooks/README.md) để đổi dataset và chọn model.

### 1. Môi trường và dữ liệu

Nếu chưa có môi trường, tạo bằng Python 3.10 và cài thư viện một lần:

```powershell
py -3.10 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

Nếu `.venv` đã có, chỉ cần kích hoạt:

```powershell
.\.venv\Scripts\Activate.ps1
```

Config mặc định dùng CPU và CSV `data/raw/GSPC_2018-01-01_2024-01-01.csv` đã tải.
Dữ liệu không được commit lên Git: khi clone sang máy khác, cần cung cấp CSV hoặc sửa
`SOURCE: yfinance` trong YAML để tải Yahoo lúc chạy. Mỗi lệnh train tự thực hiện đọc/tải dữ liệu,
tạo đặc trưng, chia train/val/test theo thời gian 70/15/15 và chuẩn hóa bằng scaler fit trên train.
Không cần chạy lệnh split riêng. Nếu đổi CSV, kiểm tra `TICKER`, `COLUMN_MAP`, `START`, `END`;
đặt `START: null`, `END: null` trong YAML để dùng toàn bộ khoảng ngày của file.

### 2. Train và tự lưu model best

Chọn một lệnh phù hợp dưới đây; không cần chạy lần lượt tất cả các ví dụ.

```powershell
# Kiểm tra nhanh cả 5 model: 2 epoch, 1 seed, PSO nhỏ.
python run_experiments.py --models all --smoke

# Train cả 5 model theo cấu hình đầy đủ; tự evaluate test và xuất thêm backtest.
python run_experiments.py --models all --backtest

# So sánh LSTM thuần với hai nhánh hybrid PSO và Fuzzy.
python run_experiments.py --models lstm lstm-pso anfis --backtest

# Chạy theo danh sách MODELS trong YAML (mặc định LSTM, LSTM_PSO, ANFIS).
python run_experiments.py --config configs/experiment.local.yaml
```

Train từng model riêng:

```powershell
python run_experiments.py --models lstm
python run_experiments.py --models lstm-pso
python run_experiments.py --models wavelet-lstm
python run_experiments.py --models wavelet-lstm-pso
python run_experiments.py --models anfis
```

Các lệnh train đều tự lưu checkpoint best và evaluate trên test. Thêm `--backtest` nếu cần
bảng/biểu đồ backtest minh họa. `--models all` gồm năm model; Naive luôn được thêm khi đánh giá.
Mặc định train cuối tối đa **60 epoch, patience 10, ba seed 42/123/2026**.
Early stopping có thể dừng sớm. `--smoke` chỉ để kiểm tra luồng, không dùng kết quả để kết luận hiệu quả model;
giữ `SMOKE_TEST: false` và bỏ `--smoke` khi chạy thí nghiệm đầy đủ.

Ví dụ thay ngân sách và cấu hình ANFIS:

```powershell
python run_experiments.py --models all --epochs 100 --seeds 42 123 2026 --particles 6 --iterations 5 --search-epochs 8
python run_experiments.py --models anfis --anfis-rules 5 --anfis-lr 0.001
```

PSO tìm siêu tham số trên train/validation trước khi train cuối, không dùng test.
Lệnh có `--smoke` sẽ ghi đè ngân sách thành cấu hình chạy ngắn.
Chọn `--device cuda` chỉ khi máy có GPU và PyTorch hỗ trợ CUDA; mặc định local là `cpu`.

### 3. Validation và lựa chọn checkpoint

**Validation chạy tự động sau mỗi epoch; hiện không có lệnh `--val` hoặc `--split val` riêng.**
Model được cập nhật trọng số trên train; validation dùng để early stopping và chọn checkpoint
có MSE validation tốt nhất. Mỗi model/seed có một `best.pt` và một `history.csv`
chứa `epoch`, `train_mse`, `val_mse`. Thông tin `best_epoch`, `best_val_mse` và tham số của mỗi model
nằm trong `experiments.json`.

Xem epoch và loss validation tốt nhất của từng model/seed trong run gần nhất:

```powershell
$runPath = (Get-Content -Encoding UTF8 -Raw reports/experiments/latest_run.txt).Trim()
Get-Content -Encoding UTF8 -Raw "$runPath/experiments.json" | ConvertFrom-Json | Select-Object run_id, best_epoch, best_val_mse, checkpoint | Format-Table
```

Loss train/validation là **MSE trên target đã chuẩn hóa**, khác RMSE/MAE theo đơn vị giá trong bảng test.
Giữ và báo cáo kết quả qua các seed; không chọn seed hay sửa siêu tham số dựa trên điểm test.

### 4. Test/evaluate lại model đã lưu

Sau train, chương trình tự đánh giá các checkpoint best trên tập test và lưu vào `evaluation/`.
Muốn đánh giá lại mà không train lại hoặc tải dữ liệu, dùng:

```powershell
# Đánh giá tất cả model có trong run gần nhất.
python run_experiments.py --evaluate-run reports/experiments/latest_run.txt

# Đánh giá lại kèm backtest minh họa.
python run_experiments.py --evaluate-run reports/experiments/latest_run.txt --backtest
```

Hoặc chọn một run cụ thể (thay `<ten-run>` bằng tên thư mục thực tế):

```powershell
python run_experiments.py --evaluate-run "reports/experiments/<ten-run>" --backtest
```

Kết quả đánh giá lại nằm trong `evaluation/recheck_<timestamp>/`, gồm metrics, dự đoán và biểu đồ.
`--evaluate-run` đánh giá toàn bộ model trong run; `--models` không lọc model ở chế độ này.
Giữ **toàn bộ thư mục run**, vì đánh giá cần checkpoint, metadata, scaler và dữ liệu snapshot.
Backtest chỉ là mô phỏng close-to-close có phí, chưa tính spread/slippage/độ trễ.

Xem bảng test được tạo ở cuối lần train:

```powershell
$runPath = (Get-Content -Encoding UTF8 -Raw reports/experiments/latest_run.txt).Trim()
Import-Csv "$runPath/evaluation/metrics_by_seed.csv" | Format-Table
```

`metrics_summary.csv` chứa mean/std qua các seed. Mỗi lệnh train tạo một run riêng;
`latest_run.txt` chỉ trỏ tới run hoàn tất gần nhất, kể cả khi đó là smoke test.
Nếu đã đổi `--output-root`, dùng file `latest_run.txt` trong thư mục output tương ứng.

### 5. Vẽ loss từ history đã lưu

Sau train, biểu đồ loss được tạo tự động. Vẽ lại từ run cũ bằng:

```powershell
python run_experiments.py --plot-loss-run reports/experiments/latest_run.txt
```

Lệnh này không train hoặc nạp checkpoint; nó đọc `history.csv` và cập nhật ảnh trong `learning_curves/`.
Có ảnh train/validation riêng cho từng model/seed, đánh dấu epoch tốt nhất và ảnh tổng hợp
`validation_curves.png`. Các đường kết thúc tại epoch thực sự đã chạy, kể cả khi early stopping.

Truyền trực tiếp `latest_run.txt` để Python tự đọc UTF-8, tránh lỗi đường dẫn tiếng Việt trên Windows.
Nếu dùng biến PowerShell như các ví dụ phía trên, cần giữ `-Encoding UTF8`.

### 6. Nơi lưu kết quả

```text
reports/experiments/<ten-run>/
  config.json, environment.json, data_source.json
  features.csv, splits.csv, preprocessing.json
  prepared_data.npz, experiments.json
  prepared_wavelet.npz                  # nếu có model Wavelet
  raw/
    LSTM_seed42/best.pt, history.csv
    LSTM_PSO_seed42/best.pt, history.csv
    ANFIS_seed42/best.pt, history.csv
    pso_history.csv, pso_best.json       # nếu chạy LSTM_PSO
  wavelet/                              # nếu chọn model Wavelet
    Wavelet_LSTM_seed42/best.pt, history.csv
    Wavelet_LSTM_PSO_seed42/best.pt, history.csv
  evaluation/
    metrics_by_seed.csv, metrics_summary.csv
    predictions.csv, comparison.png
    backtest.csv, backtest.png           # khi có --backtest
    recheck_<timestamp>/                # khi evaluate lại
  learning_curves/
    LSTM_seed42.png, ...
    validation_curves.png
```

Các seed khác có thư mục tương ứng. Chạy cả năm model với ba seed mặc định sẽ lưu 15 checkpoint best.
`reports/experiments/` đã được `.gitignore` bỏ qua; các file vẫn nằm trên máy.

## Các module của pipeline local

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
  evaluation/learning_curves.py      # Vẽ train/validation loss từ history
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

## Dữ liệu trung gian của CLI (`train.py`)

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
│   ├── experiment.local.yaml         # Cấu hình pipeline hiện tại
│   └── config.yaml                   # Cấu hình CLI cũ
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
├── saved_models/                     # Checkpoint của CLI cũ
│
├── reports/                          # Xuất báo cáo, bảng biểu phục vụ khóa luận / bài báo
│   ├── experiments/                  # Toàn bộ kết quả theo run của pipeline mới
│   ├── figures/                      # Biểu đồ của CLI cũ
│   └── tables/                       # Thư mục dự phòng của cấu trúc cũ
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
├── run_experiments.py                # CLI chính: train, validation tự động, test, backtest, loss
├── train.py                          # Script thực thi huấn luyện mô hình
├── evaluate.py                       # Script đánh giá mô hình trên tập Test
└── backtest.py                       # Script kiểm thử chiến lược giao dịch thực tế
```

---

## Các script 

`train.py`, `evaluate.py`, `backtest.py` vẫn được giữ để làm việc với kết quả của pipeline cũ.
Chúng dùng `configs/config*.yaml`, checkpoint trong `saved_models/` và biểu đồ trong `reports/figures/`.
`reports/tables/` là thư mục dự phòng. Các thư mục này không phải output của `run_experiments.py`.

Với các thí nghiệm mới, dùng các lệnh train/validation/test ở đầu README. Checkpoint và config của
hai pipeline có định dạng khác nhau; không đưa `best.pt` của pipeline mới vào `evaluate.py` cũ.

## Kiểm tra code

Lệnh sau chạy kiểm thử phần mềm, khác với đánh giá mô hình trên tập test:

```powershell
python -m pytest tests/ -q
```
