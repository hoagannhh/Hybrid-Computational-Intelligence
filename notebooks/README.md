# Notebook cho Kaggle GPU

**Chạy ngay:** upload `04_hybrid_model_experiment.ipynb` lên Kaggle, chọn Accelerator GPU,
bật Internet và Run All. Notebook này tự chạy toàn bộ quy trình tải dữ liệu → tạo đặc trưng →
split → train LSTM, LSTM–PSO và ANFIS → evaluate → xuất bảng/biểu đồ.
Không cần clone repository hoặc upload `src/`. Các notebook chứa mã độc lập và không import
`src` hay `run_experiments.py`. Chỉ cần file `.ipynb`, thư viện và nguồn dữ liệu đã cấu hình.
Notebook 05 cần thêm thư mục kết quả của một lần train trước đó.

## Các notebook

| File | Chức năng | GPU |
|---|---|---|
| `01_eda_and_data_collection.ipynb` | Tải Yahoo hoặc đọc CSV, kiểm tra OHLCV, xuất dữ liệu/EDA | Không cần |
| `02_feature_engineering_and_decomposition.ipynb` | Chỉ báo, scaler, split, cửa sổ, minh họa Wavelet | Không cần |
| `03_baseline_models.ipynb` | LSTM thuần và Naive (giá ngày mai bằng hôm nay) | Có |
| `04_hybrid_model_experiment.ipynb` | LSTM so với LSTM–PSO và ANFIS; tùy chọn ablation Wavelet | Có |
| `05_backtesting_and_ablation_study.ipynb` | Đọc output 03/04, khôi phục checkpoint, evaluate/backtest | Tùy chọn |

01–04 chạy độc lập. Không bắt buộc chạy lần lượt. 05 cần **cả thư mục output run** từ 03 hoặc 04.
Phạm vi thí nghiệm gồm NN + PSO (tối ưu bầy đàn) và ANFIS (Neural + Fuzzy). Chưa tích hợp GA, GWO hay các baseline cổ điển;
không coi các tên mô hình có trong README gốc là kết quả đã triển khai trong notebook.

## Cách dùng trên Kaggle

1. Tạo notebook, dùng chức năng Import/Upload Notebook để chọn `.ipynb`.
2. Trong Settings, bật Accelerator GPU cho 03/04; bật Internet nếu cần tải Yahoo/cài gói phụ.
3. Sửa cell `CONFIG`: ticker, ngày, nguồn dữ liệu, ngân sách train/PSO. Mặc định dùng `^GSPC`
   (chỉ số S&P 500), lịch sử 2018–2023; đổi sang AAPL/MSFT nếu muốn cổ phiếu riêng lẻ.
4. Có thể đặt `SMOKE_TEST=True` để kiểm tra nhanh trước. Sau đó đặt lại False cho thí nghiệm thật.
   Nếu chạy CPU local, đặt `REQUIRE_GPU=False`.
5. Run All; kiểm tra dòng `Device: cuda ...`. Notebook dùng GPU đầu tiên, không chạy đa GPU.
6. Save Version và giữ output. Mỗi lần chạy tạo một thư mục riêng trong `/kaggle/working/hci/`.

Tham khảo [Kaggle Notebooks](https://www.kaggle.com/docs/notebooks),
[hướng dẫn GPU Kaggle của NVIDIA](https://docs.nvidia.com/datascience/deployment/latest/platforms/kaggle/),
[yfinance API](https://ranaroussi.github.io/yfinance/reference/index.html).
Notebook giữ PyTorch/CUDA có sẵn của Kaggle, chỉ cài yfinance/PyWavelets/kagglehub nếu cần và thiếu.

## Đổi dataset bằng preset hoặc link Kaggle

Cell CONFIG có `DATASET_PRESET = "yahoo"`. Đổi thành:

```python
DATASET_PRESET = "kaggle_apple"  # Apple HistoricalQuotes.csv
# hoặc:
DATASET_PRESET = "kaggle_tech"
TECH_TICKER = "AAPL"            # AAPL / MSFT / NVDA / GOOG / TSLA
```

Preset điền sẵn tên dataset, file, mapping và định dạng ngày. `SOURCE="kaggle"` dùng
[kagglehub.dataset_download](https://github.com/Kaggle/kagglehub#download-dataset) để tải/gắn một file,
không tải cả năm mã. Bật Internet nếu chưa có gói/dữ liệu. Dataset riêng tư hoặc yêu cầu chấp thuận
có thể cần quyền tài khoản Kaggle; không ghi token vào notebook.

Với dataset khác, chọn `DATASET_PRESET="custom"` rồi sửa CONFIG, ví dụ:

```python
"SOURCE": "kaggle",
"KAGGLE_DATASET": "https://www.kaggle.com/datasets/owner/dataset-name",
"KAGGLE_FILE": "prices.csv",
"TICKER": "AAPL",
"START": None, "END": None,
"COLUMN_MAP": {"date": "Date", "open": "Open", "high": "High", "low": "Low", "close": "Close", "volume": "Volume"},
"TICKER_COLUMN": "Name",  # Nếu một CSV chứa nhiều mã; None nếu mỗi file một mã.
```

`KAGGLE_DATASET` nhận URL (kể cả link cũ không có /datasets và có query) hoặc `owner/dataset`;
có thể dùng `owner/dataset/versions/N` để cố định phiên bản. CSV_PATH là đường dẫn file, không phải URL trang Kaggle.
Mỗi run train một mã riêng; không nối các mã thành chuỗi và không dùng chung scaler giữa các dataset.

### Hai dataset đã kiểm tra trực tiếp

- [Apple Historical Stock Data](https://www.kaggle.com/datasets/tarunpaparaju/apple-aapl-historical-stock-data):
  `HistoricalQuotes.csv`, 2.518 dòng, 2010-03-01 đến 2020-02-28. Bộ đọc bỏ khoảng trắng tên cột,
  chuyển `Close/Last` thành Close, bỏ `$`, chuyển số và sắp xếp ngày tăng dần. Không giả định giá đã điều chỉnh.
  Sau warmup/split: 1.689 train, 375 val, 375 test; có 18 đặc trưng.
- [Tech Stocks Dataset](https://www.kaggle.com/datasets/sinankr/tech-stocks-dataset): đã kiểm tra file `AAPL.csv`.
  Cột thực tế là `stock_close/high/low/volume`, không có Open; chỉ dùng bốn cột này, tạo 17 đặc trưng,
  không bịa Open và không dùng các chỉ báo/vĩ mô đã tính sẵn. File có 15.161 dòng timestamp nhưng chỉ 5.534 ngày;
  3.227 ngày có nhiều giá trị stock khác nhau. README nguồn cho biết đã nội suy giá.
  Preset **mặc định strict và dừng khi gặp nhiều dòng/ngày** để tránh coi mỗi timestamp là một phiên.
  Để khám phá có chủ ý, thêm `CONFIG["DAILY_POLICY"]="last_snapshot"` sau dòng áp dụng preset:
  giữ snapshot cuối mỗi ngày UTC, không cộng volume. Cách này không khôi phục dữ liệu thô hay loại bỏ rò rỉ
  có thể có từ nội suy của nguồn. Không coi bộ này là dữ liệu sạch chỉ vì đọc được; nên dùng Apple/Yahoo
  cho thí nghiệm chính và đánh giá nguồn Tech Stocks riêng trước khi kết luận.

Thông tin trên là kết quả kiểm tra file được tải trong phiên làm việc, không phải cam kết cho mọi phiên bản sau.
Nếu đổi dataset, kiểm tra khoảng ngày, cơ sở giá (adjusted/unadjusted), tần suất và nguồn tiền xử lý.

## Dùng CSV / không truy cập được Yahoo

Tạo Kaggle Dataset chứa CSV từ `data/raw/` hoặc notebook 01, rồi Add Input dataset vào notebook.
Đặt trong config:

```python
"SOURCE": "csv",
"CSV_PATH": "/kaggle/input/ten-dataset/GSPC_2018-01-01_2024-01-01.csv",
"CSV_PRICE_BASIS": "adjusted",  # Chỉ đặt khi đã xác nhận giá điều chỉnh.
```

CSV cần cột ngày, High, Low, Close, Volume; Open tùy chọn và không bị tạo giả khi thiếu.
Không bắt buộc Date là cột đầu. `COLUMN_MAP` ánh xạ tên gốc → tên chuẩn; `DATE_COLUMN`, `DATE_FORMAT`,
`TICKER_COLUMN` hỗ trợ định dạng khác. Tên chuẩn không phân biệt hoa/thường; Ticker/Symbol tự được nhận diện.
`CSV_SEP`, `CSV_ENCODING`, `CSV_STRIP_TOKENS`, `CSV_THOUSANDS`, `CSV_DECIMAL` điều khiển cách đọc số.
Ngày có timezone được đổi về UTC, sau đó dùng ngày UTC; dữ liệu intraday thực sự cần quy tắc tổng hợp riêng.
Notebook kiểm tra ngày trùng, NaN/Inf, giá/volume và quan hệ OHLC; lọc theo START/END.
START/END đặt None để dùng toàn bộ CSV. CSV một mã phải đúng ticker đã khai báo.
Nếu CSV_PATH trống, tìm đúng KAGGLE_FILE (nếu có) hoặc `<ticker bỏ ^>_<START>_<END>.csv` trong `/kaggle/input`;
nếu không tìm thấy hoặc có nhiều file, notebook yêu cầu đặt đường dẫn rõ ràng.
Yahoo được tải với `auto_adjust=True`; dữ liệu CSV không tự đổi cơ sở giá.
Internet có thể tắt khi dùng CSV nếu các gói cần thiết đã có sẵn.
Khi đã chọn preset, đặt `CONFIG.update(SOURCE="csv", CSV_PATH="/kaggle/input/.../file.csv")`
**sau** `CONFIG.update(PRESETS[DATASET_PRESET])` để chuyển sang file Add Input mà vẫn giữ mapping của preset.

## Thiết kế thí nghiệm

- Target: Close của phiên kế tiếp; chỉ hỗ trợ HORIZON=1.
- Chỉ báo causal, bỏ các dòng warmup; không backward fill.
- Chia theo thời gian 70/15/15; scaler fit trên train, không fit lại ở evaluate.
- Val/test tận dụng 60 phiên lịch sử trước target, kể cả lịch sử ở cuối tập trước.
- Naive, LSTM, LSTM–PSO, ANFIS đánh giá cùng ngày test và cùng target thật.
- Notebook 04 mặc định `RUN_ANFIS=True`; đặt False để tắt. `ANFIS_NUM_RULES=5`,
  `ANFIS_LEARNING_RATE=0.001` là cấu hình riêng; epoch, patience, batch size và seeds dùng chung.
  ANFIS dùng vector đặc trưng của phiên cuối trong cửa sổ; LSTM đọc toàn chuỗi.
  ANFIS bậc một học hàm thuộc bell và hệ số tuyến tính bằng Adam, không ghép LSTM hoặc PSO.
  Chuẩn hóa luật trong miền log và tham số hóa độ rộng/độ dốc dương giúp ổn định số học.
  Đây là so sánh hai cách học khác nhau; không coi là ablation cùng kiến trúc.
- PSO tìm hidden size, dropout, learning rate trên toàn train/val, không đọc test.
  Mặc định 6 hạt × (1 lượt khởi tạo + 5 lượt cập nhật), tối đa 8 epoch mỗi ứng viên.
- Train cuối tối đa 60 epoch, patience 10, 3 seed. Báo cáo mean/std, không chọn seed tốt nhất từ test.
  PSO có ngân sách tìm kiếm thêm; thời gian tìm kiếm được ghi riêng và dùng chung giữa các seed.
- `RUN_WAVELET=True` trong 04 thêm Wavelet–LSTM và Wavelet–LSTM–PSO.
  Wavelet chỉ lọc Close bên trong từng cửa sổ quá khứ, giữ nguyên target thật.
  Khi bật cả RUN_ANFIS và RUN_WAVELET, có năm model được train, cộng baseline Naive.
- Kết quả một ticker và một holdout chưa chứng minh tổng quát hóa mọi thị trường/giai đoạn.

## Output và đánh giá lại

```text
<ticker>_comparison_<timestamp>/
  config.json, environment.json, data_source.json
  <ticker>_<START>_<END>.csv
  features.csv, splits.csv, preprocessing.json
  prepared_data.npz
  prepared_wavelet.npz                 # nếu bật Wavelet
  experiments.json
  raw/
    pso_history.csv, pso_best.json
    LSTM_seed42/best.pt, history.csv
    LSTM_PSO_seed42/best.pt, history.csv
    ANFIS_seed42/best.pt, history.csv
    ...
  wavelet/                            # nếu bật Wavelet
  evaluation/
    metrics_by_seed.csv, metrics_summary.csv
    predictions.csv, comparison.png
  validation_curves.png
```

Checkpoint chứa kiến trúc và tham số đã chọn để không gặp lỗi hidden size sau PSO.
LSTM dùng định dạng v1 (run cũ vẫn đọc được); ANFIS dùng v2 với `model_type="ANFIS"` và số luật.
Dùng bản notebook 05 mới khi đánh giá run có ANFIS.
Scaler/thứ tự đặc trưng/split được lưu trong `preprocessing.json`; 05 dùng dữ liệu snapshot đã lưu,
không tải Yahoo hoặc fit scaler lại. Upload/gắn **toàn bộ run** vào notebook 05 và đặt `RUN_PATH`
trỏ tới thư mục chứa `experiments.json`. Kết quả evaluate lại phải khớp lần đầu trong cùng điều kiện số học.

Backtest 05 chỉ minh họa close-to-close, có phí và thanh lý cuối kỳ. Giả định khớp tại Close đã biết,
chưa mô hình hóa spread/slippage/độ trễ; không diễn giải là hiệu suất giao dịch thực tế.
Không dùng test để tiếp tục chọn tham số sau khi xem bảng kết quả.

## Bảo trì và kiểm tra local

Để chạy cùng các mô hình ngay trong terminal local, không cần Jupyter:

```powershell
python run_experiments.py --models all --smoke
python run_experiments.py --models lstm lstm-pso
python run_experiments.py --models lstm lstm-pso anfis
python run_experiments.py --models all --backtest
```

Lệnh đầu chạy ngắn; các lệnh còn lại dùng đầy đủ cấu hình `configs/experiment.local.yaml`.
Xem phần thí nghiệm local trong README gốc để chọn từng model, đổi ngân sách và evaluate lại.
CLI local dùng code trong `src/`, hoàn toàn không phụ thuộc thư mục notebook.
Kết quả local dùng định dạng run tương thích với notebook 05 mới (checkpoint LSTM v1 / ANFIS v2).

`_kaggle_workflow.py` và `build_notebooks.py` dùng để sinh notebook; **không phải upload chúng lên Kaggle**.
Hai file này chỉ phục vụ bản Kaggle, không đọc hoặc sinh code từ `src/`.
Sửa `src/` không tự thay đổi notebook; sửa helper notebook không thay đổi local.
Khi chủ động áp dụng một thay đổi cho cả hai, sửa và kiểm tra từng bản riêng.
Sau khi sửa mã nguồn notebook, chạy từ thư mục gốc project:

```powershell
.\.venv\Scripts\python.exe -X utf8 notebooks/build_notebooks.py
.\.venv\Scripts\python.exe -X utf8 -m pytest tests/test_kaggle_notebooks.py -q
```

Tests chạy các code cell offline, kiểm tra causal features/split, checkpoint round trip và backtest.
Kiểm tra local CPU không thay thế xác nhận GPU thực tế trong phiên Kaggle của bạn.
