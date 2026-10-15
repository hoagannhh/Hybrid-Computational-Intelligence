"""Regenerate standalone Kaggle notebooks using only the Python standard library."""
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
source = (HERE / "_kaggle_workflow.py").read_text(encoding="utf-8")
SECTIONS = {}
for block in source.split("# %% ")[1:]:
    name, code = block.split("\n", 1)
    SECTIONS[name.strip()] = code.strip() + "\n"


def md(text):
    return {"cell_type": "markdown", "metadata": {}, "source": text.strip().splitlines(keepends=True)}


def code(text, tag=None):
    return {"cell_type": "code", "execution_count": None,
            "metadata": {"tags": [tag]} if tag else {}, "outputs": [],
            "source": (text.strip() + "\n").splitlines(keepends=True)}


CONFIG = '''
from pathlib import Path

DATASET_PRESET = "yahoo"  # yahoo / kaggle_apple / kaggle_tech / custom
TECH_TICKER = "AAPL"      # Khi chọn kaggle_tech: AAPL / MSFT / NVDA / GOOG / TSLA

CONFIG = {
    "TICKER": "^GSPC",                 # Chỉ số S&P 500; đổi thành AAPL/MSFT cho cổ phiếu.
    "START": "2018-01-01",
    "END": "2024-01-01",               # Ngày kết thúc không được lấy vào dữ liệu.
    "SOURCE": "yfinance",              # yfinance / csv (Add Input) / kaggle (link/handle)
    "KAGGLE_DATASET": "",              # owner/dataset hoặc URL trang dataset Kaggle
    "KAGGLE_FILE": "",                 # Tên file bên trong dataset, ví dụ AAPL.csv
    "CSV_PATH": "",                    # Ví dụ /kaggle/input/my-data/GSPC_2018-01-01_2024-01-01.csv
    "CSV_PRICE_BASIS": "unknown",       # Đặt 'adjusted' nếu đã xác nhận CSV dùng giá điều chỉnh.
    "COLUMN_MAP": {},                  # Tên cột gốc -> tên chuẩn, ví dụ {"Close/Last": "Close"}
    "DATE_COLUMN": "Date", "DATE_FORMAT": None,
    "TICKER_COLUMN": None,             # Tự nhận Ticker/Symbol; đặt 'Name' nếu file nhiều mã dùng cột Name
    "CSV_SEP": ",", "CSV_ENCODING": "utf-8",
    "CSV_STRIP_TOKENS": ["$"], "CSV_THOUSANDS": ",", "CSV_DECIMAL": ".",
    "DAILY_POLICY": "strict",          # last_snapshot chỉ dành cho snapshot dữ liệu ngày, không phải nến intraday
    "DATASET_NOTE": "",
    "OUTPUT_ROOT": "/kaggle/working/hci" if Path("/kaggle/working").exists() else "kaggle_outputs",
    "TRAIN_RATIO": 0.70, "VAL_RATIO": 0.15, "TEST_RATIO": 0.15,
    "LOOKBACK": 60, "HORIZON": 1,
    "REQUIRE_GPU": True,                # Các notebook train dừng nếu chưa bật GPU.
    "SMOKE_TEST": False,                # True: chạy ngắn để kiểm tra, không dùng để kết luận.
    "SEEDS": [42, 123, 2026],
    "HIDDEN_DIM": 64, "NUM_LAYERS": 2, "DROPOUT": 0.2,
    "LEARNING_RATE": 0.001, "BATCH_SIZE": 64,
    "RUN_ANFIS": True,                 # Notebook 04: so sánh thêm nhánh Neural + Fuzzy.
    "ANFIS_NUM_RULES": 5, "ANFIS_LEARNING_RATE": 0.001,
    "EPOCHS": 60, "PATIENCE": 10,
    "SEARCH_SEED": 2026,
    "PSO_PARTICLES": 6, "PSO_ITERATIONS": 5,
    "SEARCH_EPOCHS": 8, "SEARCH_PATIENCE": 3,
    "RUN_WAVELET": False,               # Notebook 04: thêm hai mô hình cho ablation nếu True.
    "WAVELET": "db4", "WAVELET_LEVEL": 3,
    "COMMISSION": 0.001,
}

# Preset được áp dụng sau CONFIG. Dùng CONFIG.update(...) bên dưới để ghi đè preset.
PRESETS = {
    "yahoo": {}, "custom": {},
    "kaggle_apple": {
        "SOURCE": "kaggle", "TICKER": "AAPL", "START": None, "END": None,
        "KAGGLE_DATASET": "tarunpaparaju/apple-aapl-historical-stock-data",
        "KAGGLE_FILE": "HistoricalQuotes.csv", "COLUMN_MAP": {"Close/Last": "Close"},
        "DATE_FORMAT": "%m/%d/%Y", "CSV_PRICE_BASIS": "unknown",
        "DATASET_NOTE": "HistoricalQuotes.csv: kiểm tra giá điều chỉnh/split trước khi so sánh với Yahoo.",
    },
    "kaggle_tech": {
        "SOURCE": "kaggle", "TICKER": TECH_TICKER, "START": None, "END": None,
        "KAGGLE_DATASET": "sinankr/tech-stocks-dataset", "KAGGLE_FILE": f"{TECH_TICKER}.csv",
        "COLUMN_MAP": {"stock_close": "Close", "stock_high": "High", "stock_low": "Low", "stock_volume": "Volume"},
        "DATE_FORMAT": "%Y-%m-%d %H:%M:%S%z", "DAILY_POLICY": "strict",
        "DATASET_NOTE": "Nguồn đã nội suy giá và có nhiều snapshot/ngày. Mặc định strict dừng để kiểm tra. last_snapshot chỉ dùng khám phá, không chứng minh đã loại bỏ rò rỉ từ dữ liệu nguồn.",
    },
}
if DATASET_PRESET not in PRESETS:
    raise ValueError(f"Preset không hợp lệ: {DATASET_PRESET}")
CONFIG.update(PRESETS[DATASET_PRESET])
# Ví dụ sau khi Add Input: CONFIG.update(SOURCE="csv", CSV_PATH="/kaggle/input/.../HistoricalQuotes.csv")
# START=None, END=None dùng toàn bộ khoảng ngày của CSV. Có thể ghi đè bằng ngày cụ thể.

if CONFIG["SMOKE_TEST"]:
    CONFIG.update(SEEDS=[42], EPOCHS=2, PATIENCE=2, HIDDEN_DIM=16,
                  PSO_PARTICLES=2, PSO_ITERATIONS=1, SEARCH_EPOCHS=1, SEARCH_PATIENCE=1)
print(CONFIG)
'''

DEPENDENCIES = '''
import importlib.util
import subprocess
import sys

# Dùng PyTorch/CUDA có sẵn của Kaggle. Chỉ cài gói phụ nếu thiếu.
required = {"numpy": "numpy", "pandas": "pandas", "sklearn": "scikit-learn",
            "matplotlib": "matplotlib", "torch": "torch"}
missing = [package for module, package in required.items() if importlib.util.find_spec(module) is None]
if missing:
    raise RuntimeError(f"Thiếu thư viện nền: {missing}. Chọn image Python/GPU Kaggle hoặc cài môi trường local trước.")
optional = {}
if CONFIG.get("SOURCE") == "yfinance":
    optional["yfinance"] = "yfinance>=0.2.40"
if CONFIG.get("SOURCE") == "kaggle":
    optional["kagglehub"] = "kagglehub>=0.3.0"
if CONFIG.get("RUN_WAVELET"):
    optional["pywt"] = "PyWavelets>=1.4"
for module, package in optional.items():
    if importlib.util.find_spec(module) is None:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "--quiet", package])
'''

INTRO = '''
**Notebook độc lập:** chỉ upload file `.ipynb` này lên Kaggle, không cần clone repository hay upload `src/`.
Trong Notebook Settings, chọn **Accelerator → GPU** cho notebook train và bật **Internet** nếu tải Yahoo/cài gói.
Chạy các cell từ trên xuống, hoặc Save Version → Run All. Outputs nằm dưới `/kaggle/working/hci/`.
Notebook chỉ dùng một GPU; nếu được cấp hai GPU thì `cuda:0` được sử dụng.

**Đổi dataset:** chọn `DATASET_PRESET` trong cell config: `yahoo`, `kaggle_apple`, `kaggle_tech`, hoặc `custom`.
`SOURCE="kaggle"` nhận URL/handle qua `KAGGLE_DATASET` và tên file qua `KAGGLE_FILE` (dùng kagglehub).
**Dữ liệu offline:** Add Input dataset, đặt `SOURCE="csv"` và `CSV_PATH` tới file.
`COLUMN_MAP` ánh xạ tên gốc → Date/High/Low/Close/Volume/Open; Open là tùy chọn và không được tạo giả.
Với CSV nhiều mã, đặt `TICKER_COLUMN`; chỉ chọn một ticker mỗi lần chạy. Không nối các mã thành một chuỗi.
CSV cần tối thiểu ngày, High, Low, Close, Volume. Chỉ dùng các cột giá đã chọn, tự tính lại chỉ báo.
Không đổi ticker mà vẫn trỏ vào CSV của ticker cũ. Yahoo dùng `auto_adjust=True` nhất quán.
Nếu Yahoo bị giới hạn truy cập, dùng CSV; không tắt xác minh HTTPS.

**Lưu ý Tech Stocks:** AAPL.csv thực tế có nhiều timestamp/ngày và thiếu Open; mô tả nguồn cho biết giá đã nội suy.
Preset mặc định dừng khi gặp nhiều dòng/ngày. Chỉ để khám phá, có thể đặt `CONFIG["DAILY_POLICY"]="last_snapshot"`
sau khi áp dụng preset để giữ dòng cuối mỗi ngày UTC, không cộng volume. Cách này không hoàn tác nội suy sẵn có.
Không xem dữ liệu đã nội suy là dữ liệu thô chống rò rỉ, và không dùng các cột chỉ báo/vĩ mô sẵn có trong thí nghiệm này.

Tham khảo: [Kaggle Notebooks](https://www.kaggle.com/docs/notebooks),
[GPU trên Kaggle (NVIDIA)](https://docs.nvidia.com/datascience/deployment/latest/platforms/kaggle/),
[yfinance](https://ranaroussi.github.io/yfinance/).
[kagglehub](https://github.com/Kaggle/kagglehub),
[Apple dataset](https://www.kaggle.com/datasets/tarunpaparaju/apple-aapl-historical-stock-data),
[Tech Stocks dataset](https://www.kaggle.com/datasets/sinankr/tech-stocks-dataset).
'''

SPLIT_NOTE = '''
### Đặc trưng và chia tập theo thời gian
Chỉ báo dùng lịch sử đến phiên hiện tại; bỏ 19 dòng khởi động, không `bfill`.
Scaler chỉ fit trên train. Mỗi nhãn là Close của phiên kế tiếp, đầu vào kết thúc trước nhãn.
Val/test được dùng lịch sử cuối tập trước để giữ đủ mẫu, không đưa nhãn tương lai vào đầu vào.
Không shuffle khi chia tập. Shuffle các cửa sổ train trong DataLoader là được phép.
Notebook chỉ hỗ trợ `HORIZON=1`; thay đổi khác bị từ chối để tránh sai shape/ý nghĩa nhãn.
'''

TRAIN_NOTE = '''
### Huấn luyện và tối ưu
LSTM cố định và LSTM–PSO dùng cùng dữ liệu, scaler, số epoch tối đa, early stopping và các seed cuối cùng.
ANFIS dùng cùng split/scaler/ngày test/seed, nhưng chỉ lấy vector đặc trưng của phiên cuối trong cửa sổ.
Đây là Takagi–Sugeno bậc một, học cả hàm thuộc lẫn hệ số tuyến tính bằng Adam;
không ghép LSTM và không dùng PSO cho ANFIS. `ANFIS_NUM_RULES` và `ANFIS_LEARNING_RATE`
được đặt trước khi train. Không chọn chúng dựa trên test. Trọng số luật được chuẩn hóa trong miền log
để tránh tích độ thuộc bị tiến sát 0; độ rộng và độ dốc hàm thuộc luôn dương.
PSO tối ưu hidden size, dropout và learning rate trên **toàn bộ train/validation**, mỗi ứng viên dùng cùng search seed.
Test không được dùng để chọn tham số hoặc early stopping. Ngân sách mặc định: tối đa 36 lần đánh giá ứng viên,
8 epoch/lần, rồi 3 seed cho mỗi phương pháp. Bật `SMOKE_TEST=True` để thử nhanh trước.

PSO có thêm ngân sách tính toán so với LSTM cố định; báo cáo thời gian tìm kiếm riêng, không tuyên bố so sánh ngang chi phí.
`search_seconds_shared` là chi phí tìm kiếm dùng chung giữa các seed, không cộng lại cho từng seed.
Trung bình/std qua seed chỉ phản ánh biến thiên khởi tạo mô hình, không phải bằng chứng tổng quát hóa qua mọi giai đoạn thị trường.

Mỗi mô hình lưu `best.pt` ngay khi validation cải thiện. Checkpoint có kiến trúc, tham số, seed và trọng số;
scaler, thứ tự đặc trưng, split, dữ liệu đầu vào, nguồn dữ liệu và versions được lưu bên cạnh trong cùng thư mục run.
Để đánh giá lại, giữ nguyên **cả thư mục run**, không chỉ một file `.pt`.
'''

BACKTEST_NOTE = '''
### Backtest minh họa
Long khi dự đoán Close kế tiếp cao hơn Close hiện tại, ngược lại giữ tiền mặt.
Tính phí khi đổi vị thế, vào lệnh đầu kỳ và thanh lý cuối kỳ; drawdown có tính vốn ban đầu.
Đây là mô phỏng close-to-close giả định khớp tại Close đã quan sát, chưa mô hình hóa độ trễ,
spread, slippage hay việc chỉ biết giá đóng cửa sau khi phiên kết thúc. Giá điều chỉnh và chỉ số
`^GSPC` phục vụ nghiên cứu dự báo, không đại diện một danh mục có thể khớp lệnh trực tiếp.
Không dùng bảng lợi nhuận này làm bằng chứng hệ thống sẵn sàng giao dịch.
'''


def section(name, title):
    return [md(title), code(SECTIONS[name], "helpers-" + name)]


def start(title, wavelet=False):
    config = CONFIG.replace('"RUN_WAVELET": False', '"RUN_WAVELET": True') if wavelet else CONFIG
    return [md(f"# {title}\n\n{INTRO}"), md("## 1. Cấu hình — sửa cell này trước khi Run All"),
            code(config, "config"), md("## 2. Kiểm tra thư viện"), code(DEPENDENCIES, "dependencies"),
            *section("imports", "## 3. Tiện ích và môi trường")]


def data_cells(label):
    return [*section("data", "## Tải hoặc đọc dữ liệu và kiểm tra chất lượng"),
            code(f'RUN_DIR = new_run(CONFIG, "{label}")\nraw = load_raw(CONFIG, RUN_DIR)\ndisplay(raw.head())', "load-data")]


def prepared_cells():
    return [*section("features", SPLIT_NOTE), code("data, meta = prepare_data(raw, CONFIG, RUN_DIR)", "prepare")]


def end_note():
    return md('''
### Lưu kết quả
Sau khi chạy xong, dùng **Save Version** và giữ output của phiên chạy.
Tải cả thư mục được in ở `RUN_DIR`, hoặc Add Input output này trong notebook 05.
Các lần chạy mới tạo thư mục riêng để tránh ghi đè thí nghiệm trước.
''')


def write(name, cells):
    for index, cell in enumerate(cells):
        cell["id"] = f"cell-{index:03d}"
    notebook = {"cells": cells, "metadata": {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3.10"}}, "nbformat": 4, "nbformat_minor": 5}
    (HERE / name).write_text(json.dumps(notebook, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    print(name, len(cells), "cells")


cells = start("01 — Thu thập dữ liệu Yahoo Finance và EDA") + data_cells("data")
cells += [md("## Khám phá dữ liệu\nCác hình mô tả toàn bộ lịch sử; không dùng test để chọn tham số mô hình."), code('''
display(raw.describe())
print("Missing:", raw.isna().sum().to_dict(), "Duplicate dates:", raw.index.duplicated().sum())
returns = raw.Close.pct_change().dropna()
fig, axes = plt.subplots(3, 1, figsize=(13, 10))
raw.Close.plot(ax=axes[0], title=f"{CONFIG['TICKER']} — Close")
raw.Volume.plot(ax=axes[1], title="Volume")
returns.hist(bins=60, ax=axes[2])
axes[2].set_title("Phân phối lợi suất phiên")
fig.tight_layout()
fig.savefig(RUN_DIR / "eda.png", dpi=160)
plt.show()
plt.close(fig)
raw.describe().to_csv(RUN_DIR / "describe.csv")
print("CSV dùng cho các notebook khác:", list(RUN_DIR.glob("*.csv")))
''', "eda"), end_note()]
write("01_eda_and_data_collection.ipynb", cells)

cells = start("02 — Đặc trưng, chia tập và Wavelet theo cửa sổ", wavelet=True)
cells += data_cells("features") + prepared_cells()
cells += [md('''
## Kiểm tra đầu vào và minh họa Wavelet
Wavelet chỉ lọc kênh Close **bên trong từng cửa sổ 60 phiên đã quan sát**.
Không lọc toàn chuỗi trước khi chia tập; không sửa target thật.
Cùng một ngày có thể có giá lọc khác khi nằm trong các cửa sổ khác nhau; mỗi dự báo chỉ dùng cửa sổ của nó.
'''), code('''
wavelet_data = wavelet_inputs(data, meta, CONFIG)
np.savez_compressed(RUN_DIR / "prepared_wavelet.npz", **wavelet_data)
column = meta["feature_columns"].index("Close")
fig, axes = plt.subplots(1, 2, figsize=(14, 4))
axes[0].plot(data["X_train"][0, :, column], label="Close scaled")
axes[0].plot(wavelet_data["X_train"][0, :, column], label="Wavelet Close scaled")
axes[0].legend()
axes[0].set_title("Chỉ dùng dữ liệu trong cửa sổ đầu vào")
for split in ["train", "val", "test"]:
    axes[1].plot(pd.to_datetime(data[f"dates_{split}"]), data[f"actual_{split}"], label=split)
axes[1].legend()
axes[1].set_title("Ngày của target theo split")
fig.tight_layout()
fig.savefig(RUN_DIR / "features_and_splits.png", dpi=160)
plt.show()
plt.close(fig)
assert np.array_equal(data["y_test"], wavelet_data["y_test"])
print("Feature columns:", meta["feature_columns"])
''', "wavelet-demo"), end_note()]
write("02_feature_engineering_and_decomposition.ipynb", cells)

for name, title, label, include_pso in [
    ("03_baseline_models.ipynb", "03 — LSTM thuần và baseline giá không đổi", "baseline", False),
    ("04_hybrid_model_experiment.ipynb", "04 — So sánh LSTM, LSTM–PSO và ANFIS trên GPU", "comparison", True),
]:
    cells = start(title)
    cells += [code("DEVICE = select_device(CONFIG['REQUIRE_GPU'])", "device")]
    cells += data_cells(label) + prepared_cells()
    cells += section("model", "## Kiến trúc LSTM, ANFIS và DataLoader")
    cells += section("training", TRAIN_NOTE)
    if include_pso:
        cells += [md('''
**Notebook chính cho đề tài.** Chạy độc lập từ cell đầu sẽ tạo cả baseline và hybrid trong cùng thí nghiệm.
Không cần chạy notebook 01–03 trước. Mặc định chạy LSTM, LSTM–PSO và ANFIS.
Đặt `RUN_ANFIS=False` nếu muốn tắt ANFIS. `RUN_WAVELET=True` thêm Wavelet–LSTM
và Wavelet–LSTM–PSO (tổng cộng năm model khi bật cả ANFIS).
So sánh ANFIS với LSTM đánh giá hai cách học khác nhau, không phải ablation cùng kiến trúc.
GA và GWO chưa được tích hợp.
''')]
    else:
        cells += [md("Notebook này train LSTM thuần; baseline Naive dự đoán giá phiên kế tiếp bằng giá phiên hiện tại. Để có so sánh hybrid cùng split/nguồn dữ liệu, chạy notebook 04.")]
        cells += [code('CONFIG["RUN_WAVELET"] = False\nCONFIG["RUN_ANFIS"] = False\nsave_json(RUN_DIR / "config.json", CONFIG)', "baseline-only")]
    cells += [code(f"experiments, variants = run_experiments(data, meta, CONFIG, DEVICE, RUN_DIR, include_pso={include_pso})", "train")]
    cells += section("evaluation", "## Đánh giá test sau khi hoàn tất huấn luyện\nRMSE/MAE ở thang giá thật. DA so sánh hướng dự đoán với Close ngay trước target, kể cả mẫu đầu tiên.")
    cells += [code('''
metrics, predictions = evaluate_experiments(experiments, variants, meta, RUN_DIR, RUN_DIR / "evaluation", DEVICE)
plot_results(metrics, predictions, RUN_DIR / "evaluation")
display(metrics)
''', "evaluate"), md("## Loss và diễn biến PSO"), code('''
fig, ax = plt.subplots(figsize=(11, 5))
for record in experiments:
    history = pd.read_csv((RUN_DIR / record["checkpoint"]).parent / "history.csv")
    ax.plot(history.epoch, history.val_mse, label=record["run_id"])
ax.set(title="Validation loss", xlabel="Epoch", ylabel="Scaled MSE")
ax.legend(fontsize=8)
fig.tight_layout()
fig.savefig(RUN_DIR / "validation_curves.png", dpi=160)
plt.show()
plt.close(fig)
for path in RUN_DIR.glob("*/pso_history.csv"):
    trace = pd.read_csv(path)
    trace.best_val_mse.plot(title=f"PSO convergence — {path.parent.name}", xlabel="Lần đánh giá", ylabel="Best validation MSE")
    plt.tight_layout()
    plt.savefig(path.parent / "pso_convergence.png", dpi=160)
    plt.show()
    plt.close()
''', "curves"), end_note()]
    write(name, cells)

cells = [md("# 05 — Tải checkpoint, đánh giá lại và backtest\n\nNotebook độc lập, không train hoặc tải lại Yahoo. Add Input **toàn bộ output run** của notebook 03/04 rồi đặt `RUN_PATH`. Dữ liệu test và scaler đã lưu bảo đảm đánh giá đúng tập cũ."),
         md("## 1. Chọn thư mục run"), code('''
from pathlib import Path

RUN_PATH = ""  # Ví dụ /kaggle/input/my-training-output/hci/GSPC_comparison_.../
CONFIG = {"SOURCE": "csv", "RUN_WAVELET": False}
EVAL_OUTPUT_ROOT = "/kaggle/working/hci_evaluation" if Path("/kaggle/working").exists() else "kaggle_evaluation"
COMMISSION = 0.001
''', "config"), code(DEPENDENCIES, "dependencies")]
cells += section("imports", "## Tiện ích")
cells += section("features", "## Đọc scaler đã lưu\nKhông fit lại scaler và không thay đổi split.")
cells += section("model", "## Kiến trúc dùng để khôi phục checkpoint")
cells += section("evaluation", "## Hàm đánh giá và backtest")
cells += [code('''
if not RUN_PATH:
    candidates = [p.parent for p in Path("/kaggle/input").rglob("experiments.json")]
    if len(candidates) != 1:
        raise ValueError(f"Tìm thấy {len(candidates)} run. Đặt RUN_PATH trỏ tới thư mục chứa experiments.json.")
    RUN_PATH = str(candidates[0])
SOURCE_RUN = Path(RUN_PATH)
experiments = json.loads((SOURCE_RUN / "experiments.json").read_text(encoding="utf-8"))
meta = json.loads((SOURCE_RUN / "preprocessing.json").read_text(encoding="utf-8"))
variants = {}
for variant in {"raw"} | {record["variant"] for record in experiments}:
    name = "prepared_data.npz" if variant == "raw" else "prepared_wavelet.npz"
    with np.load(SOURCE_RUN / name, allow_pickle=False) as archive:
        variants[variant] = {key: archive[key] for key in archive.files}
DEVICE = select_device(False)
OUT = Path(EVAL_OUTPUT_ROOT) / datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ")
OUT.mkdir(parents=True)
save_json(OUT / "evaluation_source.json", {"run_path": str(SOURCE_RUN), "commission": COMMISSION})
metrics, predictions = evaluate_experiments(experiments, variants, meta, SOURCE_RUN, OUT, DEVICE)
plot_results(metrics, predictions, OUT)
''', "reload-evaluate"), md(BACKTEST_NOTE), code('''
backtest_table = backtest_predictions(predictions, COMMISSION, OUT)
display(backtest_table)
print("Saved evaluation:", OUT)
''', "backtest"), md('''
## Đọc bảng so sánh/ablation
`metrics_summary.csv` chứa trung bình và độ lệch chuẩn qua các seed.
Nếu run gốc bật Wavelet, so sánh LSTM ↔ Wavelet–LSTM để quan sát ảnh hưởng Wavelet,
và LSTM ↔ LSTM–PSO để quan sát ảnh hưởng tuning; báo cáo cả bốn cấu hình và chi phí tìm kiếm.
Nếu run gốc chỉ có LSTM, notebook này không tự sinh kết quả hybrid.
Nếu run có ANFIS, checkpoint nhận diện model_type và khôi phục đúng số luật; bảng kết quả gồm ANFIS.
Không chọn lại siêu tham số từ bảng test. Muốn đánh giá độ bền, thiết kế thêm các khoảng thời gian/ticker
hoặc walk-forward với tập test mới trước khi chạy thí nghiệm tiếp theo.
''')]
write("05_backtesting_and_ablation_study.ipynb", cells)
