# Thư mục Notebooks phục vụ Nghiên cứu & Thử nghiệm

Thư mục này dành cho việc nghiên cứu, trực quan hóa và thử nghiệm nhanh theo từng giai đoạn của đề tài:

1. `01_eda_and_data_collection.ipynb`:
   - Khám phá dữ liệu (EDA), phân tích phân phối giá, khối lượng giao dịch.
   - Kiểm định tính dừng (Stationarity test - ADF Test), tự tương quan (ACF, PACF).

2. `02_feature_engineering_and_decomposition.ipynb`:
   - Tính toán và vẽ các chỉ báo kỹ thuật (RSI, MACD, Bollinger Bands, ATR).
   - Trực quan hóa biến đổi Wavelet (DWT) hoặc EMD/CEEMDAN để khử nhiễu và tách chuỗi thành xu hướng (Trend) & dao động (Fluctuations).

3. `03_baseline_models.ipynb`:
   - Huấn luyện và ghi nhận kết quả các mô hình cơ sở: ARIMA, Support Vector Regression (SVR), Random Forest, XGBoost.
   - Đo lường RMSE, MAE, MAPE làm mốc so sánh (benchmarking).

4. `04_hybrid_model_experiment.ipynb`:
   - Thử nghiệm các kiến trúc lai: Wavelet-LSTM-PSO, ANFIS-GA, CNN-LSTM-Attention.
   - Theo dõi quá trình hội tụ của thuật toán tối ưu hóa bầy đàn / di truyền (PSO/GA).

5. `05_backtesting_and_ablation_study.ipynb`:
   - Kiểm thử chiến lược giao dịch tự động (Backtesting), tính Sharpe Ratio, Maximum Drawdown.
   - Nghiên cứu triệt tiêu (Ablation Study): Đánh giá đóng góp của từng thành phần (Bỏ Wavelet, bỏ PSO, bỏ Fuzzy xem kết quả thay đổi như thế nào).
