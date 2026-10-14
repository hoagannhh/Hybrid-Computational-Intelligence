import numpy as np
from typing import Dict

def calculate_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """
    Tính toán các chỉ số đánh giá độ chính xác dự báo chuỗi thời gian chứng khoán:
    - RMSE: Root Mean Squared Error
    - MAE: Mean Absolute Error
    - MAPE: Mean Absolute Percentage Error (%)
    - R2: Coefficient of Determination
    - DA: Directional Accuracy (%) - Tỷ lệ dự đoán đúng chiều tăng/giảm giá
    """
    y_true = np.asarray(y_true).flatten()
    y_pred = np.asarray(y_pred).flatten()

    mae = np.mean(np.abs(y_true - y_pred))
    mse = np.mean((y_true - y_pred) ** 2)
    rmse = np.sqrt(mse)

    # Tránh chia cho 0 trong MAPE
    non_zero = y_true != 0
    mape = np.mean(np.abs((y_true[non_zero] - y_pred[non_zero]) / y_true[non_zero])) * 100.0

    # R2 Score
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    r2 = 1.0 - (ss_res / (ss_tot + 1e-9))

    # Directional Accuracy (DA)
    # Đoán chiều biến động so với ngày hôm trước
    if len(y_true) > 1:
        actual_direction = np.sign(y_true[1:] - y_true[:-1])
        predicted_direction = np.sign(y_pred[1:] - y_true[:-1])
        da = np.mean(actual_direction == predicted_direction) * 100.0
    else:
        da = 0.0

    return {
        "RMSE": float(rmse),
        "MAE": float(mae),
        "MAPE(%)": float(mape),
        "R2": float(r2),
        "Directional_Accuracy(%)": float(da)
    }
