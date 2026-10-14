import numpy as np
import pandas as pd
from typing import Tuple, List

try:
    import pywt
except ImportError:
    pywt = None

def wavelet_denoising(series: np.ndarray, wavelet: str = "db4", level: int = 1) -> np.ndarray:
    """
    Khử nhiễu chuỗi thời gian giá chứng khoán bằng biến đổi Wavelet (Discrete Wavelet Transform - DWT).
    Thường dùng ngưỡng mềm (Soft Thresholding) trên các hệ số chi tiết (Detail coefficients).
    """
    if pywt is None:
        raise ImportError("Vui lòng cài đặt PyWavelets qua: pip install PyWavelets")

    # Phân rã wavelet
    coeff = pywt.wavedec(series, wavelet, mode="per")
    # Ước lượng độ lệch chuẩn nhiễu bằng trung vị độ lệch tuyệt đối (MAD)
    sigma = (1 / 0.6745) * np.median(np.abs(coeff[-level] - np.median(coeff[-level])))
    # Ngưỡng Donoho-Johnstone (Universal threshold)
    threshold = sigma * np.sqrt(2 * np.log(len(series)))

    # Áp dụng ngưỡng mềm cho các hệ số chi tiết
    coeff[1:] = [pywt.threshold(c, value=threshold, mode="soft") for c in coeff[1:]]

    # Tái tạo tín hiệu đã khử nhiễu
    reconstructed = pywt.waverec(coeff, wavelet, mode="per")
    # Cắt cho đúng độ dài ban đầu
    return reconstructed[:len(series)]

def wavelet_decomposition_features(series: np.ndarray, wavelet: str = "db4", level: int = 3) -> pd.DataFrame:
    """
    Phân rã chuỗi giá thành 1 thành phần xấp xỉ (Approximation - xu hướng dài hạn)
    và nhiều thành phần chi tiết (Details - biến động tần số cao).
    """
    if pywt is None:
        raise ImportError("Vui lòng cài đặt PyWavelets qua: pip install PyWavelets")

    coeffs = pywt.wavedec(series, wavelet, level=level, mode="per")
    feature_dict = {}

    # Hệ số xấp xỉ A
    feature_dict[f"Wavelet_A{level}"] = pywt.upcoef("a", coeffs[0], wavelet, level=level)[:len(series)]

    # Các hệ số chi tiết D
    for i, d in enumerate(coeffs[1:], 1):
        curr_level = level - i + 1
        feature_dict[f"Wavelet_D{curr_level}"] = pywt.upcoef("d", d, wavelet, level=curr_level)[:len(series)]

    return pd.DataFrame(feature_dict)
