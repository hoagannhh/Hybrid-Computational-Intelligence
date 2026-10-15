"""Denoise Close within each past input window; preserve original targets."""
import numpy as np


def wavelet_inputs(data, meta, config):
    import pywt
    result = dict(data)
    close_index = meta["feature_columns"].index("Close")
    name, level = config["WAVELET"], config["WAVELET_LEVEL"]
    maximum = pywt.dwt_max_level(meta["lookback"], pywt.Wavelet(name).dec_len)
    if not 1 <= level <= maximum:
        raise ValueError(f"WAVELET_LEVEL cần trong [1, {maximum}] cho lookback hiện tại.")
    for split in ["train", "val", "test"]:
        windows = data[f"X_{split}"].copy()
        for window in windows:
            series = window[:, close_index]
            coeffs = pywt.wavedec(series, name, level=level, mode="symmetric")
            sigma = np.median(np.abs(coeffs[-1] - np.median(coeffs[-1]))) / 0.6745
            threshold = sigma * np.sqrt(2 * np.log(len(series)))
            if threshold > 0:
                coeffs[1:] = [pywt.threshold(c, threshold, mode="soft") for c in coeffs[1:]]
            window[:, close_index] = pywt.waverec(coeffs, name, mode="symmetric")[:len(series)]
        assert np.isfinite(windows).all()
        result[f"X_{split}"] = windows
    # Target thật và các đặc trưng khác giữ nguyên; không lọc toàn chuỗi trước khi split.
    return result
