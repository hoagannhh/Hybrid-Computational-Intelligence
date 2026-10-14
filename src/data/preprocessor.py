import numpy as np
import pandas as pd
from typing import Tuple, Dict, Any
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from src.utils.logger import setup_logger

logger = setup_logger(__name__)

class TimeSeriesPreprocessor:
    """
    Tiền xử lý chuỗi thời gian chứng khoán:
    - Xử lý giá trị khuyết thiếu (missing values)
    - Chia tập Train/Validation/Test theo trục thời gian (chống lookahead bias)
    - Chuẩn hóa dữ liệu (fit chỉ trên tập Train để tránh rò rỉ dữ liệu - Data Leakage)
    - Tạo cấu trúc cửa sổ trượt (Sliding Window: Lookback -> Target)
    """

    def __init__(self, scaler_type: str = "minmax", seq_length: int = 60, horizon: int = 1):
        self.seq_length = seq_length
        self.horizon = horizon
        self.scaler_type = scaler_type
        self.feature_scaler = MinMaxScaler(feature_range=(0, 1)) if scaler_type == "minmax" else StandardScaler()
        self.target_scaler = MinMaxScaler(feature_range=(0, 1)) if scaler_type == "minmax" else StandardScaler()

    def clean_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Làm sạch dữ liệu: forward fill, backward fill và drop NA."""
        cleaned_df = df.copy()
        cleaned_df.ffill(inplace=True)
        cleaned_df.bfill(inplace=True)
        cleaned_df.dropna(inplace=True)
        return cleaned_df

    def split_data(
        self,
        df: pd.DataFrame,
        train_ratio: float = 0.7,
        val_ratio: float = 0.15,
        test_ratio: float = 0.15
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """Phân chia chuỗi thời gian tuần tự theo thời gian."""
        n = len(df)
        train_end = int(n * train_ratio)
        val_end = int(n * (train_ratio + val_ratio))

        train_df = df.iloc[:train_end]
        val_df = df.iloc[train_end:val_end]
        test_df = df.iloc[val_end:]

        logger.info(f"Kích thước tập dữ liệu -> Train: {len(train_df)}, Val: {len(val_df)}, Test: {len(test_df)}")
        return train_df, val_df, test_df

    def create_sequences(
        self,
        features: np.ndarray,
        target: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Tạo dữ liệu dạng chuỗi với cửa sổ trượt.
        X shape: (num_samples, seq_length, num_features)
        y shape: (num_samples, horizon) hoặc (num_samples,)
        """
        X, y = [], []
        total_len = len(features)
        
        for i in range(self.seq_length, total_len - self.horizon + 1):
            X.append(features[i - self.seq_length:i, :])
            if self.horizon == 1:
                y.append(target[i + self.horizon - 1])
            else:
                y.append(target[i:i + self.horizon])

        return np.array(X), np.array(y)

    def fit_transform_pipeline(
        self,
        train_df: pd.DataFrame,
        val_df: pd.DataFrame,
        test_df: pd.DataFrame,
        target_col: str = "Close"
    ) -> Dict[str, Any]:
        """
        Huấn luyện bộ chuẩn hóa trên Train và biến đổi cả 3 tập.
        """
        feature_cols = [c for c in train_df.columns]

        # Fit scalers
        train_features_scaled = self.feature_scaler.fit_transform(train_df[feature_cols].values)
        val_features_scaled = self.feature_scaler.transform(val_df[feature_cols].values)
        test_features_scaled = self.feature_scaler.transform(test_df[feature_cols].values)

        train_target_scaled = self.target_scaler.fit_transform(train_df[[target_col]].values).flatten()
        val_target_scaled = self.target_scaler.transform(val_df[[target_col]].values).flatten()
        test_target_scaled = self.target_scaler.transform(test_df[[target_col]].values).flatten()

        # Tạo chuỗi trượt
        X_train, y_train = self.create_sequences(train_features_scaled, train_target_scaled)
        X_val, y_val = self.create_sequences(val_features_scaled, val_target_scaled)
        X_test, y_test = self.create_sequences(test_features_scaled, test_target_scaled)

        return {
            "X_train": X_train, "y_train": y_train,
            "X_val": X_val, "y_val": y_val,
            "X_test": X_test, "y_test": y_test,
            "feature_scaler": self.feature_scaler,
            "target_scaler": self.target_scaler,
            "feature_cols": feature_cols
        }
