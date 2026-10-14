import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from typing import Dict, Any

try:
    import xgboost as xgb
except ImportError:
    xgb = None

class BaselineModels:
    """Các mô hình đối chuẩn (Baselines) truyền thống."""

    @staticmethod
    def train_random_forest(X_train: np.ndarray, y_train: np.ndarray, **kwargs) -> RandomForestRegressor:
        # Làm phẳng chiều sequence thành vector 1D
        X_flat = X_train.reshape(X_train.shape[0], -1)
        model = RandomForestRegressor(n_estimators=kwargs.get("n_estimators", 100), random_state=42)
        model.fit(X_flat, y_train)
        return model

    @staticmethod
    def train_svr(X_train: np.ndarray, y_train: np.ndarray, **kwargs) -> SVR:
        X_flat = X_train.reshape(X_train.shape[0], -1)
        model = SVR(kernel=kwargs.get("kernel", "rbf"), C=kwargs.get("C", 1.0), epsilon=0.1)
        model.fit(X_flat, y_train)
        return model

    @staticmethod
    def train_xgboost(X_train: np.ndarray, y_train: np.ndarray, **kwargs):
        if xgb is None:
            raise ImportError("Vui lòng cài đặt xgboost qua: pip install xgboost")
        X_flat = X_train.reshape(X_train.shape[0], -1)
        model = xgb.XGBRegressor(n_estimators=kwargs.get("n_estimators", 100), random_state=42)
        model.fit(X_flat, y_train)
        return model
