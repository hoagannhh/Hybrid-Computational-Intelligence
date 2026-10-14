import numpy as np
import torch
import torch.nn as nn
from typing import Dict, Any, Tuple
from src.features.signal_decomposition import wavelet_denoising
from src.models.neural.lstm_model import StockLSTM
from src.models.optimizers.metaheuristics import ParticleSwarmOptimizer
from src.utils.logger import setup_logger

logger = setup_logger(__name__)

class WaveletLSTMHybrid:
    """
    Mô hình lai kết hợp Trí tuệ tính toán:
    1. Tiền xử lý / Khử nhiễu: Biến đổi Wavelet (DWT)
    2. Dự đoán phi tuyến tính: Mạng nơ-ron hồi quy LSTM
    3. Tối ưu hóa siêu tham số: Giải thuật bầy đàn (PSO)
    """
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.device = torch.device(config["project"].get("device", "cpu"))
        self.best_params = None
        self.model = None

    def preprocess_with_wavelet(self, data_series: np.ndarray) -> np.ndarray:
        """Khử nhiễu chuỗi giá đóng cửa bằng Wavelet."""
        wavelet_name = self.config["features"]["decomposition"].get("wavelet_name", "db4")
        return wavelet_denoising(data_series, wavelet=wavelet_name, level=1)

    def optimize_hyperparameters(self, X_train, y_train, X_val, y_val, input_dim: int):
        """
        Dùng PSO để tìm:
        - param 0: hidden_dim (trong khoảng [16, 128])
        - param 1: dropout (trong khoảng [0.0, 0.5])
        - param 2: learning_rate (trong khoảng [0.0001, 0.01])
        """
        def objective(params: np.ndarray) -> float:
            h_dim = int(np.clip(params[0], 16, 128))
            drop = float(np.clip(params[1], 0.0, 0.5))
            lr = float(np.clip(params[2], 0.0001, 0.01))

            model = StockLSTM(input_dim=input_dim, hidden_dim=h_dim, dropout=drop).to(self.device)
            criterion = nn.MSELoss()
            optimizer = torch.optim.Adam(model.parameters(), lr=lr)

            # Huấn luyện nhanh vài epochs để đánh giá hàm mất mát trên tập Val
            inputs_t = torch.tensor(X_train[:200], dtype=torch.float32).to(self.device)
            targets_t = torch.tensor(y_train[:200], dtype=torch.float32).to(self.device)

            model.train()
            for _ in range(5):
                optimizer.zero_grad()
                preds = model(inputs_t)
                loss = criterion(preds, targets_t)
                loss.backward()
                optimizer.step()

            # Đánh giá trên Val
            model.eval()
            with torch.no_grad():
                val_in = torch.tensor(X_val[:100], dtype=torch.float32).to(self.device)
                val_tar = torch.tensor(y_val[:100], dtype=torch.float32).to(self.device)
                val_loss = criterion(model(val_in), val_tar).item()

            return val_loss

        bounds = (
            np.array([16.0, 0.0, 0.0001]),
            np.array([128.0, 0.5, 0.01])
        )

        pso = ParticleSwarmOptimizer(
            cost_func=objective,
            dim=3,
            bounds=bounds,
            num_particles=self.config["models"]["optimizer"].get("population_size", 15),
            max_iter=self.config["models"]["optimizer"].get("max_iter", 10)
        )

        best_vector, best_cost = pso.optimize()
        self.best_params = {
            "hidden_dim": int(best_vector[0]),
            "dropout": float(best_vector[1]),
            "learning_rate": float(best_vector[2])
        }
        logger.info(f"Tham số tối ưu tìm được bởi PSO: {self.best_params} (Loss: {best_cost:.6f})")
        return self.best_params

    def build_model(self, input_dim: int, hidden_dim: int = 64, dropout: float = 0.2) -> nn.Module:
        """Xây dựng mô hình với các tham số đã được tối ưu."""
        self.model = StockLSTM(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            dropout=dropout
        ).to(self.device)
        return self.model
