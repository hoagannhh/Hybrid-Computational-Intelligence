import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from typing import Dict, List
from src.utils.logger import setup_logger

logger = setup_logger(__name__)

class ModelTrainer:
    """Quản lý vòng lặp huấn luyện, kiểm chuẩn và lưu checkpoint mô hình PyTorch."""
    def __init__(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        criterion: nn.Module,
        device: torch.device,
        save_dir: str = "saved_models"
    ):
        self.model = model
        self.optimizer = optimizer
        self.criterion = criterion
        self.device = device
        self.save_dir = save_dir
        os.makedirs(save_dir, exist_ok=True)

    def train_epoch(self, dataloader: DataLoader) -> float:
        self.model.train()
        total_loss = 0.0
        for X_batch, y_batch in dataloader:
            X_batch, y_batch = X_batch.to(self.device), y_batch.to(self.device)

            self.optimizer.zero_grad()
            preds = self.model(X_batch)
            loss = self.criterion(preds, y_batch)
            loss.backward()
            self.optimizer.step()

            total_loss += loss.item() * len(X_batch)
        return total_loss / len(dataloader.dataset)

    def evaluate(self, dataloader: DataLoader) -> float:
        self.model.eval()
        total_loss = 0.0
        with torch.no_grad():
            for X_batch, y_batch in dataloader:
                X_batch, y_batch = X_batch.to(self.device), y_batch.to(self.device)
                preds = self.model(X_batch)
                loss = self.criterion(preds, y_batch)
                total_loss += loss.item() * len(X_batch)
        return total_loss / len(dataloader.dataset)

    def fit(self, train_loader: DataLoader, val_loader: DataLoader, epochs: int = 50, patience: int = 10) -> Dict[str, List[float]]:
        history = {"train_loss": [], "val_loss": []}
        best_val_loss = float("inf")
        patience_counter = 0

        logger.info(f"Bắt đầu huấn luyện mô hình trong {epochs} epochs (Early stopping patience: {patience})...")

        for epoch in range(1, epochs + 1):
            train_loss = self.train_epoch(train_loader)
            val_loss = self.evaluate(val_loader)

            history["train_loss"].append(train_loss)
            history["val_loss"].append(val_loss)

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                best_model_path = os.path.join(self.save_dir, "best_model.pt")
                torch.save(self.model.state_dict(), best_model_path)
            else:
                patience_counter += 1

            if epoch % 5 == 0 or epoch == epochs or patience_counter == 0:
                logger.info(f"Epoch [{epoch:03d}/{epochs:03d}] - Train Loss: {train_loss:.6f} | Val Loss: {val_loss:.6f}")

            if patience_counter >= patience:
                logger.info(f"Kích hoạt Early Stopping tại epoch {epoch}. Val loss tốt nhất: {best_val_loss:.6f}")
                break

        # Tải lại trọng số tốt nhất
        best_model_path = os.path.join(self.save_dir, "best_model.pt")
        if os.path.exists(best_model_path):
            self.model.load_state_dict(torch.load(best_model_path, map_location=self.device))
        return history
