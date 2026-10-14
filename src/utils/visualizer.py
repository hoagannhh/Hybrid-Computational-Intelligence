import os
import matplotlib.pyplot as plt
import numpy as np
from typing import List, Dict

def plot_predictions(
    actual: np.ndarray,
    predicted: np.ndarray,
    title: str = "Thực tế vs Dự đoán",
    save_path: str = "reports/figures/prediction_comparison.png"
):
    """Vẽ biểu đồ so sánh giá trị thực tế và giá trị mô hình dự đoán."""
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.figure(figsize=(12, 6))
    plt.plot(actual, label="Giá thực tế (Actual)", color="black", alpha=0.8, linewidth=1.5)
    plt.plot(predicted, label="Mô hình lai dự đoán (Predicted)", color="crimson", linestyle="--", linewidth=1.5)
    plt.title(title, fontsize=14, fontweight="bold")
    plt.xlabel("Thời gian (Phiên giao dịch)", fontsize=12)
    plt.ylabel("Giá đóng cửa", fontsize=12)
    plt.legend(fontsize=11)
    plt.grid(True, linestyle=":", alpha=0.6)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()

def plot_training_loss(
    history: Dict[str, List[float]],
    save_path: str = "reports/figures/loss_curve.png"
):
    """Vẽ đường cong mất mát quá trình huấn luyện (Train Loss vs Val Loss)."""
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.figure(figsize=(10, 5))
    plt.plot(history["train_loss"], label="Train Loss", color="royalblue")
    plt.plot(history["val_loss"], label="Validation Loss", color="darkorange")
    plt.title("Đường cong huấn luyện (Training & Validation Loss)", fontsize=14, fontweight="bold")
    plt.xlabel("Epoch", fontsize=12)
    plt.ylabel("Loss (MSE / Huber)", fontsize=12)
    plt.legend(fontsize=11)
    plt.grid(True, linestyle=":", alpha=0.6)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()
