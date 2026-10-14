import torch
import torch.nn as nn

class DirectionalLoss(nn.Module):
    """
    Hàm mất mát kết hợp MSE và phạt sai chiều hướng biến động giá (Directional Penalty).
    Nếu mô hình dự đoán tăng mà giá thực tế giảm (hoặc ngược lại), sẽ bị phạt thêm trọng số alpha.
    """
    def __init__(self, alpha: float = 0.5):
        super(DirectionalLoss, self).__init__()
        self.mse = nn.MSELoss()
        self.alpha = alpha

    def forward(self, y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
        base_loss = self.mse(y_pred, y_true)
        # Phạt khi tích của hai giá trị mang dấu âm (sai xu hướng)
        direction_penalty = torch.mean(torch.relu(- (y_pred * y_true)))
        return base_loss + self.alpha * direction_penalty
