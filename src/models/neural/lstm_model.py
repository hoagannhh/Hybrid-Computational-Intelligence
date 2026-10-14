import torch
import torch.nn as nn

class StockLSTM(nn.Module):
    """
    Mô hình mạng nơ-ron hồi quy LSTM (hoặc BiLSTM) với tùy chọn Attention.
    """
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 64,
        num_layers: int = 2,
        output_dim: int = 1,
        dropout: float = 0.2,
        bidirectional: bool = False
    ):
        super(StockLSTM, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        self.num_directions = 2 if bidirectional else 1

        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
            bidirectional=bidirectional
        )

        self.fc_layers = nn.Sequential(
            nn.Linear(hidden_dim * self.num_directions, 32),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(32, output_dim)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: (batch_size, seq_length, input_dim)
        lstm_out, _ = self.lstm(x)
        # Lấy hidden state ở bước thời gian cuối cùng: (batch_size, hidden_dim * directions)
        last_step_out = lstm_out[:, -1, :]
        out = self.fc_layers(last_step_out)
        return out.squeeze(-1)
