import torch
import torch.nn as nn

class BellMembershipFunction(nn.Module):
    """
    Hàm liên thuộc dạng hình chuông (Generalized Bell Membership Function)
    mu(x) = 1 / (1 + |(x - c) / a|^(2b))
    """
    def __init__(self, in_features: int, num_rules: int):
        super(BellMembershipFunction, self).__init__()
        self.a = nn.Parameter(torch.ones(in_features, num_rules))
        self.b = nn.Parameter(torch.ones(in_features, num_rules))
        self.c = nn.Parameter(torch.randn(in_features, num_rules))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: (batch_size, in_features)
        x_exp = x.unsqueeze(-1) # (batch_size, in_features, 1)
        diff = torch.abs((x_exp - self.c) / (self.a + 1e-6))
        mu = 1.0 / (1.0 + torch.pow(diff, 2.0 * self.b) + 1e-6)
        return mu

class ANFIS(nn.Module):
    """
    Adaptive Neuro-Fuzzy Inference System (Takagi-Sugeno Model)
    Tích hợp trực tiếp vào PyTorch để huấn luyện End-to-End bằng Gradient Descent.
    """
    def __init__(self, input_dim: int, num_rules: int = 5):
        super(ANFIS, self).__init__()
        self.input_dim = input_dim
        self.num_rules = num_rules

        # Lớp 1: Fuzzification (Hàm liên thuộc)
        self.mf = BellMembershipFunction(input_dim, num_rules)

        # Lớp 4 & 5: Consequent parameters (Takagi-Sugeno: f_i = p_i * x + q_i)
        self.linear_weights = nn.Parameter(torch.randn(num_rules, input_dim))
        self.biases = nn.Parameter(torch.zeros(num_rules))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Giả sử x là vector đặc trưng tại bước hiện tại hoặc output từ LSTM: (batch_size, input_dim)
        if x.dim() == 3:
            x = x[:, -1, :] # Lấy bước thời gian cuối nếu truyền vào chuỗi

        batch_size = x.size(0)

        # Lớp 1: Tính độ liên thuộc mu: (batch, input_dim, num_rules)
        mu = self.mf(x)

        # Lớp 2: Rule Firing Strength (T-norm: product): (batch, num_rules)
        w = torch.prod(mu, dim=1)

        # Lớp 3: Normalization: (batch, num_rules)
        w_sum = torch.sum(w, dim=1, keepdim=True) + 1e-8
        w_norm = w / w_sum

        # Lớp 4: Consequent output cho từng rule
        # rule_out_i = x * p_i + q_i -> shape (batch, num_rules)
        rule_outputs = torch.matmul(x, self.linear_weights.t()) + self.biases

        # Lớp 5: Defuzzification (Tổng có trọng số)
        out = torch.sum(w_norm * rule_outputs, dim=1)
        return out
