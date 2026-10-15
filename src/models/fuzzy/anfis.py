"""First-order Takagi-Sugeno ANFIS trained end-to-end by gradient descent.

Each rule has a bell membership function per input and a linear consequent.
Sequence inputs use the final historical observation, never the target row.
"""
import torch
from torch import nn


class BellMembershipFunction(nn.Module):
    """Positive widths/slopes and log memberships for stable rule products."""

    def __init__(self, in_features: int, num_rules: int):
        super().__init__()
        if in_features < 1 or num_rules < 1:
            raise ValueError("in_features and num_rules must be positive.")
        self.raw_width = nn.Parameter(torch.zeros(in_features, num_rules))
        self.raw_slope = nn.Parameter(torch.zeros(in_features, num_rules))
        # Inputs are scaled from train data; random centers break rule symmetry.
        self.centers = nn.Parameter(torch.rand(in_features, num_rules))

    def log_membership(self, x):
        width = nn.functional.softplus(self.raw_width) + 1e-4
        slope = nn.functional.softplus(self.raw_slope) + 0.5
        distance = ((x.unsqueeze(-1) - self.centers) / width).abs().clamp_min(1e-6)
        # log(1 / (1 + distance ** (2 * slope))), without exponent overflow.
        return -nn.functional.softplus(2 * slope * distance.log())

    def forward(self, x):
        return self.log_membership(x).exp()


class ANFIS(nn.Module):
    """Normalized fuzzy rules with trainable premises and linear consequents."""

    def __init__(self, input_dim: int, num_rules: int = 5):
        super().__init__()
        self.input_dim = input_dim
        self.num_rules = num_rules
        self.mf = BellMembershipFunction(input_dim, num_rules)
        self.linear_weights = nn.Parameter(torch.randn(num_rules, input_dim) * 0.05)
        self.biases = nn.Parameter(torch.zeros(num_rules))

    def forward(self, x):
        if x.dim() == 3:
            x = x[:, -1, :]
        if x.dim() != 2 or x.shape[-1] != self.input_dim:
            raise ValueError("ANFIS expects [batch, features] or [batch, time, features].")
        # softmax(log product) avoids underflow as the number of features grows.
        log_firing = self.mf.log_membership(x).sum(dim=1)
        weights = torch.softmax(log_firing, dim=1)
        consequents = x @ self.linear_weights.t() + self.biases
        return (weights * consequents).sum(dim=1)
