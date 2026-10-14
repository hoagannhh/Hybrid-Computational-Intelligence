import os
import random
import yaml
import logging
import numpy as np
import torch

def set_seed(seed: int = 42) -> None:
    """Thiết lập random seed cố định để đảm bảo tính tái lập kết quả (reproducibility)."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

def load_config(config_path: str = "configs/config.yaml") -> dict:
    """Đọc file cấu hình yaml."""
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Không tìm thấy file config: {config_path}")
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    return config

def get_device(device_str: str = "cuda") -> torch.device:
    """Lựa chọn thiết bị tính toán (CUDA/CPU)."""
    if device_str.lower() == "cuda" and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")
