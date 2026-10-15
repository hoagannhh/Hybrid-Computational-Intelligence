"""Run directories, metadata and reproducible seeds for local experiments."""
import importlib.metadata
import json
import random
import re
from datetime import datetime, timezone
from pathlib import Path
import numpy as np
import torch


def save_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False), encoding="utf-8")


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def select_device(require_gpu=False):
    if require_gpu and not torch.cuda.is_available():
        raise RuntimeError("CUDA chưa khả dụng. Cần GPU và PyTorch hỗ trợ CUDA, hoặc chọn --device cpu.")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device, torch.cuda.get_device_name(0) if device.type == "cuda" else "CPU")
    return device


def new_run(config, label):
    base = Path(config["OUTPUT_ROOT"])
    base.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ")
    symbol = re.sub(r"[^A-Za-z0-9._-]", "_", config['TICKER'].replace('^', ''))
    root = base / f"{symbol}_{label}_{stamp}"
    root.mkdir(exist_ok=False)
    save_json(root / "config.json", config)
    versions = {}
    for name in ["numpy", "pandas", "scikit-learn", "torch", "matplotlib", "yfinance", "PyWavelets", "kagglehub"]:
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = None
    save_json(root / "environment.json", {"packages": versions, "cuda_runtime": torch.version.cuda,
              "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None})
    print("Output:", root)
    return root
