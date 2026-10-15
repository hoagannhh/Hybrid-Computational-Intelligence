"""Portable LSTM/ANFIS experiment checkpoints.

Version 1 uses ``head.*`` on disk, as did earlier local runs and the independent
Kaggle notebooks. The local model uses ``fc_layers.*`` internally. Translate only
at the serialization boundary so old runs and notebook 05 remain compatible.
ANFIS uses version 2 with an explicit model_type; older v1 files imply LSTM.
"""
import torch

from src.models.neural.lstm_model import StockLSTM
from src.models.fuzzy.anfis import ANFIS


def export_weights(state_dict):
    return {
        ("head." + key[len("fc_layers."):] if key.startswith("fc_layers.") else key): value
        for key, value in state_dict.items()
    }


def build_model(model_type, model_args):
    classes = {"LSTM": StockLSTM, "ANFIS": ANFIS}
    if model_type not in classes:
        raise ValueError(f"Unsupported model_type: {model_type}")
    return classes[model_type](**model_args)


def load_model(path, device):
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    version = checkpoint.get("format_version")
    if version not in {1, 2}:
        raise ValueError("Unsupported experiment checkpoint format; expected version 1 or 2.")
    model_type = checkpoint.get("model_type", "LSTM") if version == 1 else checkpoint["model_type"]
    if version == 1 and model_type != "LSTM":
        raise ValueError("Version 1 checkpoints only support LSTM.")
    model = build_model(model_type, checkpoint["model_args"]).to(device)
    weights = {
        ("fc_layers." + key[len("head."):] if key.startswith("head.") else key): value
        for key, value in checkpoint["state_dict"].items()
    }
    model.load_state_dict(weights, strict=True)
    model.eval()
    return model
