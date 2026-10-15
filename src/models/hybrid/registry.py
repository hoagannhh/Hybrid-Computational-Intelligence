"""Supported local experiments: four LSTM compositions and ANFIS.

Wavelet transforms the inputs; PSO selects LSTM hyperparameters. Neither needs
a separate neural-network class. Add a supported composition here to expose it
to the local CLI and runner together.
"""
from dataclasses import dataclass


@dataclass(frozen=True)
class ModelSpec:
    cli_name: str
    label: str
    wavelet: bool = False
    pso: bool = False
    model_type: str = "LSTM"


MODEL_SPECS = {
    spec.label: spec for spec in (
        ModelSpec("lstm", "LSTM"),
        ModelSpec("lstm-pso", "LSTM_PSO", pso=True),
        ModelSpec("wavelet-lstm", "Wavelet_LSTM", wavelet=True),
        ModelSpec("wavelet-lstm-pso", "Wavelet_LSTM_PSO", wavelet=True, pso=True),
        ModelSpec("anfis", "ANFIS", model_type="ANFIS"),
    )
}
MODEL_NAMES = {spec.cli_name: spec.label for spec in MODEL_SPECS.values()}
