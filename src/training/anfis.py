"""ANFIS entry point; the model uses the last historical feature vector."""
from src.training.supervised import fit_model


def fit_anfis(data, params, config, device, seed, epochs, patience, checkpoint=None, quiet=False):
    return fit_model(data, params, config, device, seed, epochs, patience,
                     checkpoint, quiet, model_type="ANFIS")
