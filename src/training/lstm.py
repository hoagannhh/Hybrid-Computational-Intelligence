"""LSTM entry point using the shared supervised trainer."""
from src.training.supervised import fit_model, make_loader, predict_scaled


def fit_lstm(data, params, config, device, seed, epochs, patience, checkpoint=None, quiet=False):
    return fit_model(data, params, config, device, seed, epochs, patience,
                     checkpoint, quiet, model_type="LSTM")
