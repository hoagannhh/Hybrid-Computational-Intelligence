"""Persist intermediate CLI training data for inspection and reuse."""

import json
import pickle
import re
from datetime import datetime, timezone
from pathlib import Path

import numpy as np


def save_data_artifacts(df_features, cleaned_df, splits, pipeline, config, root="data"):
    """Save one run in matching features/processed directories without overwrites."""
    ticker = re.sub(r"[^A-Za-z0-9_-]+", "", config["data"]["ticker"]) or "stock"
    run_id = f"{ticker}_{datetime.now(timezone.utc):%Y%m%dT%H%M%S%fZ}"
    features_dir = Path(root) / "features" / run_id
    processed_dir = Path(root) / "processed" / run_id
    features_dir.mkdir(parents=True, exist_ok=False)
    processed_dir.mkdir(parents=True, exist_ok=False)

    df_features.to_csv(features_dir / "features.csv", encoding="utf-8-sig")
    cleaned_df.to_csv(processed_dir / "cleaned.csv", encoding="utf-8-sig")
    for name, frame in splits.items():
        # These CSVs retain original units; model-ready arrays are in sequences.npz.
        frame.to_csv(processed_dir / f"{name}.csv", encoding="utf-8-sig")

    array_keys = [f"{kind}_{split}" for split in ("train", "val", "test") for kind in ("X", "y")]
    np.savez_compressed(processed_dir / "sequences.npz", **{key: pipeline[key] for key in array_keys})
    with (processed_dir / "scalers.pkl").open("wb") as stream:
        pickle.dump({key: pipeline[key] for key in ("feature_scaler", "target_scaler")}, stream)

    metadata = {
        "run_id": run_id,
        "config": config,
        "feature_cols": list(pipeline["feature_cols"]),
        "csv_values": "Original units, before scaling; sequences.npz contains scaled sliding windows.",
        "scaler_fit_split": "train",
        "array_shapes": {key: list(pipeline[key].shape) for key in array_keys},
        "splits": {
            name: {"rows": len(frame), "start": str(frame.index.min()), "end": str(frame.index.max())}
            for name, frame in splits.items()
        },
    }
    with (processed_dir / "metadata.json").open("w", encoding="utf-8") as stream:
        json.dump(metadata, stream, ensure_ascii=False, indent=2, default=str)
    return features_dir, processed_dir
