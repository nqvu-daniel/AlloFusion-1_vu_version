#!/usr/bin/env python3
"""Evaluate AlloFusion CNN predictions on a saved dataset pickle and log probabilities to CSV."""

import argparse
import json
import os
import pickle
from typing import Iterable, Optional

import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import StandardScaler


def _first_existing(paths: Iterable[str]) -> Optional[str]:
    """Return first path that exists from the provided iterable."""
    for path in paths:
        if path and os.path.exists(path):
            return path
    return None


def _load_pickle_dataset(path: str) -> pd.DataFrame:
    """Load pickle and normalize to a pandas DataFrame."""
    with open(path, "rb") as fh:
        payload = pickle.load(fh)
    if isinstance(payload, pd.DataFrame):
        return payload
    if isinstance(payload, dict):
        return pd.DataFrame.from_dict(payload)
    try:
        return pd.DataFrame(payload)
    except Exception as exc:
        raise ValueError(f"Cannot convert dataset from {path}: {exc}") from exc


def _convert_features(df: pd.DataFrame) -> np.ndarray:
    """Convert column 'features' into a (n_samples, n_features) numpy array."""
    if "features" not in df.columns:
        raise KeyError("Dataset must contain a 'features' column.")
    return np.asarray([np.asarray(feat, dtype=np.float32) for feat in df["features"]])


def _load_training_dataframe() -> pd.DataFrame:
    """Load the positive and negative training sets used to build the scaler."""
    env_dir = os.environ.get("ALLOFUSION_TRAIN_DATA_DIR")
    if env_dir:
        pos_search = [os.path.join(env_dir, "train_dataset_1.pkl")]
        neg_search = [os.path.join(env_dir, "train_dataset_0.pkl")]
    else:
        base_dir = os.path.dirname(__file__)
        fallback = [
            os.path.join("data", "training_datasets"),
            os.path.join("features_data", "diversity"),
            os.path.join(base_dir, "features_data", "diversity"),
        ]
        pos_search = [os.path.join(folder, "train_dataset_1.pkl") for folder in fallback]
        neg_search = [os.path.join(folder, "train_dataset_0.pkl") for folder in fallback]

    pos_path = _first_existing(pos_search)
    neg_path = _first_existing(neg_search)
    if not pos_path or not neg_path:
        raise FileNotFoundError(
            "Missing training datasets. Provide them by setting ALLOFUSION_TRAIN_DATA_DIR "
            "or placing training_datasets under data/ as in the repo."
        )

    pos_df = _load_pickle_dataset(pos_path)
    neg_df = _load_pickle_dataset(neg_path)
    n_neg = min(round(len(pos_df) * 14), len(neg_df))
    neg_sample = neg_df.sample(n=n_neg, random_state=42)
    return pd.concat([pos_df, neg_sample], ignore_index=True)


def _build_scaler(train_df: pd.DataFrame) -> StandardScaler:
    features = _convert_features(train_df)
    scaler = StandardScaler()
    scaler.fit(features)
    return scaler


def _load_hparams(feat_dim: int) -> Optional[dict]:
    """Load CNN hyperparameters from JSON if available."""
    env_path = os.environ.get("ALLOFUSION_HPARAMS_JSON")
    candidates = []
    if env_path:
        candidates.append(env_path)
    base_dir = os.path.dirname(__file__)
    suffix = f"best_hyperparameters_1dcnn_{feat_dim}d.json"
    candidates.extend([
        os.path.join(base_dir, "myModel", suffix),
        os.path.join("myModel", suffix),
    ])
    for path in candidates:
        if path and os.path.exists(path):
            try:
                with open(path, "r") as fh:
                    data = json.load(fh)
                return data.get("hyperparameters", data)
            except json.JSONDecodeError:
                continue
    return None


def _cnn_from_hparams(feat_dim: int, hp: dict) -> tf.keras.Model:
    tfl = tf.keras.layers
    model = tf.keras.Sequential()
    model.add(tfl.Conv1D(int(hp.get("conv1_filters", 32)), int(hp.get("conv1_kernel", 3)),
                         padding="same", activation="relu", input_shape=(feat_dim, 1)))
    model.add(tfl.BatchNormalization())
    model.add(tfl.Dropout(float(hp.get("dropout1", 0.2))))

    model.add(tfl.Conv1D(int(hp.get("conv2_filters", 128)), int(hp.get("conv2_kernel", 3)),
                         padding="same", activation="relu"))
    model.add(tfl.BatchNormalization())
    model.add(tfl.Dropout(float(hp.get("dropout2", 0.3))))

    model.add(tfl.Conv1D(int(hp.get("conv3_filters", 32)), int(hp.get("conv3_kernel", 5)),
                         padding="same", activation="relu"))
    model.add(tfl.BatchNormalization())
    model.add(tfl.Dropout(float(hp.get("dropout3", 0.2))))

    model.add(tfl.Conv1D(int(hp.get("conv4_filters", 32)), int(hp.get("conv4_kernel", 3)),
                         padding="same", activation="relu"))
    model.add(tfl.BatchNormalization())
    model.add(tfl.Dropout(float(hp.get("dropout4", 0.3))))

    model.add(tfl.Flatten())
    model.add(tfl.Dense(int(hp.get("dense_units", 128)), activation="relu"))
    model.add(tfl.Dense(32, activation="relu"))
    model.add(tfl.Dense(1, activation="sigmoid"))
    return model


def _cnn_vendor(feat_dim: int) -> tf.keras.Model:
    tfl = tf.keras.layers
    model = tf.keras.Sequential()
    model.add(tfl.Conv1D(32, 3, padding="same", activation="relu", input_shape=(feat_dim, 1)))
    model.add(tfl.BatchNormalization())
    model.add(tfl.Dropout(0.2))
    model.add(tfl.Conv1D(128, 3, padding="same", activation="relu"))
    model.add(tfl.BatchNormalization())
    model.add(tfl.Dropout(0.3))
    model.add(tfl.Conv1D(32, 5, padding="same", activation="relu"))
    model.add(tfl.BatchNormalization())
    model.add(tfl.Dropout(0.2))
    model.add(tfl.Conv1D(32, 3, padding="same", activation="relu"))
    model.add(tfl.BatchNormalization())
    model.add(tfl.Dropout(0.3))
    model.add(tfl.Flatten())
    model.add(tfl.Dense(128, activation="relu"))
    model.add(tfl.Dense(32, activation="relu"))
    model.add(tfl.Dense(1, activation="sigmoid"))
    return model


def _locate_weights(path_arg: Optional[str]) -> str:
    if path_arg:
        if os.path.isfile(path_arg):
            return path_arg
        if os.path.isdir(path_arg):
            candidates = ["trial1.h5", "all_1050d.h5", "originalallofusion.h5", "all.h5"]
            for name in candidates:
                candidate = os.path.join(path_arg, name)
                if os.path.exists(candidate):
                    return candidate
        raise FileNotFoundError(f"Provided weights argument is invalid: {path_arg}")

    env_file = os.environ.get("ALLOFUSION_CNN_WEIGHTS_FILE")
    if env_file and os.path.exists(env_file):
        return env_file
    env_dir = os.environ.get("ALLOFUSION_CNN_WEIGHTS_DIR")
    search_dirs = [env_dir] if env_dir else []
    base_dir = os.path.dirname(__file__)
    search_dirs.extend([
        os.path.join("data", "cnn_weights"),
        os.path.join("myModel"),
        os.path.join(base_dir, "myModel"),
    ])
    candidate_names = ["trial1.h5", "all_1050d.h5", "originalallofusion.h5", "all.h5"]
    for folder in search_dirs:
        if not folder:
            continue
        for name in candidate_names:
            candidate = os.path.join(folder, name)
            if os.path.exists(candidate):
                return candidate
    raise FileNotFoundError("CNN weights not found; specify --weights or set ALLOFUSION_CNN_WEIGHTS_DIR.")


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate AlloFusion CNN predictions on a dataset pickle.")
    parser.add_argument("--pickle", required=True, help="Path to the dataset pickle (features/labels).")
    parser.add_argument("--CHAIN", required=True, help="Protein chain identifier (single letter).")
    parser.add_argument("--weights", default=None, help="Path to CNN weights file or directory.")
    parser.add_argument("--threshold", type=float, default=None, help="Decision threshold (default 0.47).")
    parser.add_argument("--output", default=None, help="CSV file path for logging predictions.")
    args = parser.parse_args()

    dataset_path = os.path.abspath(args.pickle)
    if not os.path.isfile(dataset_path):
        raise FileNotFoundError(f"Dataset pickle not found: {dataset_path}")

    print(f"[INFO] Loading dataset from {dataset_path}")
    dataset = _load_pickle_dataset(dataset_path)
    if dataset.empty:
        raise ValueError("Dataset pickle is empty.")

    scaler = _build_scaler(_load_training_dataframe())
    test_features = _convert_features(dataset)
    feat_dim = test_features.shape[1]
    X_test = scaler.transform(test_features)

    hp = _load_hparams(feat_dim)
    if hp:
        print("[INFO] Building CNN from hyperparameters JSON.")
        cnn_model = _cnn_from_hparams(feat_dim, hp)
    else:
        print("[INFO] Building CNN using vendor baseline architecture.")
        cnn_model = _cnn_vendor(feat_dim)

    weights_path = _locate_weights(args.weights)
    print(f"[INFO] Loading weights from {weights_path}")
    cnn_model.load_weights(weights_path)

    predictions = cnn_model.predict(X_test, verbose=0)
    probs = predictions.flatten()
    threshold = args.threshold if args.threshold is not None else float(
        os.environ.get("ALLOFUSION_THRESHOLD", "0.47")
    )
    print(f"[INFO] Using decision threshold: {threshold}")

    residues = dataset.get("label", pd.Series([""] * len(dataset))).astype(str).str.strip()
    aas = dataset.get("residue", pd.Series([""] * len(dataset))).astype(str)
    pdb_names = dataset.get("pdb_name", pd.Series([""] * len(dataset))).astype(str)
    pdb_name = pdb_names.iloc[0] if not pdb_names.empty else os.path.splitext(os.path.basename(dataset_path))[0]

    predictions_df = pd.DataFrame({
        "pdb": pdb_name,
        "chain": args.CHAIN,
        "residue": residues,
        "amino_acid": aas,
        "probability": probs,
        "prediction": (probs > threshold).astype(int),
    })

    output_path = args.output or os.path.join(
        os.path.dirname(dataset_path),
        f"{pdb_name}_{args.CHAIN}_predictions.csv"
    )
    output_dir = os.path.dirname(output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    predictions_df.to_csv(output_path, index=False)
    print(f"[INFO] Predictions logged to {output_path}")
    positives = int(predictions_df["prediction"].sum())
    print(f"[INFO] {positives}/{len(predictions_df)} residues exceed threshold.")


if __name__ == "__main__":
    main()
