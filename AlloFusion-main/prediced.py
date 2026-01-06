"""
Predicting Allosteric Site Residues in Proteins
Input: Feature vector of the protein sequence
Output: Residue IDs and probabilities of allosteric site residues

"""

import os
import pickle
import warnings

import numpy as np
import pandas as pd
from matplotlib import pyplot
from sklearn.metrics import confusion_matrix, precision_score, roc_auc_score, roc_curve
from sklearn.preprocessing import StandardScaler

# Lazy import to avoid conflicts with PyTorch
_tf = None
_tfl = None


def _ensure_tensorflow_loaded():
    """Lazy load TensorFlow to avoid conflicts with PyTorch."""
    global _tf, _tfl
    if _tf is None:
        import tensorflow as tf_module
        import tensorflow.keras.layers as tfl_module

        _tf = tf_module
        _tfl = tfl_module
    return _tf, _tfl


def _first_existing(paths):
    for p in paths:
        if p and os.path.exists(p):
            return p
    return None


def predict_allosteric_sites(pdb_id, chain_id):
    # Load TensorFlow only when this function is called
    tf, tfl = _ensure_tensorflow_loaded()

    # number of features
    feat_shape = 0

    # Training dataset directory - check env var or use default under data/
    train_data_dir = os.environ.get("ALLOFUSION_TRAIN_DATA_DIR")
    if train_data_dir:
        search_paths_1 = [os.path.join(train_data_dir, "train_dataset_1.pkl")]
        search_paths_0 = [os.path.join(train_data_dir, "train_dataset_0.pkl")]
    else:
        # Default: data/training_datasets/ with legacy fallback
        search_paths_1 = [
            os.path.join("data", "training_datasets", "train_dataset_1.pkl"),
            os.path.join("features_data", "diversity", "train_dataset_1.pkl"),
            os.path.join(
                os.path.dirname(__file__),
                "features_data",
                "diversity",
                "train_dataset_1.pkl",
            ),
        ]
        search_paths_0 = [
            os.path.join("data", "training_datasets", "train_dataset_0.pkl"),
            os.path.join("features_data", "diversity", "train_dataset_0.pkl"),
            os.path.join(
                os.path.dirname(__file__),
                "features_data",
                "diversity",
                "train_dataset_0.pkl",
            ),
        ]

    ds1 = _first_existing(search_paths_1)
    if not ds1:
        raise FileNotFoundError(
            "Missing train_dataset_1.pkl. Download with:\n"
            "  python scripts/fetch_release_assets.py --repo hjb-001/AlloFusion --pattern 'train_dataset_*.pkl' --dest data/training_datasets --tag v1.0.0"
        )
    with open(ds1, "rb") as file:
        positive_set = pickle.load(file)

    ds0 = _first_existing(search_paths_0)
    if not ds0:
        raise FileNotFoundError(
            "Missing train_dataset_0.pkl. Download with:\n"
            "  python scripts/fetch_release_assets.py --repo hjb-001/AlloFusion --pattern 'train_dataset_*.pkl' --dest data/training_datasets --tag v1.0.0"
        )
    with open(ds0, "rb") as file:
        negative_set_entire = pickle.load(file)
    column_names = ["pdb_name", "residue", "features", "label"]
    # 确保 positive_set 和 negative_set_entire 是 DataFrame
    if isinstance(positive_set, dict):
        positive_set = pd.DataFrame.from_dict(positive_set)
    if isinstance(negative_set_entire, dict):
        negative_set_entire = pd.DataFrame.from_dict(negative_set_entire)

    Negative_Samples = negative_set_entire.sample(
        n=round(len(positive_set) * 14), random_state=42
    )

    # combine positive and negative sets to make the final dataset
    Train_set = pd.concat([positive_set, Negative_Samples], ignore_index=True, axis=0)

    # collect the features and labels of train set
    np.set_printoptions(suppress=True)
    X_val = [0] * len(Train_set)
    for i in range(len(Train_set)):
        feat = Train_set["features"][i]
        X_val[i] = feat
    X_train_orig = np.asarray(X_val)

    # Generate a random order of elements with np.random.permutation and simply index into the arrays Feature and label
    idx = np.random.permutation(len(X_train_orig))
    X_train = X_train_orig[idx]
    scaler = StandardScaler()
    scaler.fit(X_train)  # fit on training set only
    X_train = scaler.transform(X_train)  # apply transform to the training set

    # Resolve dataset location (root Case Study or AlloFusion-main/Case Study)
    ds_case = _first_existing(
        [
            os.path.join("Case Study", f"{pdb_id}_{chain_id}_dataset.pkl"),
            os.path.join(
                os.path.dirname(__file__),
                "Case Study",
                f"{pdb_id}_{chain_id}_dataset.pkl",
            ),
        ]
    )
    if not ds_case:
        raise FileNotFoundError(
            f"Dataset pickle not found for {pdb_id}_{chain_id}. Expected under 'Case Study/' (root) or 'AlloFusion-main/Case Study/'."
        )
    case_dir = os.path.dirname(ds_case)
    with open(ds_case, "rb") as file:
        test_set_entire = pickle.load(file)
    if isinstance(test_set_entire, dict):
        test_set = pd.DataFrame.from_dict(test_set_entire)

    X_independent = [0] * len(test_set)
    for i in range(len(test_set)):
        feat1 = test_set["features"][i]
        X_independent[i] = feat1
    X_test = np.asarray(X_independent)

    # Some run settings (e.g., topology augmentation) append extra features
    # so align the scaler if we unexpectedly see more dims.
    def _extend_scaler_for_extra_features(scaler_obj, extra_dims):
        if extra_dims <= 0:
            return
        pad = np.zeros(extra_dims, dtype=scaler_obj.mean_.dtype)
        scaler_obj.mean_ = np.concatenate((scaler_obj.mean_, pad))
        scaler_obj.scale_ = np.concatenate(
            (scaler_obj.scale_, np.ones(extra_dims, dtype=scaler_obj.scale_.dtype))
        )
        scaler_obj.var_ = np.concatenate(
            (scaler_obj.var_, np.ones(extra_dims, dtype=scaler_obj.var_.dtype))
        )
        scaler_obj.n_features_in_ = scaler_obj.n_features_in_ + extra_dims

    diff = X_test.shape[1] - scaler.n_features_in_
    if diff > 0:
        _extend_scaler_for_extra_features(scaler, diff)
    X_test = scaler.transform(X_test)
    feat_shape = X_test[0].size
    try:
        expected_map = {
            1047: "baseline (1024+20+3)",
            1050: "baseline + DCI(2) + betweenness(1)",
            1058: "baseline + StingAllo(11)",
            1060: "baseline + DCI(2) + MSF(1) + CFbetw(1) + CFclose(1) + LapPE(8)",
            1061: "baseline + StingAllo(11) + DCI(2) + betweenness(1)",
            1071: "baseline + StingAllo(11) + extended topology(+13)",
        }
        exp = expected_map.get(int(feat_shape))
        if exp:
            print(f"[INFO] Feature dim: {feat_shape} ({exp})")
        else:
            print(f"[INFO] Feature dim: {feat_shape}")
    except Exception:
        pass

    # Optionally build model from saved hyperparameters JSON; fallback to Trial 1 architecture
    def _load_hparams_dict(feat_dim: int):
        import json

        # Only load hyperparameters when the user explicitly provides a path.
        # This prevents accidentally building a 96-filter conv stack when
        # the default architecture should be 32-128-32-32.
        env_path = os.environ.get("ALLOFUSION_HPARAMS_JSON")
        if not env_path:
            return None
        try:
            if os.path.exists(env_path):
                with open(env_path, "r") as f:
                    data = json.load(f)
                hp = data.get("hyperparameters", data)
                return hp
        except Exception:
            pass
        return None

    def _cnn_from_hparams(feat_dim: int, hp: dict):
        m = tf.keras.Sequential()
        m.add(
            tfl.Conv1D(
                int(hp.get("conv1_filters", 32)),
                int(hp.get("conv1_kernel", 3)),
                padding="same",
                activation="relu",
                input_shape=(feat_dim, 1),
            )
        )
        m.add(tfl.BatchNormalization())
        m.add(tfl.Dropout(float(hp.get("dropout1", 0.2))))

        m.add(
            tfl.Conv1D(
                int(hp.get("conv2_filters", 128)),
                int(hp.get("conv2_kernel", 3)),
                padding="same",
                activation="relu",
            )
        )
        m.add(tfl.BatchNormalization())
        m.add(tfl.Dropout(float(hp.get("dropout2", 0.3))))

        m.add(
            tfl.Conv1D(
                int(hp.get("conv3_filters", 32)),
                int(hp.get("conv3_kernel", 5)),
                padding="same",
                activation="relu",
            )
        )
        m.add(tfl.BatchNormalization())
        m.add(tfl.Dropout(float(hp.get("dropout3", 0.2))))

        m.add(
            tfl.Conv1D(
                int(hp.get("conv4_filters", 32)),
                int(hp.get("conv4_kernel", 3)),
                padding="same",
                activation="relu",
            )
        )
        m.add(tfl.BatchNormalization())
        m.add(tfl.Dropout(float(hp.get("dropout4", 0.3))))

        m.add(tfl.Flatten())
        m.add(tfl.Dense(int(hp.get("dense_units", 128)), activation="relu"))
        m.add(tfl.Dense(32, activation="relu"))
        m.add(tfl.Dense(1, activation="sigmoid"))
        return m

    # Vendor baseline architecture retained as fallback
    def _cnn_vendor(feat_dim: int):
        model = tf.keras.Sequential()
        model.add(
            tfl.Conv1D(
                32, 3, padding="same", activation="relu", input_shape=(feat_dim, 1)
            )
        )
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

    hp_dict = _load_hparams_dict(feat_shape)
    if hp_dict:
        print(
            "[INFO] Building CNN from explicit hyperparameters JSON (ALLOFUSION_HPARAMS_JSON)"
        )
        cnn_model = _cnn_from_hparams(feat_shape, hp_dict)
    else:
        # Default to the 32-128-32-32 architecture
        print("[INFO] Building CNN with default 32-128-32-32 architecture")
        cnn_model = _cnn_vendor(feat_shape)

    # CNN weights: allow explicit file via env, else dir/env, else defaults
    weights_file_env = os.environ.get("ALLOFUSION_CNN_WEIGHTS_FILE")
    weights_path = None
    if weights_file_env and os.path.exists(weights_file_env):
        weights_path = weights_file_env
    else:
        # CNN weights directory - check env var or use default under data/
        cnn_weights_dir = os.environ.get("ALLOFUSION_CNN_WEIGHTS_DIR")
        preferred_dim = None
        try:
            preferred_dim = int(feat_shape)
        except Exception:
            preferred_dim = None
        if cnn_weights_dir:
            weights_search = []
            if preferred_dim is not None:
                weights_search.append(os.path.join(cnn_weights_dir, f"all_{preferred_dim}d.h5"))
            weights_search.extend(
                [
                    os.path.join(cnn_weights_dir, "trial1.h5"),
                    os.path.join(cnn_weights_dir, "trial1_final_model.h5"),
                    os.path.join(cnn_weights_dir, "all_1050d.h5"),
                    os.path.join(cnn_weights_dir, "all_1060d.h5"),
                    os.path.join(cnn_weights_dir, "originalallofusion.h5"),
                    os.path.join(cnn_weights_dir, "all.h5"),
                ]
            )
        else:
            # Default: data/cnn_weights/ with legacy fallback
            weights_search = []
            if preferred_dim is not None:
                weights_search.extend(
                    [
                        os.path.join("data", "cnn_weights", f"all_{preferred_dim}d.h5"),
                        os.path.join("myModel", f"all_{preferred_dim}d.h5"),
                        os.path.join(os.path.dirname(__file__), "myModel", f"all_{preferred_dim}d.h5"),
                    ]
                )
            weights_search.extend(
                [
                    os.path.join("data", "cnn_weights", "trial1.h5"),
                    os.path.join("data", "cnn_weights", "trial1_final_model.h5"),
                    os.path.join("data", "cnn_weights", "all_1050d.h5"),
                    os.path.join("data", "cnn_weights", "all_1060d.h5"),
                    os.path.join("data", "cnn_weights", "originalallofusion.h5"),
                    os.path.join("data", "cnn_weights", "all.h5"),
                    os.path.join("myModel", "trial1.h5"),
                    os.path.join("myModel", "trial1_final_model.h5"),
                    os.path.join("myModel", "all_1050d.h5"),
                    os.path.join("myModel", "all_1060d.h5"),
                    os.path.join("myModel", "originalallofusion.h5"),
                    os.path.join("myModel", "all.h5"),
                    os.path.join(os.path.dirname(__file__), "myModel", "trial1.h5"),
                    os.path.join(os.path.dirname(__file__), "myModel", "trial1_final_model.h5"),
                    os.path.join(os.path.dirname(__file__), "myModel", "all_1050d.h5"),
                    os.path.join(os.path.dirname(__file__), "myModel", "all_1060d.h5"),
                    os.path.join(os.path.dirname(__file__), "myModel", "originalallofusion.h5"),
                    os.path.join(os.path.dirname(__file__), "myModel", "all.h5"),
                ]
            )
        weights_path = _first_existing(weights_search)
    if not weights_path:
        raise FileNotFoundError(
            "Missing CNN weights. Provide via:\n"
            "  - ALLOFUSION_CNN_WEIGHTS_FILE=/path/to/your_model.h5\n"
            "  - or ALLOFUSION_CNN_WEIGHTS_DIR=/path/with/{trial1,all_<DIM>d,originalallofusion,all}.h5\n"
            "  - or download: python scripts/fetch_release_assets.py --repo hjb-001/AlloFusion --pattern 'trial1.h5' --dest data/cnn_weights\n"
        )
    print(f"[INFO] Loading CNN weights from: {weights_path}")
    try:
        cnn_model.load_weights(weights_path)
    except Exception as e:
        raise RuntimeError(
            "Failed to load CNN weights. This is commonly caused by a feature-dimension mismatch.\n"
            f"  - Feature dim in dataset: {feat_shape}\n"
            f"  - Weights file: {weights_path}\n"
            "If you enabled topology augmentation, you need CNN weights trained for that feature dim "
            "(e.g., 1050D for DCI+betweenness, 1060D for extended topology with LapPE k=8)."
        ) from e
    Inde_test_prob = cnn_model.predict(X_test)

    # Threshold for classification (default 0.5), overridable via env
    try:
        threshold = float(os.environ.get("ALLOFUSION_THRESHOLD", "0.5"))
    except Exception:
        threshold = 0.5
    print(f"[INFO] Using decision threshold: {threshold}")

    afr_sites = []
    for i, prob in enumerate(Inde_test_prob):
        if prob[0] > threshold:
            # print resid
            # print(f"african_sites:{test_set['label'][i]}: Probability of being positive class = {prob[0]}")
            afr_sites.append(test_set["label"][i])

    # write the allosteric residues to a file
    with open(os.path.join(case_dir, f"{pdb_id}_allosteric_residues.txt"), "w") as f:
        afr_sites_str = ",".join(afr_sites)
        f.write("AlloFusion Allosteric Site Forming Residues:\n")
        f.write("Residues: ( Chain " + chain_id + " and resid " + afr_sites_str + " )")

    # generate pml file
    def generate_pml_content(pdb_code, chain_name, prediction_data):
        # Build residue selection string
        resi_str = "+".join(map(str, prediction_data)) if prediction_data else "none"

        pml_content = f"""# PyMOL script for AlloFusion predictions
# PDB: {pdb_code}
# Chain: {chain_name}
# Predicted residues: {len(prediction_data)}

# Fetch structure from PDB
fetch {pdb_code}, async=0

# Basic visualization setup
hide everything
show cartoon
color gray60, all
set cartoon_transparency, 0.3
bg_color white

# Define predicted residue selection
select predicted_residues, (chain {chain_name} and resi {resi_str})

# Visualize predicted residues as spheres
show spheres, predicted_residues and name CA
set sphere_scale, 1.5, predicted_residues and name CA
color red, predicted_residues and name CA
set sphere_transparency, 0.0, predicted_residues

# Hide ligands
hide everything, organic
hide everything, inorganic
hide everything, solvent

# Set optimal view
zoom chain {chain_name}
orient

# Ray-traced rendering settings
set ray_shadows, 0
set antialias, 2
set ambient, 0.4
set specular, 0.5

# Render image
ray 2400, 2400
png {pdb_code}_allosteric_sites.png, dpi=300

print 'AlloFusion visualization complete for {pdb_code}!'
"""
        return pml_content

    pml_content = generate_pml_content(pdb_id, chain_id, afr_sites)
    # save pml file
    pml_file_path = os.path.join(case_dir, f"{pdb_id}_allosteric_sites.pml")
    with open(pml_file_path, "w") as f:
        f.write(pml_content)


# Example usage:
# predict_allosteric_sites('5xjy', 'A')
