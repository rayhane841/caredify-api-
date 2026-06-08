import json
import numpy as np  # type: ignore[import]

_config_cache: dict | None = None


def load_config(path: str = "weights/caredify_config.json") -> dict:
    global _config_cache
    if _config_cache is None:
        with open(path, "r") as f:
            _config_cache = json.load(f)
    return _config_cache


def preprocess_ecg(
    ecg_values: list,
    config: dict,
    patient_meta: dict | None = None,
) -> np.ndarray:
    """
    Prétraitement complet identique à l'entraînement PTB-XL :
    1. Conversion float32
    2. Nettoyage NaN / Inf
    3. Clipping artéfacts (± 5 mV)
    4. Rééchantillonnage → 187 points
    5. StandardScaler (mean/scale depuis config.json)
    6. Reshape (1, 187, 1)
    """
    target_len = config["input_shape"][0]  # 187
    mean  = np.array(config["scaler"]["mean"],  dtype=np.float32)
    scale = np.array(config["scaler"]["scale"], dtype=np.float32)

    arr = np.array(ecg_values, dtype=np.float32)

    if len(arr) == 0:
        raise ValueError("ecg_values vide")

    # ── 1. Nettoyage NaN / Inf ────────────────────────────────────────────────
    arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)

    # ── 2. Clipping artéfacts extrêmes ────────────────────────────────────────
    arr = np.clip(arr, -5.0, 5.0)

    # ── 3. Rééchantillonnage vers 187 points ──────────────────────────────────
    if len(arr) != target_len:
        x_orig = np.linspace(0.0, 1.0, len(arr))
        x_new  = np.linspace(0.0, 1.0, target_len)
        arr    = np.interp(x_new, x_orig, arr).astype(np.float32)

    # ── 4. Normalisation StandardScaler ───────────────────────────────────────
    arr = (arr - mean) / (scale + 1e-8)

    # ── 5. Log métadonnées patient ────────────────────────────────────────────
    if patient_meta:
        print(
            f"[PREPROCESS] age={patient_meta.get('age', '?')} "
            f"sex={patient_meta.get('sex', '?')} "
            f"weight={patient_meta.get('weight', '?')} "
            f"bmi={patient_meta.get('bmi', '?')} "
            f"pathology={patient_meta.get('cardiac_pathology', '?')}"
        )

    return arr.reshape(1, target_len, 1)