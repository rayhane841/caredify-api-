import numpy as np  # type: ignore[import]
import json

def load_config(path="weights/caredify_config.json") -> dict:
    with open(path) as f:
        return json.load(f)

def preprocess_ecg(ecg_values: list[float], config: dict) -> np.ndarray:
    """
    Prend les valeurs brutes ECG du buffer Flutter,
    les rééchantillonne à 187 points (input shape du modèle),
    puis applique le scaler stocké dans config.json.
    """
    mean  = np.array(config["scaler"]["mean"],  dtype=np.float32)
    scale = np.array(config["scaler"]["scale"], dtype=np.float32)
    input_len = config["input_shape"][0]  # 187

    arr = np.array(ecg_values, dtype=np.float32)

    # Rééchantillonnage linéaire vers 187 points
    if len(arr) != input_len:
        x_old = np.linspace(0, 1, len(arr))
        x_new = np.linspace(0, 1, input_len)
        arr   = np.interp(x_new, x_old, arr).astype(np.float32)

    # Normalisation (même scaler que l'entraînement)
    arr = (arr - mean) / (scale + 1e-8)

    # Shape attendue par TFLite : (1, 187, 1)
    return arr.reshape(1, input_len, 1)