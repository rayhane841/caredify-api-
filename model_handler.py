import importlib
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from numpy import ndarray  # type: ignore[import]
else:
    ndarray = Any

try:
    np = importlib.import_module("numpy")
except ImportError as exc:
    raise ImportError(
        "numpy is required. Install it with: pip install numpy"
    ) from exc

try:
    tflite = importlib.import_module("tflite_runtime.interpreter")
    Interpreter = tflite.Interpreter
    print("[CAREDIFY] ✅ tflite_runtime chargé")
except ImportError:
    try:
        tf = importlib.import_module("tensorflow")
        Interpreter = tf.lite.Interpreter
        print("[CAREDIFY] ✅ tensorflow.lite chargé")
    except ImportError as exc:
        raise ImportError("Installe tensorflow-cpu ou tflite-runtime") from exc


class ModelHandler:
    def __init__(
        self,
        model_path: str = "weights/caredify_modele_a_1lead_mobile.tflite",
        config: dict = None,
    ):
        self.config = config
        self.interpreter = Interpreter(model_path=model_path)
        self.interpreter.allocate_tensors()
        self.input_idx  = self.interpreter.get_input_details()[0]["index"]
        self.output_idx = self.interpreter.get_output_details()[0]["index"]
        print(f"[CAREDIFY] ✅ Modèle A TFLite chargé : {model_path}")

    def predict(self, input_array: ndarray) -> dict:
        self.interpreter.set_tensor(self.input_idx, input_array)
        self.interpreter.invoke()
        proba = self.interpreter.get_tensor(self.output_idx)[0]
        # proba shape : (3,) → [P_Normal, P_Suspect, P_Critique]

        cfg        = self.config
        weights    = cfg["score_weights"]
        thresholds = cfg["thresholds"]
        classes    = cfg["classes"]

        # Score composite 0-100
        score = int(
            proba[1] * weights["suspect"]  * 100
          + proba[2] * weights["critique"] * 100
        )

        predicted_i = int(np.argmax(proba))
        confidence  = float(proba[predicted_i])

        if score >= thresholds["critique"]:
            status = "critical"
        elif score >= thresholds["suspect"]:
            status = "warning"
        else:
            status = "normal"

        return {
            "status":          status,
            "score":           score,
            "confidence":      round(confidence, 3),
            "predicted_class": classes[predicted_i],
            # ✅ Probas brutes exposées pour le Modèle B
            "proba":           proba.tolist(),
        }