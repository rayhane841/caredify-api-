import json

try:
    import numpy as np  # type: ignore[import]
except ImportError as exc:
    raise ImportError("numpy requis pour le Modèle B") from exc

try:
    import tensorflow as tf  # type: ignore[import]
    print("[CAREDIFY] ✅ Modèle B — tensorflow chargé")
except ImportError as exc:
    raise ImportError("tensorflow-cpu requis pour le Modèle B") from exc


class ModelBHandler:
    """
    Modèle B — MLP résiduel contextuel.
    Raffine le score du Modèle A avec les métadonnées patient.
    NE déclenche JAMAIS d'alerte seul (cahier des charges §5.2).
    """

    def __init__(
        self,
        model_path: str = "weights/model_b_mlp_final.keras",
        config_path: str = "weights/caredify_model_b_config.json",
    ):
        with open(config_path) as f:
            self.config = json.load(f)

        self.model = tf.keras.models.load_model(model_path)

        self.mean  = np.array(self.config["scaler"]["mean_"],  dtype=np.float32)
        self.scale = np.array(self.config["scaler"]["scale_"], dtype=np.float32)

        self.w_a          = float(self.config["score_combined"]["w_a"])   # 0.5
        self.w_b          = float(self.config["score_combined"]["w_b"])   # 0.5
        self.temp         = float(self.config["calibration"]["temperature"])        # 0.601
        self.thr_critique = float(self.config["calibration"]["thresholds"]["critique"])  # 0.5
        self.thr_suspect  = float(self.config["calibration"]["thresholds"]["suspect"])   # 0.45

        print(f"[CAREDIFY] ✅ Modèle B chargé : {model_path}")

    def _classe_a_to_int(self, predicted_class: str) -> int:
        """Normal=0, Suspect=1, Critique=2"""
        mapping = {"Normal": 0, "Suspect": 1, "Critique": 2}
        return mapping.get(predicted_class, 0)

    def predict(
        self,
        result_a: dict,
        patient_meta: dict | None = None,
    ) -> dict:
        """
        result_a     : sortie du Modèle A
                       {score, predicted_class, confidence, proba}
        patient_meta : {age, sex, bmi, bmi_category}

        Retourne {status, score_a, score_b, score_combined, predicted_class_a}
        """
        # ── 1. Features Modèle A ─────────────────────────────────────────────
        score_a      = float(result_a.get("score", 0))
        classe_a_int = float(self._classe_a_to_int(
            result_a.get("predicted_class", "Normal")
        ))

        proba_raw = result_a.get("proba", None)
        if proba_raw is not None and len(proba_raw) == 3:
            prob_normal, prob_suspect, prob_critique = (
                float(proba_raw[0]),
                float(proba_raw[1]),
                float(proba_raw[2]),
            )
        else:
            # Reconstruction approximative si probas absentes
            prob_critique = score_a / 100.0 * 0.85
            prob_suspect  = score_a / 100.0 * 0.15
            prob_normal   = max(0.0, 1.0 - prob_critique - prob_suspect)

        # ── 2. Métadonnées patient ────────────────────────────────────────────
        age         = float(patient_meta.get("age")    or 53.7)  if patient_meta else 53.7
        sex         = float(patient_meta.get("sex")    or 0.5)   if patient_meta else 0.5
        bmi_raw     = patient_meta.get("bmi")                     if patient_meta else None
        bmi         = float(bmi_raw) if bmi_raw else 25.0
        bmi_missing = 0.0 if bmi_raw else 1.0

        # ── 3. Vecteur features (9 features) ─────────────────────────────────
        # [score_a, classe_a, prob_normal, prob_suspect, prob_critique,
        #  age, sex, bmi, bmi_missing]
        features = np.array([[
            score_a,
            classe_a_int,
            prob_normal,
            prob_suspect,
            prob_critique,
            age,
            sex,
            bmi,
            bmi_missing,
        ]], dtype=np.float32)

        # ── 4. Normalisation StandardScaler ──────────────────────────────────
        features_scaled = (features - self.mean) / (self.scale + 1e-8)

        # ── 5. Inférence Modèle B ─────────────────────────────────────────────
        logits = self.model.predict(features_scaled, verbose=0)[0]

        # Calibration température
        logits_cal = logits / self.temp
        exp_l      = np.exp(logits_cal - np.max(logits_cal))
        proba_b    = exp_l / exp_l.sum()

        # Score B composite 0-100
        score_b = float(proba_b[1] * 0.15 * 100 + proba_b[2] * 0.85 * 100)

        # ── 6. Score combiné (formule cahier des charges) ─────────────────────
        score_combined = self.w_a * score_a + self.w_b * score_b

        # ── 7. Status final depuis score_combined ─────────────────────────────
        if score_combined >= self.thr_critique * 100:
            status = "critical"
        elif score_combined >= self.thr_suspect * 100:
            status = "warning"
        else:
            status = "normal"

        print(
            f"[MODEL_B] score_a={score_a:.1f} | "
            f"score_b={score_b:.1f} | "
            f"score_combined={score_combined:.1f} | "
            f"status={status}"
        )

        return {
            "status":            status,
            "score_a":           int(score_a),
            "score_b":           round(score_b, 1),
            "score_combined":    round(score_combined, 1),
            "predicted_class_a": result_a.get("predicted_class", "Normal"),
            "confidence_a":      result_a.get("confidence", 0.0),
        }