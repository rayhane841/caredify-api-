import os
from fastapi import FastAPI, Request, HTTPException, Header  # type: ignore[import]
from fastapi.responses import JSONResponse  # type: ignore[import]
from typing import Optional

from ecg_processor   import load_config, preprocess_ecg
from model_handler   import ModelHandler
from model_b_handler import ModelBHandler
from supabase_client import supabase

# ── Init ──────────────────────────────────────────────────────────────────────
app    = FastAPI(title="CAREDIFY AI API", version="1.0.0")
config = load_config()

WEBHOOK_SECRET = os.environ.get("WEBHOOK_SECRET", "")

# ── Modèle A ──────────────────────────────────────────────────────────────────
TFLITE_PATH = "weights/caredify_modele_a_1lead_mobile.tflite"
model_a = None
if os.path.exists(TFLITE_PATH):
    try:
        model_a = ModelHandler(model_path=TFLITE_PATH, config=config)
    except Exception as e:
        print(f"[CAREDIFY] ❌ Modèle A erreur : {e}")
else:
    print(f"[CAREDIFY] ⚠️ {TFLITE_PATH} introuvable")

# ── Modèle B ──────────────────────────────────────────────────────────────────
KERAS_B_PATH  = "weights/model_b_mlp_final.keras"
CONFIG_B_PATH = "weights/caredify_model_b_config.json"
model_b = None
if os.path.exists(KERAS_B_PATH) and os.path.exists(CONFIG_B_PATH):
    try:
        model_b = ModelBHandler(
            model_path=KERAS_B_PATH,
            config_path=CONFIG_B_PATH,
        )
    except Exception as e:
        print(f"[CAREDIFY] ⚠️ Modèle B non chargé (non bloquant) : {e}")
else:
    print(f"[CAREDIFY] ⚠️ Modèle B fichiers manquants — fonctionnement sans Modèle B")


# ── Health ────────────────────────────────────────────────────────────────────
@app.get("/health")
def health():
    return {
        "status":         "ok",
        "model_a":        config["model_name"],
        "model_b":        "CAREDIFY_ModelB_MLP" if model_b else "non chargé",
        "version":        config["version"],
        "classes":        config["classes"],
        "thresholds":     config["thresholds"],
        "model_a_loaded": model_a is not None,
        "model_b_loaded": model_b is not None,
    }


# ── Test mock (sans Supabase) ─────────────────────────────────────────────────
@app.post("/test/predict")
async def test_predict(request: Request):
    """
    Test direct sans Supabase.
    Body : {"ecg_values": [...]}
    """
    if model_a is None:
        raise HTTPException(503, "Modèle A non chargé")

    body       = await request.json()
    ecg_values = body.get("ecg_values", [])

    if len(ecg_values) < 10:
        raise HTTPException(400, "ecg_values doit contenir au moins 10 points")

    arr      = preprocess_ecg(ecg_values, config)
    result_a = model_a.predict(arr)

    if model_b:
        try:
            result_b = model_b.predict(result_a)
            return {**result_a, **result_b}
        except Exception as e:
            print(f"[CAREDIFY] ⚠️ Modèle B test erreur (fallback A) : {e}")

    return result_a


# ── Webhook Supabase ──────────────────────────────────────────────────────────
@app.post("/webhook/ecg-new")
async def ecg_webhook(
    request: Request,
    x_webhook_secret: Optional[str] = Header(None),
):
    """
    INSERT ecg_readings → Modèle A → Modèle B → UPDATE status
    pending → normal / warning / critical
    """
    if WEBHOOK_SECRET and x_webhook_secret != WEBHOOK_SECRET:
        raise HTTPException(status_code=401, detail="Webhook secret invalide")

    if model_a is None:
        raise HTTPException(503, "Modèle A non chargé")

    payload    = await request.json()
    record     = payload.get("record", {})
    ecg_id     = record.get("id")
    ecg_values = record.get("ecg_values", [])
    patient_id = record.get("patient_id")

    if not ecg_id:
        return JSONResponse({"skipped": True, "reason": "no id"})
    if len(ecg_values) < 10:
        return JSONResponse({"skipped": True, "reason": "buffer trop court"})

    # ── Métadonnées patient ───────────────────────────────────────────────────
    patient_meta = None
    try:
        if supabase and patient_id:
            row = supabase.table("patients").select(
                "age, sex, weight, bmi, bmi_category, cardiac_pathology"
            ).eq("id", patient_id).single().execute()
            if row.data:
                patient_meta = row.data
                print(f"[CAREDIFY] 👤 Patient récupéré: {patient_meta}")
    except Exception as e:
        print(f"[CAREDIFY] ⚠️ Patient meta non récupérée (non bloquant): {e}")

    # ── Modèle A ──────────────────────────────────────────────────────────────
    try:
        arr      = preprocess_ecg(ecg_values, config, patient_meta=patient_meta)
        result_a = model_a.predict(arr)
    except Exception as e:
        print(f"[CAREDIFY] ❌ Modèle A erreur : {e}")
        raise HTTPException(500, str(e))

    # ── Modèle B — raffine le score (ne déclenche pas d'alerte seul) ──────────
    final_result = result_a.copy()
    if model_b:
        try:
            result_b = model_b.predict(result_a, patient_meta=patient_meta)
            final_result["status"]         = result_b["status"]
            final_result["score_combined"] = result_b["score_combined"]
            final_result["score_b"]        = result_b["score_b"]
        except Exception as e:
            print(f"[CAREDIFY] ⚠️ Modèle B erreur — fallback Modèle A : {e}")

    # ── UPDATE Supabase ───────────────────────────────────────────────────────
    try:
        supabase.table("ecg_readings").update(
            {"status": final_result["status"]}
        ).eq("id", ecg_id).execute()
    except Exception as e:
        print(f"[CAREDIFY] ❌ UPDATE Supabase : {e}")
        raise HTTPException(500, f"Supabase UPDATE failed: {e}")

    print(
        f"[CAREDIFY] ✅ patient={patient_id} | "
        f"score_a={result_a['score']} | "
        f"score_b={final_result.get('score_b', 'N/A')} | "
        f"score_combined={final_result.get('score_combined', result_a['score'])} | "
        f"status={final_result['status']}"
    )

    return {
        "ecg_id":    ecg_id,
        "patient_id": patient_id,
        **final_result,
    }