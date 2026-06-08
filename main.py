import os
from fastapi import FastAPI, Request, HTTPException, Header  # type: ignore[import]
from fastapi.responses import JSONResponse  # type: ignore[import]
from typing import Optional

from ecg_processor   import load_config, preprocess_ecg
from model_handler   import ModelHandler
from supabase_client import supabase

# ── Init ──────────────────────────────────────────────────────────────────────
app    = FastAPI(title="CAREDIFY AI API", version="1.0.0")
config = load_config()

WEBHOOK_SECRET = os.environ.get("WEBHOOK_SECRET", "")

# Chargement du modèle
TFLITE_PATH = "weights/caredify_modele_a_1lead_mobile.tflite"
model = None

if os.path.exists(TFLITE_PATH):
    try:
        model = ModelHandler(model_path=TFLITE_PATH, config=config)
        print(f"[CAREDIFY] ✅ Modèle chargé : {TFLITE_PATH}")
    except Exception as e:
        print(f"[CAREDIFY] ❌ Erreur chargement modèle : {e}")
else:
    print(f"[CAREDIFY] ⚠️ {TFLITE_PATH} introuvable")


# ── Health ────────────────────────────────────────────────────────────────────
@app.get("/health")
def health():
    return {
        "status":       "ok",
        "model":        config["model_name"],
        "version":      config["version"],
        "classes":      config["classes"],
        "thresholds":   config["thresholds"],
        "model_loaded": model is not None,
    }


# ── Test mock (sans Supabase) ─────────────────────────────────────────────────
@app.post("/test/predict")
async def test_predict(request: Request):
    """
    Test direct sans Supabase.
    Body : {"ecg_values": [...]}
    """
    if model is None:
        raise HTTPException(503, "Modèle non chargé")

    body       = await request.json()
    ecg_values = body.get("ecg_values", [])

    if len(ecg_values) < 10:
        raise HTTPException(400, "ecg_values doit contenir au moins 10 points")

    arr    = preprocess_ecg(ecg_values, config)
    result = model.predict(arr)
    return result


# ── Webhook Supabase ──────────────────────────────────────────────────────────
@app.post("/webhook/ecg-new")
async def ecg_webhook(
    request: Request,
    x_webhook_secret: Optional[str] = Header(None),
):
    """
    INSERT ecg_readings → analyse TFLite → UPDATE status
    pending → normal / warning / critical
    """
    # Vérification secret
    if WEBHOOK_SECRET and x_webhook_secret != WEBHOOK_SECRET:
        raise HTTPException(status_code=401, detail="Webhook secret invalide")

    if model is None:
        raise HTTPException(503, "Modèle non chargé")

    payload    = await request.json()
    record     = payload.get("record", {})
    ecg_id     = record.get("id")
    ecg_values = record.get("ecg_values", [])
    patient_id = record.get("patient_id")

    # Validation
    if not ecg_id:
        return JSONResponse({"skipped": True, "reason": "no id"})
    if len(ecg_values) < 10:
        return JSONResponse({"skipped": True, "reason": "buffer trop court"})

    # ── Récupérer métadonnées patient depuis Supabase ─────────────────────────
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

    # ── Preprocessing + inférence ─────────────────────────────────────────────
    try:
        arr    = preprocess_ecg(ecg_values, config, patient_meta=patient_meta)
        result = model.predict(arr)
    except Exception as e:
        print(f"[CAREDIFY] ❌ Erreur inférence: {e}")
        raise HTTPException(500, str(e))

    # ── UPDATE Supabase ───────────────────────────────────────────────────────
    try:
        supabase.table("ecg_readings").update(
            {"status": result["status"]}
        ).eq("id", ecg_id).execute()
    except Exception as e:
        print(f"[CAREDIFY] ❌ Erreur UPDATE Supabase: {e}")
        raise HTTPException(500, f"Supabase UPDATE failed: {e}")

    print(
        f"[CAREDIFY] ✅ patient={patient_id} | "
        f"score={result['score']} | "
        f"class={result['predicted_class']} | "
        f"status={result['status']}"
    )

    return {"ecg_id": ecg_id, "patient_id": patient_id, **result}