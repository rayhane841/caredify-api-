import os
import json
from fastapi import FastAPI, Request, HTTPException, Header  # type: ignore[import]
from fastapi.responses import JSONResponse  # type: ignore[import]
from typing import Optional

from ecg_processor   import load_config, preprocess_ecg
from model_handler   import ModelHandler
from supabase_client import supabase

# ── Init ──────────────────────────────────────────────────────────────────────
app    = FastAPI(title="CAREDIFY AI API", version="1.0.0")
config = load_config()
model  = ModelHandler(config=config)

WEBHOOK_SECRET = os.environ.get("WEBHOOK_SECRET", "")


# ── Health ────────────────────────────────────────────────────────────────────
@app.get("/health")
def health():
    return {
        "status":  "ok",
        "model":   config["model_name"],
        "version": config["version"],
        "classes": config["classes"],
        "thresholds": config["thresholds"],
    }


# ── Test mock local (sans Supabase) ──────────────────────────────────────────
@app.post("/test/predict")
async def test_predict(request: Request):
    """
    Endpoint de test : envoie directement des ecg_values JSON.
    Ne touche pas Supabase. Utilisé pour valider le modèle en local.

    Body JSON attendu :
    {
      "ecg_values": [0.1, -0.2, 0.5, ...]   // n'importe quelle longueur
    }
    """
    body = await request.json()
    ecg_values = body.get("ecg_values", [])

    if len(ecg_values) < 10:
        raise HTTPException(400, "ecg_values doit contenir au moins 10 points")

    arr    = preprocess_ecg(ecg_values, config)
    result = model.predict(arr)
    return result


# ── Webhook principal Supabase ────────────────────────────────────────────────
@app.post("/webhook/ecg-new")
async def ecg_webhook(
    request: Request,
    x_webhook_secret: Optional[str] = Header(None),
):
    """
    Reçoit l'événement INSERT de Supabase sur ecg_readings.
    1. Vérifie le secret
    2. Prétraite les ecg_values
    3. Lance l'inférence TFLite
    4. UPDATE status dans Supabase : 'pending' → 'normal'/'warning'/'critical'
    """
    # Vérification secret
    if WEBHOOK_SECRET and x_webhook_secret != WEBHOOK_SECRET:
        raise HTTPException(status_code=401, detail="Webhook secret invalide")

    payload = await request.json()

    # Supabase webhook format : {"type": "INSERT", "record": {...}}
    record = payload.get("record", {})

    ecg_id     = record.get("id")
    ecg_values = record.get("ecg_values", [])
    patient_id = record.get("patient_id")
    heart_rate = record.get("heart_rate", 0)

    # Validation minimale
    if not ecg_id:
        return JSONResponse({"skipped": True, "reason": "no id"})
    if len(ecg_values) < 10:
        return JSONResponse({"skipped": True, "reason": "buffer trop court"})

    # Inférence
    try:
        arr    = preprocess_ecg(ecg_values, config)
        result = model.predict(arr)
    except Exception as e:
        # On ne bloque pas Supabase, on log et on sort
        print(f"[CAREDIFY] ❌ Erreur inférence: {e}")
        raise HTTPException(500, str(e))

    # UPDATE Supabase : pending → status réel
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

    return {
        "ecg_id":    ecg_id,
        "patient_id": patient_id,
        **result,
    }