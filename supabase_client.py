import os

try:
    from supabase import create_client, Client  # type: ignore
except ImportError:
    from supabase_py import create_client, Client  # type: ignore

SUPABASE_URL        = os.environ.get("SUPABASE_URL", "")
SUPABASE_SERVICE_KEY = os.environ.get("SUPABASE_SERVICE_KEY", "")

if not SUPABASE_URL or not SUPABASE_SERVICE_KEY:
    print("[CAREDIFY] ⚠️ Variables Supabase manquantes — client désactivé")
    supabase = None
else:
    supabase: Client = create_client(SUPABASE_URL, SUPABASE_SERVICE_KEY)
    print("[CAREDIFY] ✅ Supabase connecté")