import os

try:
    # noqa: E402 - supabase package may not be installed in some environments
    from supabase import create_client, Client  # type: ignore
except ImportError:
    # Fallback for environments using the older supabase_py package
    from supabase_py import create_client, Client  # type: ignore

SUPABASE_URL = os.environ["SUPABASE_URL"]
SUPABASE_SERVICE_KEY = os.environ["SUPABASE_SERVICE_KEY"]

# Service role → bypasse la RLS pour UPDATE depuis l'API
supabase: Client = create_client(SUPABASE_URL, SUPABASE_SERVICE_KEY)