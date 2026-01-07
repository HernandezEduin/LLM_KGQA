import json
from pathlib import Path
import requests

# ---- Config loading ----
# Expected JSON format:
# {
#   "base_url": "http://localhost:8080",
#   "api_key": "YOUR_API_KEY"
# }
CONFIG_PATH = Path(__file__).with_name("openwebui_config.json").parent / "configs" / "openwebui_config.json"

def load_config(path: Path = CONFIG_PATH) -> tuple[str, str]:
    if not path.exists():
        raise FileNotFoundError(
            f"Config file not found: {path}\n"
            "Create it with:\n"
            '{\n  "base_url": "http://localhost:8080",\n  "api_key": "YOUR_API_KEY"\n}'
        )

    with path.open("r", encoding="utf-8") as f:
        cfg = json.load(f)

    base_url = str(cfg.get("base_url", "")).strip().rstrip("/")
    api_key = str(cfg.get("api_key", "")).strip()

    if not base_url:
        raise ValueError(f'Missing/empty "base_url" in {path}')
    if not api_key:
        raise ValueError(f'Missing/empty "api_key" in {path}')

    return base_url, api_key


BASE_URL, API_KEY = load_config()

HEADERS = {
    "Authorization": f"Bearer {API_KEY}",
    "Content-Type": "application/json",
}

def list_models():
    r = requests.get(f"{BASE_URL}/api/models", headers=HEADERS, timeout=30)
    if r.status_code != 200:
        print("Status:", r.status_code)
        print("Body:", r.text)
    r.raise_for_status()
    return r.json()

def chat(model: str, user_text: str):
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": user_text}],
    }
    r = requests.post(f"{BASE_URL}/api/chat/completions", headers=HEADERS, json=payload, timeout=120)
    if r.status_code != 200:
        print("Status:", r.status_code)
        print("Body:", r.text)
    r.raise_for_status()
    return r.json()

def extract_model_ids(models_resp) -> list[str]:
    # Common patterns: {"data":[{"id":"..."}]} or [{"id":"..."}] etc.
    model_ids: list[str] = []
    if isinstance(models_resp, dict) and isinstance(models_resp.get("data"), list):
        model_ids = [m.get("id") for m in models_resp["data"] if isinstance(m, dict)]
    elif isinstance(models_resp, list):
        model_ids = [m.get("id") for m in models_resp if isinstance(m, dict)]
    return [m for m in model_ids if m]

def pick_model(model_ids: list[str]) -> str:
    # Prefer any model whose id is exactly "gemma3" or ends with "/gemma3"
    # (some providers namespace models)
    for m in model_ids:
        if m == "gemma3" or m.endswith("/gemma3"):
            return m

    # Also accept common Ollama tags like gemma3:latest, gemma3:12b, etc.
    for m in model_ids:
        if m.startswith("gemma3:"):
            return m

    # Fallback: first available
    return model_ids[0]

if __name__ == "__main__":
    models_resp = list_models()
    model_ids = extract_model_ids(models_resp)

    if not model_ids:
        raise RuntimeError(f"Couldn't parse model list response: {models_resp}")

    print("Available models:")
    for i, mid in enumerate(model_ids, start=1):
        print(f"  {i:>2}. {mid}")

    chosen = pick_model(model_ids)
    print("\nUsing model:", chosen)

    out = chat(chosen, "Write a short Python function that computes gcd(a,b).")
    print("\nResponse JSON:")
    print(out["choices"][0]["message"]["content"])
