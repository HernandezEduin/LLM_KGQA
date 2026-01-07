import requests

import json
from pathlib import Path


# ---- Config loading ----
# Expected JSON format:
# {
#   "base_url": "http://localhost:8080",
#   "api_key": "YOUR_API_KEY"
# }

CONFIG_PATH = Path(__file__).with_name("openwebui_config.json").parent / "configs" / "openwebui_config.json"

def load_api_config(path: Path) -> tuple[str, str]:
    """
    Load configuration from a JSON file.

    Args:
        path (Path): Path to the configuration file.

    Returns:
        tuple[str, str]: Base URL and API key from the configuration.

    Raises:
        FileNotFoundError: If the configuration file does not exist.
        ValueError: If required fields are missing or empty.
    """
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

def list_models(base_url: str, headers: dict) -> dict:
    """
    Fetch the list of available models from the API.

    Args:
        base_url (str): The base URL of the API.
        headers (dict): HTTP headers for the request.

    Returns:
        dict: JSON response containing the list of models.

    Raises:
        HTTPError: If the API request fails.
    """
    r = requests.get(f"{base_url}/api/models", headers=headers, timeout=30)
    if r.status_code != 200:
        print("Status:", r.status_code)
        print("Body:", r.text)
    r.raise_for_status()
    return r.json()

def chat(base_url: str, headers: dict, model: str, user_text: str) -> dict:
    """
    Send a chat message to the API and get the response.

    Args:
        base_url (str): The base URL of the API.
        headers (dict): HTTP headers for the request.
        model (str): The model ID to use for the chat.
        user_text (str): The user's input text.

    Returns:
        dict: JSON response from the API.

    Raises:
        HTTPError: If the API request fails.
    """
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": user_text}],
    }
    r = requests.post(f"{base_url}/api/chat/completions", headers=headers, json=payload, timeout=120)
    if r.status_code != 200:
        print("Status:", r.status_code)
        print("Body:", r.text)
    r.raise_for_status()
    return r.json()

def extract_model_ids(models_resp: dict) -> list[str]:
    """
    Extract model IDs from the API response.

    Args:
        models_resp (dict): JSON response containing model data.

    Returns:
        list[str]: List of model IDs.
    """
    model_ids: list[str] = []
    if isinstance(models_resp, dict) and isinstance(models_resp.get("data"), list):
        model_ids = [m.get("id") for m in models_resp["data"] if isinstance(m, dict)]
    elif isinstance(models_resp, list):
        model_ids = [m.get("id") for m in models_resp if isinstance(m, dict)]
    return [m for m in model_ids if m]

def pick_model(model_ids: list[str]) -> str:
    """
    Select a model ID based on predefined preferences.

    Args:
        model_ids (list[str]): List of available model IDs.

    Returns:
        str: Selected model ID.
    """
    # Prefer any model whose id is exactly "gemma3" or ends with "/gemma3"
    for m in model_ids:
        if m == "gemma3" or m.endswith("/gemma3"):
            return m

    # Also accept common Ollama tags like gemma3:latest, gemma3:12b, etc.
    for m in model_ids:
        if m.startswith("gemma3:"):
            return m

    # Fallback: first available
    return model_ids[0]