import atexit
import signal
import requests
import time

import json
from pathlib import Path

from typing import Tuple, Dict, Callable, List, Optional

from utils.kgqa_types import APIResponse, StatusInfo

# ---- Config loading ----
# Expected JSON format:
# {
#   "base_url": "http://localhost:8080",
#   "api_key": "YOUR_API_KEY"
# }

CONFIG_PATH = Path(__file__).with_name("openwebui_config.json").parent / "configs" / "openwebui_config.json"

def load_api_config(path: Path) -> Tuple[str, str]:
    """
    Load configuration from a JSON file.

    Args:
        path (Path): Path to the configuration file.

    Returns:
        Tuple[str, str]: Base URL and API key from the configuration.

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

def list_models(base_url: str, headers: Dict[str, str]) -> APIResponse:
    """
    Fetch the list of available models from the API.

    Args:
        base_url (str): The base URL of the API.
        headers (Dict[str, str]): HTTP headers for the request.

    Returns:
        APIResponse: JSON response containing the list of models.

    Raises:
        HTTPError: If the API request fails.
    """
    r = requests.get(f"{base_url}/api/models", headers=headers, timeout=30)
    if r.status_code != 200:
        print("Status:", r.status_code)
        print("Body:", r.text)
    r.raise_for_status()
    return r.json()

def chat(
    base_url: str, 
    headers: Dict[str, str], 
    model: str, 
    user_text: str,
    stream: bool = False,
    context_window: int = 4096,
    seed: int | None = None,
    temperature: float | None = None,
    timeout: int = 120
) -> Tuple[APIResponse, StatusInfo]:
    """
    Send a chat message to the API and get the response.

    Args:
        base_url (str): The base URL of the API.
        headers (Dict[str, str]): HTTP headers for the request.
        model (str): The model ID to use for the chat.
        user_text (str): The user's input text.
        stream (bool): Whether to use streaming responses.
        context_window (int): Context window size for the model.
        seed (int | None): Optional random seed for the request.
        temperature (float | None): Optional sampling temperature for the request.
        timeout (int): Timeout in seconds for the API request.

    Returns:
        Tuple[APIResponse, StatusInfo]: JSON response and status information including success, timeout, or error.

    Raises:
        HTTPError: If the API request fails.
    """
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": user_text}],
        "stream": stream,
        "options": { "num_ctx": context_window }
    }

    if seed is not None:
        payload["options"]["seed"] = int(seed)
    if temperature is not None:
        payload["options"]["temperature"] = float(temperature)

    start_time = time.time()
    try:
        r = requests.post(f"{base_url}/ollama/api/chat", headers=headers, json=payload, timeout=timeout)
        if r.status_code != 200:
            print("Status:", r.status_code)
            print("Body:", r.text)
        r.raise_for_status()
        elapsed_time = time.time() - start_time
        return r.json(), {"status": "success", "elapsed_time": elapsed_time, "message": "Request successful"}
    except requests.exceptions.Timeout:
        elapsed_time = time.time() - start_time
        return {}, {"status": "timeout", "elapsed_time": elapsed_time, "message": f"Request timed out after {timeout} seconds"}
    except requests.exceptions.ConnectionError as e:
        elapsed_time = time.time() - start_time
        print("Error: Connection error occurred.", str(e))
        return {}, {"status": "connection_error", "elapsed_time": elapsed_time, "message": str(e)}
    except Exception as e:
        elapsed_time = time.time() - start_time
        print("Error: An unexpected error occurred.", str(e))
        return {}, {"status": "error", "elapsed_time": elapsed_time, "message": str(e)}

def extract_model_ids(models_resp: APIResponse) -> Dict[str, str]:
    """
    Extract model IDs from the API response.

    Args:
        models_resp (APIResponse): JSON response containing model data.

    Returns:
        Dict[str, str]: Model IDs mapped to their names.
    """
    model_ids: Dict[str, str] = {}

    if isinstance(models_resp, dict) and isinstance(models_resp.get("data"), list):
        for m in models_resp["data"]:
            if isinstance(m, dict):
                mid = m.get("id")
                name = m.get("name")
                if mid:
                    model_ids[name] = mid
    elif isinstance(models_resp, list):
        for m in models_resp:
            if isinstance(m, dict):
                mid = m.get("id")
                name = m.get("name")
                if mid:
                    model_ids[name] = mid
    return model_ids

def pick_model(model_ids: Dict[str, str], choice: str = 'gemma3') -> str:
    """
    Select a model ID based on predefined preferences.

    Args:
        model_ids (Dict[str, str]): Dictionary of model IDs mapped to their names.

    Returns:
        str: Selected model ID.
    """

    for m in model_ids:
        if m == choice:
            return model_ids[m]
    
    raise ValueError(f"Model '{choice}' not found in available models.")

    # Fallback: first available
    return list(model_ids.values())[0]

def unload_model(
    base_url: str,
    headers: Dict[str, str],
    model: str,
    timeout: int = 30,
) -> None:
    """
    Ask Ollama (behind OpenWebUI) to unload the model from memory.

    This uses Ollama's `keep_alive=0` behavior to release RAM/VRAM.
    It is safe to call multiple times and should be best-effort.
    """
    try:
        # Ollama accepts keep_alive on /api/chat (and /api/generate).
        payload = {
            "model": model,
            "messages": [{"role": "user", "content": ""}],
            "stream": False,
            "keep_alive": 0,  # unload immediately
        }
        r = requests.post(
            f"{base_url}/ollama/api/chat",
            headers=headers,
            json=payload,
            timeout=timeout,
        )
        # Best-effort cleanup; don't raise if shutdown path is already in progress.
        try:
            r.close()
        finally:
            pass
    except Exception:
        # Swallow everything: unloading is a cleanup step and should not mask the real exit.
        pass


def register_cleanup_handlers(
    base_url: str,
    headers: Dict[str, str],
    model: str,
    on_cleanup: Optional[Callable[[], None]] = None,
) -> None:
    """
    Ensure cleanup runs on normal exit, Ctrl+C, and SIGTERM.

    - atexit: runs on normal interpreter shutdown
    - signal handlers: runs on SIGINT/SIGTERM to unload ASAP
    """
    def _cleanup() -> None:
        unload_model(base_url, headers, model)
        if on_cleanup is not None:
            try:
                on_cleanup()
            except Exception:
                pass

    # 1) Normal interpreter shutdown
    atexit.register(_cleanup)

    # 2) Signals (Ctrl+C is SIGINT; many schedulers send SIGTERM)
    def _handler(signum, frame) -> None:
        _cleanup()
        raise KeyboardInterrupt  # preserve expected Ctrl+C semantics

    for sig in (signal.SIGINT, signal.SIGTERM):
        try:
            signal.signal(sig, _handler)
        except Exception:
            # Not all environments allow setting handlers (e.g., some notebooks/threads)
            pass