import atexit
import signal
import requests
import time

import json
from pathlib import Path

from typing import Tuple, Dict, Callable, Optional

from utils.kgqa_types import APIResponse, StatusInfo

VALID_LLM_BACKENDS = {"openwebui", "ollama"}
DEFAULT_OLLAMA_URL = "http://localhost:11434"

# ---- Config loading ----
# Existing OpenWebUI configs remain valid and remain the default:
# {
#   "base_url": "http://localhost:8080",
#   "api_key": "YOUR_API_KEY"
# }
#
# For direct Ollama, select it in the same config file:
# {
#   "backend": "ollama",
#   "ollama_url": "http://localhost:11434"
# }
#
# The OpenWebUI fields may remain in the file when backend="ollama", which makes
# switching between the two backends a one-line config change. Direct local
# Ollama does not require an API key.
CONFIG_PATH = Path(__file__).with_name("openwebui_config.json").parent / "configs" / "openwebui_config.json"


def validate_backend(backend: str) -> str:
    """Validate and normalize an LLM backend name."""
    normalized = str(backend).strip().lower()
    if normalized not in VALID_LLM_BACKENDS:
        raise ValueError(
            f"Unsupported LLM backend '{backend}'. Valid options are: {sorted(VALID_LLM_BACKENDS)}"
        )
    return normalized


def _backend_from_headers(headers: Dict[str, str]) -> str:
    """Infer the configured backend from the auth header produced by the base client."""
    authorization = str(headers.get("Authorization", "")).strip()
    return "ollama" if authorization in {"", "Bearer"} else "openwebui"


def load_api_config(path: Path) -> Tuple[str, str]:
    """
    Load configuration from a JSON file.

    Existing configuration files without a ``backend`` field use OpenWebUI.
    Set ``backend`` to ``ollama`` to use native Ollama directly. In that mode,
    ``ollama_url`` defaults to ``http://localhost:11434`` and no API key is
    required.

    Returns:
        Tuple[str, str]: Selected backend base URL and API key. Direct Ollama
        returns an empty API key so the existing base client remains compatible.
    """
    if not path.exists():
        raise FileNotFoundError(
            f"Config file not found: {path}\n"
            "Create it with either:\n"
            '{\n  "base_url": "http://localhost:8080",\n  "api_key": "YOUR_API_KEY"\n}\n'
            "or for direct Ollama:\n"
            '{\n  "backend": "ollama",\n  "ollama_url": "http://localhost:11434"\n}'
        )

    with path.open("r", encoding="utf-8") as f:
        cfg = json.load(f)

    backend = validate_backend(cfg.get("backend", "openwebui"))
    if backend == "ollama":
        base_url = str(cfg.get("ollama_url", DEFAULT_OLLAMA_URL)).strip().rstrip("/")
        if not base_url:
            raise ValueError(f'Missing/empty "ollama_url" in {path}')
        return base_url, ""

    base_url = str(cfg.get("base_url", "")).strip().rstrip("/")
    api_key = str(cfg.get("api_key", "")).strip()

    if not base_url:
        raise ValueError(f'Missing/empty "base_url" in {path}')
    if not api_key:
        raise ValueError(f'Missing/empty "api_key" in {path}')

    return base_url, api_key


def model_list_endpoint(base_url: str, backend: str) -> str:
    """Return the backend-specific model-list endpoint."""
    backend = validate_backend(backend)
    if backend == "openwebui":
        return f"{base_url}/api/models"
    return f"{base_url}/api/tags"


def chat_endpoint(base_url: str, backend: str) -> str:
    """Return the backend-specific chat endpoint."""
    backend = validate_backend(backend)
    if backend == "openwebui":
        return f"{base_url}/ollama/api/chat"
    return f"{base_url}/api/chat"


def list_models(
    base_url: str,
    headers: Dict[str, str],
    session: requests.Session | None = None,
) -> APIResponse:
    """Fetch available models from OpenWebUI or native Ollama."""
    backend = _backend_from_headers(headers)
    requester = session or requests
    with requester.get(
        model_list_endpoint(base_url, backend),
        headers=headers,
        timeout=(5, 30),
    ) as r:
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
    timeout: int = 120,
    connect_timeout: int = 5,
    timeout_cooldown: float = 5.0,
    max_output_tokens: int | None = 256,
    session: requests.Session | None = None,
) -> Tuple[APIResponse, StatusInfo]:
    """Send a non-streaming-compatible chat request to the configured backend."""
    backend = _backend_from_headers(headers)
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": user_text}],
        "stream": stream,
        "options": {"num_ctx": context_window},
    }

    if timeout <= 0 or connect_timeout <= 0:
        raise ValueError("timeout and connect_timeout must be positive")
    if timeout_cooldown < 0:
        raise ValueError("timeout_cooldown must be non-negative")
    if max_output_tokens is not None and max_output_tokens <= 0:
        raise ValueError("max_output_tokens must be positive when provided")

    if seed is not None:
        payload["options"]["seed"] = int(seed)
    if temperature is not None:
        payload["options"]["temperature"] = float(temperature)
    if max_output_tokens is not None:
        payload["options"]["num_predict"] = int(max_output_tokens)

    requester = session or requests
    start_time = time.monotonic()
    try:
        with requester.post(
            chat_endpoint(base_url, backend),
            headers=headers,
            json=payload,
            timeout=(connect_timeout, timeout),
        ) as r:
            if r.status_code != 200:
                print("Status:", r.status_code)
                print("Body:", r.text)
            r.raise_for_status()
            elapsed_time = time.monotonic() - start_time
            return r.json(), {
                "status": "success",
                "backend": backend,
                "elapsed_time": elapsed_time,
                "message": "Request successful",
            }
    except requests.exceptions.ConnectTimeout as e:
        elapsed_time = time.monotonic() - start_time
        return {}, {
            "status": "timeout",
            "backend": backend,
            "timeout_type": "connect",
            "elapsed_time": elapsed_time,
            "message": f"Connection timed out after {connect_timeout} seconds: {e}",
        }
    except requests.exceptions.ReadTimeout as e:
        # Exiting the response context closes the client-side socket. Give the
        # selected backend time to observe the disconnect before submitting
        # another generation, then verify that its HTTP API remains reachable.
        elapsed_time = time.monotonic() - start_time
        if timeout_cooldown > 0:
            time.sleep(timeout_cooldown)

        recovery = "health probe not attempted"
        backend_label = "OpenWebUI" if backend == "openwebui" else "Ollama"
        try:
            with requester.get(
                model_list_endpoint(base_url, backend),
                headers=headers,
                timeout=(connect_timeout, min(10, timeout)),
            ) as probe:
                recovery = f"{backend_label} health probe returned HTTP {probe.status_code}"
        except requests.exceptions.RequestException as probe_error:
            recovery = (
                f"{backend_label} health probe failed: "
                f"{type(probe_error).__name__}: {probe_error}"
            )

        return {}, {
            "status": "timeout",
            "backend": backend,
            "timeout_type": "read",
            "elapsed_time": elapsed_time,
            "cooldown_seconds": timeout_cooldown,
            "recovery": recovery,
            "message": (
                f"No response data received for {timeout} seconds: {e}. "
                f"Waited {timeout_cooldown:g}s after closing the request; {recovery}"
            ),
        }
    except requests.exceptions.Timeout as e:
        elapsed_time = time.monotonic() - start_time
        return {}, {
            "status": "timeout",
            "backend": backend,
            "timeout_type": "unknown",
            "elapsed_time": elapsed_time,
            "message": f"Request timed out: {e}",
        }
    except requests.exceptions.ConnectionError as e:
        elapsed_time = time.monotonic() - start_time
        print("Error: Connection error occurred.", str(e))
        return {}, {
            "status": "connection_error",
            "backend": backend,
            "elapsed_time": elapsed_time,
            "message": str(e),
        }
    except Exception as e:
        elapsed_time = time.monotonic() - start_time
        print("Error: An unexpected error occurred.", str(e))
        return {}, {
            "status": "error",
            "backend": backend,
            "elapsed_time": elapsed_time,
            "message": str(e),
        }


def extract_model_ids(models_resp: APIResponse) -> Dict[str, str]:
    """Extract model IDs from OpenWebUI or native Ollama model-list responses."""
    model_ids: Dict[str, str] = {}

    if isinstance(models_resp, dict) and isinstance(models_resp.get("data"), list):
        # OpenWebUI /api/models response.
        for m in models_resp["data"]:
            if isinstance(m, dict):
                mid = m.get("id")
                name = m.get("name") or mid
                if mid:
                    model_ids[str(name)] = str(mid)
    elif isinstance(models_resp, dict) and isinstance(models_resp.get("models"), list):
        # Native Ollama /api/tags response.
        for m in models_resp["models"]:
            if isinstance(m, dict):
                mid = m.get("model") or m.get("name")
                name = m.get("name") or m.get("model")
                if mid:
                    model_ids[str(name)] = str(mid)
    elif isinstance(models_resp, list):
        for m in models_resp:
            if isinstance(m, dict):
                mid = m.get("id")
                name = m.get("name") or mid
                if mid:
                    model_ids[str(name)] = str(mid)
    return model_ids


def _model_tokens(value: str) -> set[str]:
    """Tokenize a model tag for logical variant matching."""
    normalized = value.lower().replace(":", "-").replace("_", "-")
    return {token for token in normalized.split("-") if token}


def _quantization_tokens(tokens: set[str]) -> set[str]:
    """Return compact quantization markers such as q4 or q5 from model tokens."""
    return {
        token
        for token in tokens
        if token.startswith("q") and token[1:].isdigit()
    }


def pick_model(model_ids: Dict[str, str], choice: str = "gemma3") -> str:
    """
    Select a model by exact name/ID or by the repository's logical model variant.

    Native Ollama tags often expose parameter size and quantization details that
    OpenWebUI hides behind aliases. For example, ``qwen2.5:instruct`` should map
    to ``qwen2.5:7b-instruct`` rather than becoming ambiguous with its Q4/Q5
    variants. Likewise, a plain ``qwen2.5`` request should not silently select an
    instruct-tuned or quantized tag.
    """
    for name, model_id in model_ids.items():
        if name == choice or model_id == choice:
            return model_id

    requested_tokens = _model_tokens(choice)
    requested_quantization = _quantization_tokens(requested_tokens)
    wants_instruct = "instruct" in requested_tokens
    candidates = []

    for name, model_id in model_ids.items():
        candidate_tokens = _model_tokens(f"{name}-{model_id}")
        if not requested_tokens.issubset(candidate_tokens):
            continue

        candidate_quantization = _quantization_tokens(candidate_tokens)
        if requested_quantization:
            # A requested q4/q5/etc. must resolve to exactly that quantization.
            if candidate_quantization != requested_quantization:
                continue
        elif candidate_quantization:
            # An unquantized logical request must not select a quantized model.
            continue

        if wants_instruct:
            if "instruct" not in candidate_tokens:
                continue
        elif "instruct" in candidate_tokens:
            # Plain model aliases refer to the non-instruct/default variant.
            continue

        candidates.append(model_id)

    unique_candidates = sorted(set(candidates))
    if len(unique_candidates) == 1:
        return unique_candidates[0]

    detail = (
        f" Matching candidates: {unique_candidates}."
        if unique_candidates
        else ""
    )
    raise ValueError(
        f"Model '{choice}' not found unambiguously in available models.{detail} "
        f"Available model names: {sorted(model_ids)}"
    )


def unload_model(
    base_url: str,
    headers: Dict[str, str],
    model: str,
    timeout: int = 30,
) -> None:
    """Ask Ollama, directly or through OpenWebUI, to unload a model from memory."""
    try:
        backend = _backend_from_headers(headers)
        payload = {
            "model": model,
            "messages": [{"role": "user", "content": ""}],
            "stream": False,
            "keep_alive": 0,
        }
        r = requests.post(
            chat_endpoint(base_url, backend),
            headers=headers,
            json=payload,
            timeout=timeout,
        )
        r.close()
    except Exception:
        # Unloading is best-effort cleanup and should never mask the real exit.
        pass


def register_cleanup_handlers(
    base_url: str,
    headers: Dict[str, str],
    model: str,
    on_cleanup: Optional[Callable[[], None]] = None,
) -> None:
    """Ensure backend-aware cleanup runs on normal exit, Ctrl+C, and SIGTERM."""
    def _cleanup() -> None:
        unload_model(base_url, headers, model)
        if on_cleanup is not None:
            try:
                on_cleanup()
            except Exception:
                pass

    atexit.register(_cleanup)

    def _handler(signum, frame) -> None:
        _cleanup()
        raise KeyboardInterrupt

    for sig in (signal.SIGINT, signal.SIGTERM):
        try:
            signal.signal(sig, _handler)
        except Exception:
            pass