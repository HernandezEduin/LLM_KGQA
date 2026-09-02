import atexit
import signal
import threading
import requests
import time

import json
from pathlib import Path

from typing import Tuple, Dict, Callable, Optional

from utils.kgqa_types import APIResponse, StatusInfo

VALID_LLM_BACKENDS = {"openwebui", "ollama"}
DEFAULT_OLLAMA_URL = "http://localhost:11434"

# A timeout makes the backend/model pair unsafe for another generation until
# recovery has been confirmed. This state is process-local, matching the
# sequential benchmark process that owns the HTTP client.
_DIRTY_BACKENDS: Dict[Tuple[str, str, str], StatusInfo] = {}
_DIRTY_BACKENDS_LOCK = threading.Lock()

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


def _backend_key(base_url: str, backend: str, model: str) -> Tuple[str, str, str]:
    """Return the process-local identity used for dirty/recovery state."""
    return backend, base_url.rstrip("/"), model


def _mark_backend_dirty(
    base_url: str,
    backend: str,
    model: str,
    reason: StatusInfo,
) -> None:
    """Prevent new generations for this backend/model until recovery succeeds."""
    key = _backend_key(base_url, backend, model)
    with _DIRTY_BACKENDS_LOCK:
        _DIRTY_BACKENDS[key] = dict(reason)


def _get_backend_dirty_reason(
    base_url: str,
    backend: str,
    model: str,
) -> StatusInfo | None:
    """Return a copy of the dirty-state reason, if any."""
    key = _backend_key(base_url, backend, model)
    with _DIRTY_BACKENDS_LOCK:
        reason = _DIRTY_BACKENDS.get(key)
        return dict(reason) if reason is not None else None


def _clear_backend_dirty(base_url: str, backend: str, model: str) -> None:
    """Clear dirty state only after recovery has been verified."""
    key = _backend_key(base_url, backend, model)
    with _DIRTY_BACKENDS_LOCK:
        _DIRTY_BACKENDS.pop(key, None)


def _attach_recovery_telemetry(
    status: StatusInfo,
    recovery_info: StatusInfo | None,
) -> StatusInfo:
    """Attach pre-request recovery details to any resulting request status."""
    if recovery_info is not None:
        status["backend_recovered_before_request"] = True
        status["recovery"] = dict(recovery_info)
    return status


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


def running_models_endpoint(base_url: str, backend: str) -> str:
    """Return the Ollama running-model endpoint, directly or through OpenWebUI."""
    backend = validate_backend(backend)
    if backend == "openwebui":
        return f"{base_url}/ollama/api/ps"
    return f"{base_url}/api/ps"


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


def _running_model_ids(response: APIResponse) -> set[str]:
    """Extract model identifiers from Ollama's /api/ps response."""
    running: set[str] = set()
    if not isinstance(response, dict) or not isinstance(response.get("models"), list):
        return running
    for entry in response["models"]:
        if not isinstance(entry, dict):
            continue
        for key in ("model", "name"):
            value = entry.get(key)
            if value:
                running.add(str(value))
    return running


def _canonical_model_name(model: str) -> str:
    """Normalize the only harmless Ollama tag alias needed for state checks."""
    model = str(model).strip()
    return model[:-7] if model.endswith(":latest") else model


def _model_is_running(response: APIResponse, model: str) -> bool:
    """Check whether the selected resolved model is still resident in Ollama."""
    target = _canonical_model_name(model)
    return any(_canonical_model_name(candidate) == target for candidate in _running_model_ids(response))


def _probe_running_models(
    base_url: str,
    headers: Dict[str, str],
    backend: str,
    session: requests.Session,
    connect_timeout: int,
    read_timeout: float,
) -> APIResponse:
    """Fetch Ollama worker/model residency state."""
    with session.get(
        running_models_endpoint(base_url, backend),
        headers=headers,
        timeout=(connect_timeout, max(0.1, read_timeout)),
    ) as response:
        response.raise_for_status()
        return response.json()


def recover_backend(
    base_url: str,
    headers: Dict[str, str],
    model: str,
    connect_timeout: int = 5,
    recovery_timeout: float = 30.0,
    poll_interval: float = 1.0,
) -> StatusInfo:
    """
    Recover a dirty Ollama worker before allowing another generation.

    Recovery uses a fresh HTTP session, asks the selected model to unload with
    ``keep_alive=0``, then verifies through Ollama's ``/api/ps`` that the model is
    no longer resident. OpenWebUI uses its transparent ``/ollama/api/*`` proxy for
    the same native Ollama operations.

    A failed verification leaves the backend dirty. The caller must not submit a
    new generation in that state.
    """
    backend = _backend_from_headers(headers)
    recovery_timeout = max(float(recovery_timeout), 1.0)
    poll_interval = max(float(poll_interval), 0.1)
    start_time = time.monotonic()
    deadline = start_time + recovery_timeout
    events = []

    with requests.Session() as recovery_session:
        # First check: if the model has already disappeared since the timeout,
        # the backend is safe without submitting any additional model request.
        try:
            remaining = max(0.1, deadline - time.monotonic())
            running = _probe_running_models(
                base_url,
                headers,
                backend,
                recovery_session,
                connect_timeout,
                min(10.0, remaining),
            )
            if not _model_is_running(running, model):
                elapsed = time.monotonic() - start_time
                return {
                    "status": "success",
                    "backend": backend,
                    "backend_recovered": True,
                    "recovery_elapsed_time": elapsed,
                    "recovery_events": ["model already absent from /api/ps"],
                    "message": "Backend recovery confirmed; model is not running.",
                }
            events.append("model still present in /api/ps")
        except requests.exceptions.RequestException as exc:
            events.append(f"initial /api/ps probe failed: {type(exc).__name__}: {exc}")

        # Request a real worker reset. If the timed-out generation is still
        # occupying Ollama, this request may wait behind it; the recovery budget
        # bounds that wait. We still probe afterwards because the unload may have
        # completed even if the client-side request timed out near the boundary.
        unload_payload = {
            "model": model,
            "messages": [{"role": "user", "content": ""}],
            "stream": False,
            "keep_alive": 0,
        }
        remaining = deadline - time.monotonic()
        if remaining > 0:
            try:
                with recovery_session.post(
                    chat_endpoint(base_url, backend),
                    headers=headers,
                    json=unload_payload,
                    timeout=(connect_timeout, max(0.1, remaining)),
                ) as response:
                    response.raise_for_status()
                    events.append(f"keep_alive=0 unload returned HTTP {response.status_code}")
            except requests.exceptions.RequestException as exc:
                events.append(f"keep_alive=0 unload failed: {type(exc).__name__}: {exc}")

        # Do not infer recovery from HTTP reachability. Only clear the dirty bit
        # once /api/ps confirms that this model is no longer resident.
        while time.monotonic() < deadline:
            remaining = deadline - time.monotonic()
            try:
                running = _probe_running_models(
                    base_url,
                    headers,
                    backend,
                    recovery_session,
                    connect_timeout,
                    min(5.0, remaining),
                )
                if not _model_is_running(running, model):
                    elapsed = time.monotonic() - start_time
                    events.append("model absent from /api/ps")
                    return {
                        "status": "success",
                        "backend": backend,
                        "backend_recovered": True,
                        "recovery_elapsed_time": elapsed,
                        "recovery_events": events,
                        "message": "Backend recovery confirmed after model unload.",
                    }
                events.append("model still present in /api/ps")
            except requests.exceptions.RequestException as exc:
                events.append(f"/api/ps verification failed: {type(exc).__name__}: {exc}")

            sleep_for = min(poll_interval, max(0.0, deadline - time.monotonic()))
            if sleep_for > 0:
                time.sleep(sleep_for)

    elapsed = time.monotonic() - start_time
    return {
        "status": "backend_recovery_failed",
        "backend": backend,
        "backend_dirty": True,
        "recovery_elapsed_time": elapsed,
        "recovery_events": events,
        "message": (
            f"Could not confirm backend recovery within {recovery_timeout:g}s; "
            "no new generation was submitted."
        ),
    }


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
    use_think: bool = False,
    session: requests.Session | None = None,
) -> Tuple[APIResponse, StatusInfo]:
    """Send a non-streaming-compatible chat request to the configured backend."""
    backend = _backend_from_headers(headers)

    if timeout <= 0 or connect_timeout <= 0:
        raise ValueError("timeout and connect_timeout must be positive")
    if timeout_cooldown < 0:
        raise ValueError("timeout_cooldown must be non-negative")
    if max_output_tokens is not None and max_output_tokens <= 0:
        raise ValueError("max_output_tokens must be positive when provided")

    # A previous timeout is a hard barrier. Recover before this request rather
    # than allowing a new question to queue behind potentially abandoned work.
    dirty_reason = _get_backend_dirty_reason(base_url, backend, model)
    recovery_info: StatusInfo | None = None
    if dirty_reason is not None:
        recovery_timeout = max(30.0, min(120.0, float(timeout) * 2.0))
        recovery_info = recover_backend(
            base_url=base_url,
            headers=headers,
            model=model,
            connect_timeout=connect_timeout,
            recovery_timeout=recovery_timeout,
        )
        if recovery_info.get("status") != "success":
            recovery_info["backend_recovered_before_request"] = False
            recovery_info["dirty_reason"] = dirty_reason
            return {}, recovery_info
        _clear_backend_dirty(base_url, backend, model)

    payload = {
        "model": model,
        "messages": [{"role": "user", "content": user_text}],
        "stream": stream,
        "think": use_think,
        "options": {"num_ctx": context_window},
    }

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
            status: StatusInfo = {
                "status": "success",
                "backend": backend,
                "backend_dirty": False,
                "elapsed_time": elapsed_time,
                "message": "Request successful",
            }
            return r.json(), _attach_recovery_telemetry(status, recovery_info)
    except requests.exceptions.ConnectTimeout as e:
        elapsed_time = time.monotonic() - start_time
        status = {
            "status": "timeout",
            "backend": backend,
            "backend_dirty": True,
            "timeout_type": "connect",
            "elapsed_time": elapsed_time,
            "message": f"Connection timed out after {connect_timeout} seconds: {e}",
        }
        _attach_recovery_telemetry(status, recovery_info)
        _mark_backend_dirty(base_url, backend, model, status)
        return {}, status
    except requests.exceptions.ReadTimeout as e:
        # Closing the timed-out client request is not treated as proof that the
        # model stopped. Mark dirty now; the next chat must unload and verify the
        # worker through /api/ps before it can send another generation.
        elapsed_time = time.monotonic() - start_time
        if timeout_cooldown > 0:
            time.sleep(timeout_cooldown)

        status = {
            "status": "timeout",
            "backend": backend,
            "backend_dirty": True,
            "timeout_type": "read",
            "elapsed_time": elapsed_time,
            "cooldown_seconds": timeout_cooldown,
            "message": (
                f"No response data received for {timeout} seconds: {e}. "
                "Backend marked dirty; recovery is required before the next generation."
            ),
        }
        _attach_recovery_telemetry(status, recovery_info)
        _mark_backend_dirty(base_url, backend, model, status)
        return {}, status
    except requests.exceptions.Timeout as e:
        elapsed_time = time.monotonic() - start_time
        status = {
            "status": "timeout",
            "backend": backend,
            "backend_dirty": True,
            "timeout_type": "unknown",
            "elapsed_time": elapsed_time,
            "message": (
                f"Request timed out: {e}. Backend marked dirty; recovery is required "
                "before the next generation."
            ),
        }
        _attach_recovery_telemetry(status, recovery_info)
        _mark_backend_dirty(base_url, backend, model, status)
        return {}, status
    except requests.exceptions.ConnectionError as e:
        elapsed_time = time.monotonic() - start_time
        print("Error: Connection error occurred.", str(e))
        status = {
            "status": "connection_error",
            "backend": backend,
            "elapsed_time": elapsed_time,
            "message": str(e),
        }
        return {}, _attach_recovery_telemetry(status, recovery_info)
    except Exception as e:
        elapsed_time = time.monotonic() - start_time
        print("Error: An unexpected error occurred.", str(e))
        status = {
            "status": "error",
            "backend": backend,
            "elapsed_time": elapsed_time,
            "message": str(e),
        }
        return {}, _attach_recovery_telemetry(status, recovery_info)


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
