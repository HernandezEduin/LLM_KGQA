import atexit
import signal
import threading
from pathlib import Path

import requests

from model.constants import (
    context_window_limits,
    has_instruct_versions,
    has_quantized_versions,
    has_think_option,
    valid_models,
)
from typing import Dict, Tuple

from utils.api_utils import (
    chat,
    extract_model_ids,
    list_models,
    load_api_config,
    pick_model,
    register_cleanup_handlers,
    unload_model,
)

from utils.kgqa_types import (
    APIResponse,
    EntityId,
    EntityTitleMap,
    RelationId,
    RelationTitleMap,
    StatusInfo,
    Triplet,
)


# Durations: often in nanoseconds for Ollama-style stats
def ns_to_s(x: object) -> float | None:
    """Convert nanoseconds to seconds, returning None when unavailable."""
    try:
        return float(x) / 1e9
    except Exception:
        return None


class BaseLLMKGQAClient:
    """Shared LLM API client for KGQA experiments."""

    @staticmethod
    def _format_mapped_reference(identifier: str, title_map: Dict[str, str]) -> str:
        """Format a KG identifier with its title only when a real mapping exists.

        Args:
            identifier (str): Raw entity or relation identifier.
            title_map (Dict[str, str]): Optional mapping from identifiers to readable titles.

        Returns:
            str: ``title (identifier)`` when mapped to a different title, otherwise ``identifier``.
        """
        if not title_map:
            return identifier
        title = title_map.get(identifier, identifier)
        if title == identifier:
            return identifier
        return f"{title} ({identifier})"

    @staticmethod
    def _format_entity_reference(entity: EntityId, entity_title: EntityTitleMap) -> str:
        """Format an entity for prompts without duplicating identity labels.

        Args:
            entity (EntityId): Raw entity identifier.
            entity_title (EntityTitleMap): Optional entity title mapping.

        Returns:
            str: Readable prompt reference for the entity.
        """
        return BaseLLMKGQAClient._format_mapped_reference(entity, entity_title)

    @staticmethod
    def _format_relation_reference(relation: RelationId, relation_title: RelationTitleMap) -> str:
        """Format a relation for prompts without duplicating identity labels.

        Args:
            relation (RelationId): Raw relation identifier.
            relation_title (RelationTitleMap): Optional relation title mapping.

        Returns:
            str: Readable prompt reference for the relation.
        """
        return BaseLLMKGQAClient._format_mapped_reference(relation, relation_title)

    @staticmethod
    def _format_triplet(
        triplet: Triplet,
        entity_title: EntityTitleMap,
        relation_title: RelationTitleMap,
    ) -> str:
        """Format one KG triplet for prompts without duplicating identity labels.

        Args:
            triplet (Triplet): Raw KG triplet as head, relation, tail.
            entity_title (EntityTitleMap): Optional entity title mapping.
            relation_title (RelationTitleMap): Optional relation title mapping.

        Returns:
            str: Prompt-ready triplet text.
        """
        head, relation, tail = triplet
        head_str = BaseLLMKGQAClient._format_entity_reference(head, entity_title)
        relation_str = BaseLLMKGQAClient._format_relation_reference(relation, relation_title)
        tail_str = BaseLLMKGQAClient._format_entity_reference(tail, entity_title)
        return f"({head_str}, {relation_str}, {tail_str})"

    def __init__(
        self,
        config_path: Path,
        model_choice: str = "gemma3",
        use_instruct: bool = False,
        use_quantized: bool = False,
        quantization_bits: int = 4,
        context_window: int = 4096,
        seed: int | None = None,
        temperature: float | None = None,
        timeout: int = 120,
        connect_timeout: int = 5,
        timeout_cooldown: float = 5.0,
        max_output_tokens: int | None = 256,
        use_think: bool = False,
        debug: bool = False,
    ) -> None:
        """
        Initialize the LLM_KGQA_Client with configuration.

        Args:
            config_path (Path): Path to the configuration file.
            model_choice (str): Default model to use for the LLM API.
            use_instruct (bool): Whether to use the instruction-tuned version of the model.
            use_quantized (bool): Whether to use the quantized version of the model.
            quantization_bits (int): Number of bits for quantization (if using quantized model).
            context_window (int): Context window size for the model.
            seed (int | None): Optional random seed for the requests.
            temperature (float | None): Optional sampling temperature for the requests.
            timeout (int): Timeout in seconds for LLM API requests.
            connect_timeout (int): Connection-establishment timeout in seconds.
            timeout_cooldown (float): Grace period after a read timeout.
            max_output_tokens (int | None): Maximum generation length.
            use_think (bool): Whether to enable "think" mode for the LLM API.
            debug (bool): Enable debug mode for verbose output.
        """
        if model_choice not in valid_models:
            raise ValueError(
                f"Invalid model choice: {model_choice}. Valid options are: {valid_models}"
            )

        if context_window > context_window_limits.get(model_choice, 4096):
            raise ValueError(
                f"Context window {context_window} exceeds limit for model {model_choice} "
                f"({context_window_limits.get(model_choice)})."
            )

        can_think = has_think_option.get(model_choice, False)
        think_option = None # model does not support think by default
        if can_think: # if the model supports think, we can set it to True or False based on user input
            think_option = use_think
        elif use_think:
            raise ValueError(
                f"Model {model_choice} does not support the 'think' option."
            )

        model_name = model_choice
        if use_instruct and has_instruct_versions.get(model_choice, False):
            model_name += ":instruct"
            if use_quantized and has_quantized_versions.get(model_choice, False):
                model_name += f"-q{quantization_bits}"

        self.use_instruct = use_instruct
        self.use_quantized = use_quantized
        self.quantization_bits = quantization_bits
        self.timeout = timeout
        self.connect_timeout = connect_timeout
        self.timeout_cooldown = timeout_cooldown
        self.max_output_tokens = max_output_tokens
        self.context_window = context_window
        self.seed = seed
        self.temperature = temperature
        self.think_option = think_option
        self.debug = debug
        self.base_url, self.api_key = load_api_config(config_path)
        self.headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        self.session = requests.Session()

        self.models_resp = self._fetch_models()
        self.model_ids = extract_model_ids(self.models_resp)
        if not self.model_ids:
            raise RuntimeError(f"Couldn't parse model list response: {self.models_resp}")

        if self.debug:
            self._log_available_models()

        self.change_llm(model_name)

        self._closed = False
        self._cleanup_lock = threading.Lock()
        self._register_cleanup()

    def change_llm(
        self, 
        model_name: str
    ) -> None:
        """
        Change the current LLM model.
        Unload the previous model first to avoid GPU memory staying allocated.
        """
        previous = getattr(self, "model_choice", None)
        if previous is not None and previous != model_name:
            try:
                unload_model(self.base_url, self.headers, previous)
            except Exception:
                pass

        self.model_choice = pick_model(self.model_ids, choice=model_name)
        print("\nUsing model:", self.model_choice)

    def _fetch_models(self) -> APIResponse:
        """
        Fetch the list of available models from the API.

        Returns:
            APIResponse: JSON response containing the list of models.
        """
        return list_models(base_url=self.base_url, headers=self.headers, session=self.session)

    def _log_available_models(self) -> None:
        """
        Log the available models in debug mode.
        """
        print("Available models:")
        for index, model_id in enumerate(self.model_ids, start=1):
            print(f"  {index:>2}. {model_id}")

    def chat(
        self, 
        user_text: str
    ) -> Tuple[APIResponse, StatusInfo]:
        """
        Send a chat message to the API and get the response.

        Args:
            user_text (str): The user's input text.

        Returns:
            Tuple[APIResponse, StatusInfo]: JSON response and status metadata from the API.
        """
        return chat(
            base_url=self.base_url,
            headers=self.headers,
            model=self.model_choice,
            user_text=user_text,
            context_window=self.context_window,
            seed=self.seed,
            temperature=self.temperature,
            timeout=self.timeout,
            connect_timeout=self.connect_timeout,
            timeout_cooldown=self.timeout_cooldown,
            max_output_tokens=self.max_output_tokens,
            think_option=self.think_option,
            session=self.session,
        )

    def normalize_usage(
        self, 
        raw: APIResponse
    ) -> StatusInfo:
        """
        Normalize token usage returned by different backends (OpenAI-style, Ollama/OpenWebUI-style, etc.)
        into a stable schema.

        Returns keys:
        - prompt_tokens
        - completion_tokens
        - total_tokens
        - prompt_tps (optional)
        - completion_tps (optional)
        - total_seconds (optional)
        - prompt_seconds (optional)
        - completion_seconds (optional)
        """
        if not isinstance(raw, dict):
            return {}

        # Prefer explicit fields if present
        prompt_tokens = raw.get("prompt_tokens", raw.get("prompt_eval_count"))
        completion_tokens = raw.get("completion_tokens", raw.get("eval_count"))
        total_tokens = raw.get("total_tokens")

        # Fill total if missing
        if total_tokens is None and prompt_tokens is not None and completion_tokens is not None:
            total_tokens = int(prompt_tokens) + int(completion_tokens)

        out = {}
        if prompt_tokens is not None:
            out["prompt_tokens"] = int(prompt_tokens)
        if completion_tokens is not None:
            out["response_tokens"] = int(completion_tokens)
        if total_tokens is not None:
            out["total_tokens"] = int(total_tokens)

        # Throughput
        if "prompt_token/s" in raw and raw["prompt_token/s"] is not None:
            out["prompt_tps"] = float(raw["prompt_token/s"])
        if "response_token/s" in raw and raw["response_token/s"] is not None:
            out["completion_tps"] = float(raw["response_token/s"])

        if "total_duration" in raw and raw["total_duration"] is not None:
            out["total_seconds"] = ns_to_s(raw["total_duration"])
        if "prompt_eval_duration" in raw and raw["prompt_eval_duration"] is not None:
            out["prompt_seconds"] = ns_to_s(raw["prompt_eval_duration"])
        if "eval_duration" in raw and raw["eval_duration"] is not None:
            out["response_seconds"] = ns_to_s(raw["eval_duration"])

        return out

    def _register_cleanup(self) -> None:
        """
        Register process-level cleanup hooks once per client instance.
        """
        # If you want to use the standalone helper from api_utils:
        register_cleanup_handlers(self.base_url, self.headers, self.model_choice)

        # Additionally register atexit that calls the instance method (keeps it idempotent)
        atexit.register(self.close)

        # And catch signals here too (so close() is used, not a raw unload)
        def _handler(signum, frame):
            self.close()
            raise KeyboardInterrupt

        for sig in (signal.SIGINT, signal.SIGTERM):
            try:
                signal.signal(sig, _handler)
            except Exception:
                pass

    def close(self) -> None:
        """
        Explicitly unload the model to free RAM/VRAM.
        Safe to call multiple times.
        """
        with self._cleanup_lock:
            if self._closed:
                return
            self._closed = True
        try:
            # best-effort unload (keeps process alive but frees model memory)
            unload_model(self.base_url, self.headers, self.model_choice)
        except Exception:
            pass
        finally:
            self.session.close()

    def __del__(self) -> None:
        # Destructor is NOT guaranteed to run, but it's a helpful fallback.
        try:
            self.close()
        except Exception:
            pass
