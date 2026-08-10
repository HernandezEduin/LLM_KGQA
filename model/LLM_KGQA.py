import atexit
import json
import re
import signal

from pathlib import Path
import random
import threading

from model.constants import valid_models, has_instruct_versions, has_quantized_versions, context_window_limits
from utils.kgqa_utils import translate_path
from utils.api_utils import list_models, chat, extract_model_ids, pick_model, load_api_config, register_cleanup_handlers, unload_model

from typing import List, Tuple

# Durations: often in nanoseconds for Ollama-style stats
def ns_to_s(x):
    try:
        return float(x) / 1e9
    except Exception:
        return None

class LLM_KGQA_Client:
    def __init__(
        self, 
        config_path: Path, 
        model_choice: str = 'gemma3',
        use_instruct: bool = False,
        use_quantized: bool = False,
        quantization_bits: int = 4,
        context_window: int = 4096,
        seed: int | None = None, 
        temperature: float | None = None,
        timeout: int = 120, 
        debug: bool = False
    ):
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
            debug (bool): Enable debug mode for verbose output.
        """
        if model_choice not in valid_models:
            raise ValueError(f"Invalid model choice: {model_choice}. Valid options are: {valid_models}")
        
        if context_window > context_window_limits.get(model_choice, 4096):
            raise ValueError(
                f"Context window {context_window} exceeds limit for model {model_choice} "
                f"({context_window_limits.get(model_choice)})."
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
        self.context_window = context_window
        self.seed = seed
        self.temperature = temperature
        self.debug = debug
        self.base_url, self.api_key = load_api_config(config_path)
        self.headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

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

    def change_llm(self, model_name: str):
        """
        Change the current LLM model.
        Unload the previous model first to avoid GPU memory staying allocated.
        """
        prev = getattr(self, "model_choice", None)
        if prev is not None and prev != model_name:
            # unload previous model best-effort
            try:
                unload_model(self.base_url, self.headers, prev)
            except Exception:
                pass

        self.model_choice = pick_model(self.model_ids, choice=model_name)
        print("\nUsing model:", self.model_choice)

    def prepare_prompt(
            self, 
            question: str,
            start_node: str, 
            triplets: List[Tuple[str, str, str]], 
            entity_title: dict,
            relation_title: dict
        ) -> Tuple[str, str]:
        """
        Prepare the prompt for the LLM based on the question and triplets.

        Args:
            question (str): The natural-language question.
            start_node (str): The starting node for the subgraph.
            triplets (List[Tuple[str, str, str]]): Knowledge-graph triplets.
            entity_title (dict): Mapping of entity IDs to titles.
            relation_title (dict): Mapping of relation IDs to titles.

        Returns:
            str: The formatted prompt string.
        """
        start_node_str = entity_title.get(start_node, start_node)
        triplets_str = translate_path(triplets, entity_title, relation_title)
        triplets_str = "{\n" + "\n".join([f"\t({h}, {r}, {t})" for h, r, t in triplets_str]) + "\n}"
        template = (
            "You will be given a natural-language question, a starting node, and a set of knowledge-graph triplets.\n"
            "Answer the question using ONLY the information supported by the provided triplets.\n"
            # "If the answer is not entailed by the triplets, reply exactly: UNKNOWN.\n\n"
            "Each question contains a unique answer.\n"
            "Return only the final answer (no explanation, no reasoning, no extra text).\n"
            "Double-check the spelling of your answer.\n\n"
            f"Question: {question}\n"
            f"Starting Node: {start_node_str}\n"
            "Triplets (head, relation, tail):\n"
            f"{triplets_str}\n\n"
        )
        return template, triplets_str

    def prepare_navigation_prompt(
        self,
        question: str,
        current_entity: str,
        history: List[Tuple[str, str, str]],
        actions: List[Tuple[str, str, str]],
        entity_title: dict,
        relation_title: dict,
    ) -> Tuple[str, str]:
        """Build one graph-navigation prompt from controller-owned state."""
        current_entity_str = entity_title.get(current_entity, current_entity)
        readable_history = translate_path(history, entity_title, relation_title)
        if readable_history:
            history_str = "\n".join(
                f"  {index}. ({head}, {relation}, {tail})"
                for index, (head, relation, tail) in enumerate(readable_history)
            )
        else:
            history_str = "  (none)"

        action_lines = []
        for action_id, (_, relation, tail) in enumerate(actions):
            relation_str = relation_title.get(relation, relation)
            tail_str = entity_title.get(tail, tail)
            action_lines.append(
                f"  {action_id}. --{relation_str} ({relation})--> {tail_str} ({tail})"
            )
        action_lines.append("  STOP. Stop navigating and provide the final answer")
        actions_str = "\n".join(action_lines)

        template = (
            "Navigate the knowledge graph to answer the question.\n"
            "Choose exactly one of the available actions. Do not invent an action.\n"
            "An integer action moves to the destination entity for that action.\n"
            "Choose STOP only when the traversed path supports a final answer, or when no useful action remains.\n"
            "Return JSON only, with no markdown or explanation.\n"
            "To move: {\"action\": 0}\n"
            "To stop: {\"action\": \"STOP\", \"answer\": \"final answer\"}\n\n"
            f"Question: {question}\n"
            f"Current entity: {current_entity_str} ({current_entity})\n"
            "History of actions taken:\n"
            f"{history_str}\n"
            "Available actions:\n"
            f"{actions_str}\n"
        )
        return template, history_str

    @staticmethod
    def parse_navigation_decision(content: str) -> dict:
        """Parse and minimally validate a model navigation response."""
        if not isinstance(content, str):
            raise ValueError("Navigation response must be text.")

        text = re.sub(r"^```(?:json)?\s*|\s*```$", "", content.strip(), flags=re.IGNORECASE)
        start = text.find("{")
        end = text.rfind("}")
        if start < 0 or end < start:
            raise ValueError("Navigation response does not contain a JSON object.")

        decision = json.loads(text[start:end + 1])
        if not isinstance(decision, dict) or "action" not in decision:
            raise ValueError("Navigation response must contain an action field.")
        return decision

    def process_navigation_question(
        self,
        question: str,
        start_node: str,
        outgoing_index: dict,
        entity_title: dict,
        relation_title: dict,
        max_steps: int = 4,
        trace=None,
    ) -> Tuple[str, str, dict]:
        """Navigate from ``start_node`` until the model selects STOP."""
        current_entity = start_node
        history = []
        aggregate_status = {
            "status": "success",
            "elapsed_time": 0.0,
            "predicted_path": history,
            "final_entity": current_entity,
        }
        additive_usage_fields = (
            "prompt_tokens", "response_tokens", "total_tokens",
            "prompt_seconds", "response_seconds", "total_seconds",
        )

        for _ in range(max_steps + 1):
            actions = (
                outgoing_index.get(current_entity, [])
                if len(history) < max_steps
                else []
            )
            prompt, history_str = self.prepare_navigation_prompt(
                question, current_entity, history, actions, entity_title, relation_title
            )
            if trace is not None:
                trace(
                    f"\n=== Navigation step {len(history)} ===\n"
                    f"MODEL INPUT\n{prompt}"
                )

            out, status_info = self.chat(user_text=prompt)
            status_info.update(self.normalize_usage(out))

            aggregate_status["elapsed_time"] += status_info.get("elapsed_time", 0.0)
            for field in additive_usage_fields:
                if field in status_info:
                    aggregate_status[field] = aggregate_status.get(field, 0) + status_info[field]

            if status_info.get("status") != "success":
                aggregate_status.update({
                    "status": status_info.get("status", "error"),
                    "message": status_info.get("message", "Navigation request failed"),
                    "navigation_steps": len(history),
                })
                prediction = "TIMEOUT" if status_info.get("status") == "timeout" else "ERROR"
                if trace is not None:
                    trace(f"NAVIGATION ERROR\n{aggregate_status['message']}")
                return prediction, history_str, aggregate_status

            try:
                content = out["message"]["content"]
                if trace is not None:
                    trace(f"MODEL OUTPUT\n{content}")
                decision = self.parse_navigation_decision(content)
            except (KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
                aggregate_status.update({
                    "status": "error",
                    "message": f"Invalid navigation response: {exc}",
                    "navigation_steps": len(history),
                })
                if trace is not None:
                    trace(f"NAVIGATION ERROR\n{aggregate_status['message']}")
                return "ERROR", history_str, aggregate_status

            selected_action = decision["action"]
            if isinstance(selected_action, str) and selected_action.upper() == "STOP":
                answer = decision.get("answer")
                aggregate_status["navigation_steps"] = len(history)
                if not isinstance(answer, str) or not answer.strip():
                    aggregate_status["message"] = "STOP response did not include an answer."
                    if trace is not None:
                        trace(f"STOP REJECTED\n{aggregate_status['message']}")
                    return "UNKNOWN", history_str, aggregate_status
                if trace is not None:
                    trace(f"STOP SELECTED\nFinal answer: {answer.strip()}")
                return answer.strip(), history_str, aggregate_status

            if isinstance(selected_action, bool):
                selected_action = -1
            try:
                action_id = int(selected_action)
            except (TypeError, ValueError):
                action_id = -1

            if action_id < 0 or action_id >= len(actions):
                aggregate_status.update({
                    "status": "error",
                    "message": f"Invalid action ID: {selected_action}",
                    "navigation_steps": len(history),
                })
                if trace is not None:
                    trace(f"NAVIGATION ERROR\n{aggregate_status['message']}")
                return "ERROR", history_str, aggregate_status

            selected_triplet = actions[action_id]
            history.append(selected_triplet)
            current_entity = selected_triplet[2]
            aggregate_status["final_entity"] = current_entity
            if trace is not None:
                readable_move = translate_path(
                    [selected_triplet], entity_title, relation_title
                )[0]
                trace(
                    f"VALIDATED MOVE [{action_id}]\n"
                    f"  ({readable_move[0]}, {readable_move[1]}, {readable_move[2]})\n"
                    f"New current entity: "
                    f"{entity_title.get(current_entity, current_entity)} ({current_entity})"
                )

        readable_history = translate_path(history, entity_title, relation_title)
        history_str = "\n".join(
            f"  {index}. ({head}, {relation}, {tail})"
            for index, (head, relation, tail) in enumerate(readable_history)
        )
        aggregate_status.update({
            "message": f"Maximum navigation steps ({max_steps}) reached without STOP.",
            "navigation_steps": len(history),
        })
        return "UNKNOWN", history_str, aggregate_status

    def _fetch_models(self):
        """
        Fetch the list of available models from the API.

        Returns:
            dict: JSON response containing the list of models.
        """
        return list_models(base_url=self.base_url, headers=self.headers)

    def _log_available_models(self):
        """
        Log the available models in debug mode.
        """
        print("Available models:")
        for i, model_id in enumerate(self.model_ids, start=1):
            print(f"  {i:>2}. {model_id}")

    def chat(self, user_text: str):
        """
        Send a chat message to the API and get the response.

        Args:
            user_text (str): The user's input text.

        Returns:
            dict: JSON response from the API.
        """
        return chat(
            base_url=self.base_url, 
            headers=self.headers, 
            model=self.model_choice, 
            user_text=user_text,
            context_window=self.context_window, 
            seed=self.seed, 
            temperature=self.temperature,
            timeout=self.timeout
        )

    def process_question(
        self, 
        question: str,
        start_node: str,
        sub_graph: set, 
        entity_title: dict,
        relation_title: dict, 
        random_seed: int = 42, 
        sort_graph: bool = True
    ) -> str:
        """
        Process a single question by preparing the prompt, sending it to the API, and extracting the prediction.

        Args:
            question (str): The natural-language question.
            start_node (str): The starting node for the subgraph.
            sub_graph (set): The subgraph of triplets to use for the question.
            entity_title (dict): Mapping of entity IDs to titles.
            relation_title (dict): Mapping of relation IDs to titles.
            random_seed (int): Seed for random operations to ensure reproducibility.
            sort_graph (bool): Whether to randomly shuffle the subgraph triplets.

        Returns:
            str: The predicted answer from the LLM.
        """
        # randomly shuffle the subgraph triplets to avoid any ordering bias
        sub_graph = list(sub_graph)
        if sort_graph:
            random.Random(random_seed).shuffle(sub_graph)
        template, triplets_str = self.prepare_prompt(question, start_node, sub_graph, entity_title, relation_title)
        out, status_info = self.chat(user_text=template)
        status_info.update( self.normalize_usage(out))

        if self.debug and status_info["status"] != "success":
            print(f"LLM response status: {status_info['status']}, message: {status_info.get('message', '')}")

        if status_info["status"] == "timeout":
            return "TIMEOUT", triplets_str, status_info
        elif status_info["status"] != "success":
            return "ERROR", triplets_str, status_info

        if out is None:
            return "UNKNOWN", triplets_str, status_info

        if type(out) != dict or "message" not in out or "content" not in out["message"]:
            return "UNKNOWN", triplets_str, status_info
        return out["message"]["content"], triplets_str, status_info

    def normalize_usage(self, raw: dict) -> dict:
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

    def __del__(self):
        # Destructor is NOT guaranteed to run, but it's a helpful fallback.
        try:
            self.close()
        except Exception:
            pass