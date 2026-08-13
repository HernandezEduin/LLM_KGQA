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

from typing import Any, Dict, List, Tuple

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

    @staticmethod
    def _format_navigation_history(
        history: List[Tuple[str, str, str]],
        entity_title: dict,
        relation_title: dict,
        include_history: bool = True,
    ) -> str:
        if not include_history:
            return "  (not shown)"

        readable_history = translate_path(history, entity_title, relation_title)
        if not readable_history:
            return "  (none)"
        return "\n".join(
            f"  {index}. ({head}, {relation}, {tail})"
            for index, (head, relation, tail) in enumerate(readable_history)
        )

    @staticmethod
    def _format_readable_path(
        history: List[Tuple[str, str, str]],
        entity_title: dict,
        relation_title: dict,
    ) -> List[Tuple[str, str, str]]:
        return translate_path(history, entity_title, relation_title)

    @staticmethod
    def _strip_optional_json_fence(content: str) -> str:
        text = content.strip()
        fence_match = re.fullmatch(
            r"```(?:json)?\s*(.*?)\s*```",
            text,
            flags=re.IGNORECASE | re.DOTALL,
        )
        if fence_match:
            text = fence_match.group(1).strip()
        return text

    @classmethod
    def _parse_json_object(cls, content: str) -> dict:
        if not isinstance(content, str):
            raise ValueError("Navigation response must be text.")
        text = cls._strip_optional_json_fence(content)
        try:
            decision = json.loads(text)
        except json.JSONDecodeError as exc:
            raise ValueError(f"Navigation response must be exactly one JSON object: {exc}") from exc
        if not isinstance(decision, dict):
            raise ValueError("Navigation response must be a JSON object.")
        return decision

    @staticmethod
    def _require_exact_keys(decision: dict, expected_keys: set[str], schema_name: str) -> None:
        observed_keys = set(decision.keys())
        if observed_keys != expected_keys:
            raise ValueError(
                f"{schema_name} response must contain exactly keys "
                f"{sorted(expected_keys)}; received {sorted(observed_keys)}."
            )

    @staticmethod
    def _require_bool(value: Any, field_name: str) -> bool:
        if type(value) is not bool:
            raise ValueError(f"{field_name} must be a boolean.")
        return value

    @staticmethod
    def _require_index(value: Any, field_name: str, option_count: int) -> int:
        if type(value) is not int:
            raise ValueError(f"{field_name} must be an integer ID.")
        if value < 0 or value >= option_count:
            raise ValueError(
                f"{field_name} ID {value} is outside the valid range 0..{option_count - 1}."
            )
        return value

    def prepare_navigation_prompt(
        self,
        question: str,
        start_node: str,
        current_entity: str,
        history: List[Tuple[str, str, str]],
        actions: List[Tuple[str, str, str]],
        step: int,
        max_steps: int,
        entity_title: dict,
        relation_title: dict,
        include_history: bool = True,
    ) -> Tuple[str, str]:
        """Build one tuple-action graph-navigation prompt from controller-owned state."""
        start_entity_str = entity_title.get(start_node, start_node)
        current_entity_str = entity_title.get(current_entity, current_entity)
        history_str = self._format_navigation_history(
            history,
            entity_title,
            relation_title,
            include_history=include_history,
        )

        action_lines = []
        for action_id, (_, relation, tail) in enumerate(actions):
            relation_str = relation_title.get(relation, relation)
            tail_str = entity_title.get(tail, tail)
            action_lines.append(
                f"  [{action_id}]. --{relation_str} ({relation})--> {tail_str} ({tail})"
            )
        actions_str = "\n".join(action_lines) if action_lines else "  (none)"
        evidence_scope = (
            "question, traversed path, current entity, and available actions"
            if include_history
            else "question, start entity, current entity, and available actions"
        )

        template = (
            "You are navigating a knowledge graph to answer a question.\n\n"

            "At each step, make up to two decisions:\n"
            "1. Select at most one of the available actions to move to its destination entity.\n"
            "2. Decide whether to stop navigating.\n\n"

            "Rules:\n"
            "- If you choose an action, it must be the integer ID of exactly one listed action.\n"
            "- Do not invent actions, relations, or entities.\n"
            "- Set \"stop\" to true when the current entity, or the destination entity of the "
            "selected action, answers the question based on the traversed path.\n"
            "- If \"stop\" is true, the terminal entity will be treated as the final answer: "
            "the destination entity if an action is selected, otherwise the current entity.\n"
            "- You may stop without selecting an action if the current entity already answers the question.\n"
            "- Otherwise, select the action that best continues the reasoning path and set \"stop\" to false.\n"
            f"- Base your decision only on the {evidence_scope}.\n\n"

            "Return exactly one JSON object and nothing else.\n"
            "Move and continue: {\"action\": 0, \"stop\": false}\n"
            "Move and stop: {\"action\": 0, \"stop\": true}\n"
            "Stop at current entity: {\"action\": null, \"stop\": true}\n\n"

            f"Question: {question}\n"
            f"Start entity: {start_entity_str} ({start_node})\n"
            f"Current entity: {current_entity_str} ({current_entity})\n"
            f"Step: {step} / {max_steps}\n\n"

            "Traversed path:\n"
            f"{history_str}\n\n"

            "Available actions:\n"
            f"{actions_str}\n"
        )
        return template, history_str

    @staticmethod
    def _group_actions_by_relation(
        actions: List[Tuple[str, str, str]],
    ) -> List[Tuple[str, List[Tuple[str, str, str]]]]:
        grouped: Dict[str, List[Tuple[str, str, str]]] = {}
        for triplet in actions:
            grouped.setdefault(triplet[1], []).append(triplet)
        return [
            (relation, sorted(grouped[relation], key=lambda triplet: triplet[2]))
            for relation in sorted(grouped)
        ]

    def prepare_relation_navigation_prompt(
        self,
        question: str,
        start_node: str,
        current_entity: str,
        history: List[Tuple[str, str, str]],
        relation_groups: List[Tuple[str, List[Tuple[str, str, str]]]],
        step: int,
        max_steps: int,
        entity_title: dict,
        relation_title: dict,
        include_history: bool = True,
    ) -> Tuple[str, str]:
        """Build the relation-selection stage prompt for factorized navigation."""
        start_entity_str = entity_title.get(start_node, start_node)
        current_entity_str = entity_title.get(current_entity, current_entity)
        history_str = self._format_navigation_history(
            history,
            entity_title,
            relation_title,
            include_history=include_history,
        )
        relation_lines = []
        for relation_id, (relation, relation_actions) in enumerate(relation_groups):
            relation_str = relation_title.get(relation, relation)
            relation_lines.append(
                f"  [{relation_id}]. {relation_str} ({relation}) "
                f"[{len(relation_actions)} destination(s)]"
            )
        relations_str = "\n".join(relation_lines) if relation_lines else "  (none)"
        evidence_scope = (
            "question, traversed path, current entity, and available relations"
            if include_history
            else "question, start entity, current entity, and available relations"
        )

        template = (
            "You are navigating a knowledge graph to answer a question.\n\n"
            "This is the relation-selection stage. Choose one available relation to continue, "
            "or stop at the current entity if it already answers the question.\n\n"
            "Rules:\n"
            "- If you choose a relation, it must be the integer ID of exactly one listed relation.\n"
            "- Do not invent relations or entities.\n"
            "- Set \"stop\" to true only when stopping at the current entity without selecting a relation.\n"
            "- If you select a relation, set \"stop\" to false; the controller will then ask for a destination entity.\n"
            f"- Base your decision only on the {evidence_scope}.\n\n"
            "Return exactly one JSON object and nothing else.\n"
            "Select relation: {\"relation\": 0, \"stop\": false}\n"
            "Stop at current entity: {\"relation\": null, \"stop\": true}\n\n"
            f"Question: {question}\n"
            f"Start entity: {start_entity_str} ({start_node})\n"
            f"Current entity: {current_entity_str} ({current_entity})\n"
            f"Step: {step} / {max_steps}\n\n"
            "Traversed path:\n"
            f"{history_str}\n\n"
            "Available relations:\n"
            f"{relations_str}\n"
        )
        return template, history_str

    def prepare_entity_navigation_prompt(
        self,
        question: str,
        start_node: str,
        current_entity: str,
        history: List[Tuple[str, str, str]],
        selected_relation: str,
        relation_actions: List[Tuple[str, str, str]],
        step: int,
        max_steps: int,
        entity_title: dict,
        relation_title: dict,
        include_history: bool = True,
    ) -> Tuple[str, str]:
        """Build the entity-selection stage prompt for factorized navigation."""
        start_entity_str = entity_title.get(start_node, start_node)
        current_entity_str = entity_title.get(current_entity, current_entity)
        relation_str = relation_title.get(selected_relation, selected_relation)
        history_str = self._format_navigation_history(
            history,
            entity_title,
            relation_title,
            include_history=include_history,
        )
        entity_lines = []
        for entity_id, (_, _, tail) in enumerate(relation_actions):
            tail_str = entity_title.get(tail, tail)
            entity_lines.append(f"  [{entity_id}]. {tail_str} ({tail})")
        entities_str = "\n".join(entity_lines) if entity_lines else "  (none)"
        evidence_scope = (
            "question, traversed path, current entity, selected relation, and available destination entities"
            if include_history
            else "question, start entity, current entity, selected relation, and available destination entities"
        )

        template = (
            "You are navigating a knowledge graph to answer a question.\n\n"
            "This is the destination-entity selection stage for the already selected relation.\n\n"
            "Rules:\n"
            "- Choose exactly one listed destination entity.\n"
            "- Do not invent entities.\n"
            "- Set \"stop\" to true when the selected destination entity answers the question.\n"
            "- If \"stop\" is true, the selected destination entity will be treated as the final answer.\n"
            "- Otherwise, choose the destination that best continues the reasoning path and set \"stop\" to false.\n"
            f"- Base your decision only on the {evidence_scope}.\n\n"
            "Return exactly one JSON object and nothing else.\n"
            "Move and continue: {\"entity\": 0, \"stop\": false}\n"
            "Move and stop: {\"entity\": 0, \"stop\": true}\n\n"
            f"Question: {question}\n"
            f"Start entity: {start_entity_str} ({start_node})\n"
            f"Current entity: {current_entity_str} ({current_entity})\n"
            f"Selected relation: {relation_str} ({selected_relation})\n"
            f"Step: {step} / {max_steps}\n\n"
            "Traversed path:\n"
            f"{history_str}\n\n"
            "Available destination entities:\n"
            f"{entities_str}\n"
        )
        return template, history_str

    @classmethod
    def parse_navigation_decision(cls, content: str, num_actions: int) -> dict:
        """Parse and strictly validate a tuple-action navigation response."""
        decision = cls._parse_json_object(content)
        cls._require_exact_keys(decision, {"action", "stop"}, "Tuple navigation")
        stop = cls._require_bool(decision["stop"], "stop")
        action = decision["action"]
        if action is None:
            if not stop:
                raise ValueError('action=null is only valid when stop=true.')
            return {"action": None, "stop": stop}
        action_id = cls._require_index(action, "action", num_actions)
        return {"action": action_id, "stop": stop}

    @classmethod
    def parse_relation_decision(cls, content: str, num_relations: int) -> dict:
        """Parse and strictly validate a factorized relation-stage response."""
        decision = cls._parse_json_object(content)
        cls._require_exact_keys(decision, {"relation", "stop"}, "Relation navigation")
        stop = cls._require_bool(decision["stop"], "stop")
        relation = decision["relation"]
        if relation is None:
            if not stop:
                raise ValueError('relation=null is only valid when stop=true.')
            return {"relation": None, "stop": stop}
        relation_id = cls._require_index(relation, "relation", num_relations)
        if stop:
            raise ValueError('relation-stage stop=true is only valid with relation=null.')
        return {"relation": relation_id, "stop": stop}

    @classmethod
    def parse_entity_decision(cls, content: str, num_entities: int) -> dict:
        """Parse and strictly validate a factorized entity-stage response."""
        decision = cls._parse_json_object(content)
        cls._require_exact_keys(decision, {"entity", "stop"}, "Entity navigation")
        stop = cls._require_bool(decision["stop"], "stop")
        entity_id = cls._require_index(decision["entity"], "entity", num_entities)
        return {"entity": entity_id, "stop": stop}

    def _accumulate_navigation_usage(self, aggregate_status: dict, status_info: dict) -> None:
        for field in (
            "prompt_tokens",
            "response_tokens",
            "completion_tokens",
            "total_tokens",
            "prompt_seconds",
            "response_seconds",
            "total_seconds",
            "prompt_tps",
            "completion_tps",
        ):
            if field in status_info:
                aggregate_status[field] = aggregate_status.get(field, 0) + status_info[field]

    @staticmethod
    def estimate_prompt_tokens(prompt: str) -> int:
        """Conservative tokenizer-free context estimate for safety checks."""
        return max(len(prompt.split()), (len(prompt) + 3) // 4)

    def prompt_fits_context(self, prompt: str) -> Tuple[bool, int, int | None]:
        estimated_tokens = self.estimate_prompt_tokens(prompt)
        context_window = getattr(self, "context_window", None)
        if context_window is None:
            return True, estimated_tokens, None
        return estimated_tokens <= int(context_window), estimated_tokens, int(context_window)

    def _call_navigation_stage(
        self,
        prompt: str,
        stage: str,
        strategy: str,
        step: int,
        current_entity: str,
        aggregate_status: dict,
        trace=None,
    ) -> Tuple[str | None, dict]:
        if trace is not None:
            trace(
                f"\n=== Navigation step {step} ({strategy}/{stage}) ===\n"
                f"MODEL INPUT\n{prompt}"
            )

        out, status_info = self.chat(user_text=prompt)
        aggregate_status["actual_llm_calls"] += 1
        status_info.update(self.normalize_usage(out))
        self._accumulate_navigation_usage(aggregate_status, status_info)

        call_record = {
            "step": step,
            "stage": stage,
            "strategy": strategy,
            "status": status_info.get("status"),
            "elapsed_time": status_info.get("elapsed_time", 0.0),
            "prompt_tokens": status_info.get("prompt_tokens"),
            "completion_tokens": status_info.get("completion_tokens", status_info.get("response_tokens")),
            "total_tokens": status_info.get("total_tokens"),
        }
        aggregate_status["model_calls"].append(call_record)
        aggregate_status["elapsed_time"] += status_info.get("elapsed_time", 0.0)

        if status_info.get("status") != "success":
            call_record["message"] = status_info.get("message", "Navigation request failed")
            return None, status_info

        try:
            content = out["message"]["content"]
        except (KeyError, TypeError) as exc:
            status_info.update({
                "status": "error",
                "message": f"Navigation response missing message content: {exc}",
            })
            call_record["status"] = "error"
            call_record["message"] = status_info["message"]
            return None, status_info

        call_record["raw_output"] = content
        aggregate_status["raw_model_outputs"].append({
            "step": step,
            "stage": stage,
            "strategy": strategy,
            "content": content,
        })
        if trace is not None:
            trace(f"MODEL OUTPUT ({strategy}/{stage})\n{content}")
        return content, status_info

    def process_navigation_question(
        self,
        question: str,
        start_node: str,
        outgoing_index: dict,
        entity_title: dict,
        relation_title: dict,
        max_steps: int = 4,
        max_actions: int | None = None,
        navigation_approach: str = "tuple",
        memory_approach: str = "full",
        prompting_approach: str = "zero-shot",
        hybrid_threshold: int = 50,
        trace=None,
    ) -> Tuple[str, str, dict]:
        """Navigate from ``start_node`` and use the terminal KG entity as the answer."""
        if max_steps < 0:
            raise ValueError("max_steps must be non-negative.")
        if max_actions is not None and max_actions < 1:
            raise ValueError("max_actions must be positive when provided.")
        if navigation_approach not in {"tuple", "factorized", "hybrid"}:
            raise ValueError(f"Unsupported navigation approach: {navigation_approach}")
        if memory_approach not in {"none", "full"}:
            raise ValueError(f"Unsupported memory approach: {memory_approach}")
        if prompting_approach != "zero-shot":
            raise NotImplementedError(
                f"Prompting approach '{prompting_approach}' is not implemented for navigation. "
                "Use --prompting-approach zero-shot."
            )
        if hybrid_threshold < 0:
            raise ValueError("hybrid_threshold must be non-negative.")

        current_entity = start_node
        history: List[Tuple[str, str, str]] = []
        include_history = memory_approach == "full"
        aggregate_status = {
            "status": "success",
            "message": "",
            "termination_reason": None,
            "elapsed_time": 0.0,
            "predicted_path": history,
            "readable_predicted_path": [],
            "final_entity": current_entity,
            "last_current_entity": current_entity,
            "navigation_steps": 0,
            "executed_graph_edges": 0,
            "neighborhood_sizes": [],
            "selected_actions": [],
            "decision_records": [],
            "strategy_by_step": [],
            "navigation_approach": navigation_approach,
            "memory_approach": memory_approach,
            "prompting_approach": prompting_approach,
            "hybrid_threshold": hybrid_threshold,
            "max_actions": max_actions,
            "logical_decisions": [],
            "logical_decision_count": 0,
            "actual_llm_calls": 0,
            "api_retries": 0,
            "model_calls": [],
            "raw_model_outputs": [],
            "parse_validation_errors": [],
            "max_actions_exceeded": False,
            "context_window_exceeded": False,
            "graph_directionality": "outgoing",
        }

        def final_history_text() -> str:
            return self._format_navigation_history(
                history,
                entity_title,
                relation_title,
                include_history=True,
            )

        def finalize(
            status: str,
            termination_reason: str,
            final_entity: str | None,
            message: str,
        ) -> Tuple[str, str, dict]:
            aggregate_status.update({
                "status": status,
                "termination_reason": termination_reason,
                "message": message,
                "predicted_path": list(history),
                "readable_predicted_path": self._format_readable_path(
                    history,
                    entity_title,
                    relation_title,
                ),
                "final_entity": final_entity,
                "last_current_entity": current_entity,
                "navigation_steps": len(history),
                "executed_graph_edges": len(history),
                "logical_decisions": list(aggregate_status["decision_records"]),
                "logical_decision_count": len(aggregate_status["decision_records"]),
            })
            if final_entity is not None:
                aggregate_status["predicted_answer"] = final_entity
            prediction = final_entity if status == "success" and final_entity is not None else (
                "TIMEOUT" if status == "timeout" else "ERROR"
            )
            if trace is not None:
                trace(
                    f"NAVIGATION TERMINATED ({termination_reason})\n"
                    f"Final entity: {final_entity}\n"
                    f"Message: {message}"
                )
            return prediction, final_history_text(), aggregate_status

        def fail_stage(stage_status: dict, termination_reason: str) -> Tuple[str, str, dict]:
            status = stage_status.get("status", "error")
            if status not in {"timeout", "success"}:
                status = "error"
            return finalize(
                status=status,
                termination_reason=termination_reason,
                final_entity=None,
                message=stage_status.get("message", "Navigation request failed"),
            )

        def fail_parse(step: int, stage: str, strategy: str, content: str, exc: Exception):
            error_record = {
                "step": step,
                "stage": stage,
                "strategy": strategy,
                "raw_output": content,
                "error": str(exc),
            }
            aggregate_status["parse_validation_errors"].append(error_record)
            return finalize(
                status="error",
                termination_reason="invalid_output",
                final_entity=None,
                message=f"Invalid {strategy}/{stage} navigation response: {exc}",
            )

        def fail_max_options(step: int, strategy: str, option_kind: str, option_count: int):
            aggregate_status["max_actions_exceeded"] = True
            aggregate_status["max_actions_kind"] = option_kind
            aggregate_status["max_actions_count"] = option_count
            return finalize(
                status="error",
                termination_reason="max_actions_exceeded",
                final_entity=None,
                message=(
                    f"{strategy} {option_kind} count {option_count} exceeds "
                    f"configured --max-actions={max_actions}; no legal actions were discarded."
                ),
            )

        def fail_context_window(step: int, stage: str, strategy: str, prompt: str):
            fits, estimated_tokens, context_window = self.prompt_fits_context(prompt)
            if fits:
                return None
            aggregate_status["context_window_exceeded"] = True
            aggregate_status["context_window_stage"] = stage
            aggregate_status["context_window_strategy"] = strategy
            aggregate_status["estimated_prompt_tokens"] = estimated_tokens
            aggregate_status["context_window"] = context_window
            return finalize(
                status="error",
                termination_reason="context_window_exceeded",
                final_entity=None,
                message=(
                    f"Estimated {stage} prompt size ({estimated_tokens} tokens) exceeds "
                    f"configured context window ({context_window}); no legal actions were discarded."
                ),
            )

        def record_move(
            step: int,
            strategy: str,
            current_before: str,
            actions: List[Tuple[str, str, str]],
            selected_triplet: Tuple[str, str, str],
            selected_action: int,
            stop: bool,
            extra: dict | None = None,
        ) -> None:
            readable_move = translate_path([selected_triplet], entity_title, relation_title)[0]
            record = {
                "step": step,
                "strategy": strategy,
                "current_entity": current_before,
                "neighborhood_size": len(actions),
                "selected_action": selected_action,
                "selected_relation": selected_triplet[1],
                "selected_destination": selected_triplet[2],
                "selected_triplet": selected_triplet,
                "readable_selected_triplet": readable_move,
                "stop": stop,
            }
            if extra:
                record.update(extra)
            aggregate_status["selected_actions"].append(record)
            aggregate_status["decision_records"].append(record)

        for step_index in range(max_steps):
            step = step_index + 1
            actions = sorted(
                outgoing_index.get(current_entity, []),
                key=lambda triplet: (triplet[1], triplet[2]),
            )
            neighborhood_size = len(actions)
            if not actions:
                aggregate_status["neighborhood_sizes"].append(0)
                aggregate_status["strategy_by_step"].append("none")
                aggregate_status["decision_records"].append({
                    "step": step,
                    "strategy": "none",
                    "current_entity": current_entity,
                    "neighborhood_size": 0,
                    "selected_action": None,
                    "stop": True,
                    "termination_reason": "no_actions",
                })
                return finalize(
                    status="success",
                    termination_reason="no_actions",
                    final_entity=current_entity,
                    message="No outgoing legal actions; terminal current entity used as prediction.",
                )

            strategy = navigation_approach
            if navigation_approach == "hybrid":
                strategy = "tuple" if neighborhood_size <= hybrid_threshold else "factorized"

            aggregate_status["neighborhood_sizes"].append(neighborhood_size)
            aggregate_status["strategy_by_step"].append(strategy)
            current_before = current_entity

            if strategy == "tuple":
                if max_actions is not None and neighborhood_size > max_actions:
                    return fail_max_options(step, strategy, "tuple_action", neighborhood_size)

                prompt, _ = self.prepare_navigation_prompt(
                    question=question,
                    start_node=start_node,
                    current_entity=current_entity,
                    history=history,
                    actions=actions,
                    step=step,
                    max_steps=max_steps,
                    entity_title=entity_title,
                    relation_title=relation_title,
                    include_history=include_history,
                )
                context_failure = fail_context_window(step, "tuple", strategy, prompt)
                if context_failure is not None:
                    return context_failure
                content, status_info = self._call_navigation_stage(
                    prompt,
                    stage="tuple",
                    strategy=strategy,
                    step=step,
                    current_entity=current_entity,
                    aggregate_status=aggregate_status,
                    trace=trace,
                )
                if status_info.get("status") != "success" or content is None:
                    return fail_stage(status_info, "api_error")
                try:
                    decision = self.parse_navigation_decision(content, len(actions))
                except ValueError as exc:
                    return fail_parse(step, "tuple", strategy, content, exc)

                action_id = decision["action"]
                if action_id is None:
                    aggregate_status["decision_records"].append({
                        "step": step,
                        "strategy": strategy,
                        "current_entity": current_before,
                        "neighborhood_size": neighborhood_size,
                        "selected_action": None,
                        "selected_relation": None,
                        "selected_destination": current_before,
                        "stop": True,
                        "raw_output": content,
                    })
                    return finalize(
                        status="success",
                        termination_reason="llm_stop",
                        final_entity=current_entity,
                        message="LLM stopped at current entity.",
                    )

                selected_triplet = actions[action_id]
                history.append(selected_triplet)
                current_entity = selected_triplet[2]
                aggregate_status["final_entity"] = current_entity
                record_move(
                    step=step,
                    strategy=strategy,
                    current_before=current_before,
                    actions=actions,
                    selected_triplet=selected_triplet,
                    selected_action=action_id,
                    stop=decision["stop"],
                    extra={"raw_output": content},
                )
                if trace is not None:
                    readable_move = translate_path([selected_triplet], entity_title, relation_title)[0]
                    trace(
                        f"VALIDATED MOVE [{action_id}]\n"
                        f"  ({readable_move[0]}, {readable_move[1]}, {readable_move[2]})\n"
                        f"New current entity: "
                        f"{entity_title.get(current_entity, current_entity)} ({current_entity})"
                    )
                if decision["stop"]:
                    return finalize(
                        status="success",
                        termination_reason="llm_stop",
                        final_entity=current_entity,
                        message="LLM selected an action and stopped at its destination.",
                    )
                continue

            relation_groups = self._group_actions_by_relation(actions)
            if max_actions is not None and len(relation_groups) > max_actions:
                return fail_max_options(step, strategy, "relation", len(relation_groups))

            relation_prompt, _ = self.prepare_relation_navigation_prompt(
                question=question,
                start_node=start_node,
                current_entity=current_entity,
                history=history,
                relation_groups=relation_groups,
                step=step,
                max_steps=max_steps,
                entity_title=entity_title,
                relation_title=relation_title,
                include_history=include_history,
            )
            context_failure = fail_context_window(step, "relation", strategy, relation_prompt)
            if context_failure is not None:
                return context_failure
            relation_content, relation_status = self._call_navigation_stage(
                relation_prompt,
                stage="relation",
                strategy=strategy,
                step=step,
                current_entity=current_entity,
                aggregate_status=aggregate_status,
                trace=trace,
            )
            if relation_status.get("status") != "success" or relation_content is None:
                return fail_stage(relation_status, "api_error")
            try:
                relation_decision = self.parse_relation_decision(
                    relation_content,
                    len(relation_groups),
                )
            except ValueError as exc:
                return fail_parse(step, "relation", strategy, relation_content, exc)

            relation_id = relation_decision["relation"]
            if relation_id is None:
                aggregate_status["decision_records"].append({
                    "step": step,
                    "strategy": strategy,
                    "current_entity": current_before,
                    "neighborhood_size": neighborhood_size,
                    "selected_action": None,
                    "selected_relation": None,
                    "selected_destination": current_before,
                    "stop": True,
                    "raw_relation_output": relation_content,
                })
                return finalize(
                    status="success",
                    termination_reason="llm_stop",
                    final_entity=current_entity,
                    message="LLM stopped at current entity during relation selection.",
                )

            selected_relation, relation_actions = relation_groups[relation_id]
            if max_actions is not None and len(relation_actions) > max_actions:
                return fail_max_options(step, strategy, "destination_entity", len(relation_actions))

            entity_prompt, _ = self.prepare_entity_navigation_prompt(
                question=question,
                start_node=start_node,
                current_entity=current_entity,
                history=history,
                selected_relation=selected_relation,
                relation_actions=relation_actions,
                step=step,
                max_steps=max_steps,
                entity_title=entity_title,
                relation_title=relation_title,
                include_history=include_history,
            )
            context_failure = fail_context_window(step, "entity", strategy, entity_prompt)
            if context_failure is not None:
                return context_failure
            entity_content, entity_status = self._call_navigation_stage(
                entity_prompt,
                stage="entity",
                strategy=strategy,
                step=step,
                current_entity=current_entity,
                aggregate_status=aggregate_status,
                trace=trace,
            )
            if entity_status.get("status") != "success" or entity_content is None:
                return fail_stage(entity_status, "api_error")
            try:
                entity_decision = self.parse_entity_decision(entity_content, len(relation_actions))
            except ValueError as exc:
                return fail_parse(step, "entity", strategy, entity_content, exc)

            selected_triplet = relation_actions[entity_decision["entity"]]
            selected_action = actions.index(selected_triplet)
            history.append(selected_triplet)
            current_entity = selected_triplet[2]
            aggregate_status["final_entity"] = current_entity
            record_move(
                step=step,
                strategy=strategy,
                current_before=current_before,
                actions=actions,
                selected_triplet=selected_triplet,
                selected_action=selected_action,
                stop=entity_decision["stop"],
                extra={
                    "relation_choice": relation_id,
                    "entity_choice": entity_decision["entity"],
                    "raw_relation_output": relation_content,
                    "raw_entity_output": entity_content,
                },
            )
            if trace is not None:
                readable_move = translate_path([selected_triplet], entity_title, relation_title)[0]
                trace(
                    f"VALIDATED FACTORIZED MOVE [{selected_action}]\n"
                    f"  relation [{relation_id}] -> entity [{entity_decision['entity']}]\n"
                    f"  ({readable_move[0]}, {readable_move[1]}, {readable_move[2]})\n"
                    f"New current entity: "
                    f"{entity_title.get(current_entity, current_entity)} ({current_entity})"
                )
            if entity_decision["stop"]:
                return finalize(
                    status="success",
                    termination_reason="llm_stop",
                    final_entity=current_entity,
                    message="LLM selected a factorized action and stopped at its destination.",
                )

        return finalize(
            status="success",
            termination_reason="max_steps",
            final_entity=current_entity,
            message=(
                f"Maximum navigation steps ({max_steps}) reached; terminal current entity "
                "used as prediction."
            ),
        )

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