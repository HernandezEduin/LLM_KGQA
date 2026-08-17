import json
import re
from typing import Any, Dict, List, Tuple

from model.llm_kgqa_base import BaseLLMKGQAClient
from utils.kgqa_utils import translate_path


class NavigationLLMKGQAClient(BaseLLMKGQAClient):
    """LLM client for iterative knowledge-graph navigation KGQA experiments."""

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
        for action_id, (head, relation, tail) in enumerate(actions):
            head_str = entity_title.get(head, head)
            relation_str = relation_title.get(relation, relation)
            tail_str = entity_title.get(tail, tail)
            action_lines.append(
                f"  [{action_id}]. ({head_str} ({head}), {relation_str} ({relation}), {tail_str} ({tail}))"
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
            "- Resolve every relationship phrase in the question from the start entity outward; do not skip a "
            "modifier just because a nearby entity seems plausible.\n"
            "- Do not stop at an intermediate entity that answers only part of the question.\n"
            "- If the current entity or selected destination fully answers the question, stop there immediately; "
            "do not move onward to occupation, instance of, subclass of, family name, description source, "
            "category, or other metadata unless the question asks for that.\n"
            "- Prefer actions whose relation label resolves the next unmet phrase in the question; avoid generic "
            "metadata actions unless they are directly requested.\n"
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
            "- Set \"stop\" to true only when the current entity fully answers the question.\n"
            "- Do not stop at an intermediate entity that answers only part of the question.\n"
            "- Resolve every relationship phrase in the question from the start entity outward; do not skip a "
            "modifier just because a nearby entity seems plausible.\n"
            "- If you select a relation, set \"stop\" to false; the controller will then ask for a destination entity.\n"
            "- Prefer a relation that resolves the next unmet phrase in the question; avoid generic metadata "
            "relations unless they are directly requested.\n"
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
        for entity_id, (head, relation, tail) in enumerate(relation_actions):
            head_str = entity_title.get(head, head)
            relation_str = relation_title.get(relation, relation)
            tail_str = entity_title.get(tail, tail)
            entity_lines.append(
                f"  [{entity_id}]. ({head_str} ({head}), {relation_str} ({relation}), {tail_str} ({tail}))"
            )
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
            "- Set \"stop\" to true only when the selected destination entity fully answers the question.\n"
            "- If \"stop\" is true, the selected destination entity will be treated as the final answer.\n"
            "- Otherwise, choose the destination that best continues the reasoning path and set \"stop\" to false.\n"
            "- Do not stop at an intermediate entity that answers only part of the question.\n"
            "- Resolve every relationship phrase in the question from the start entity outward; do not skip a "
            "modifier just because a nearby entity seems plausible.\n"
            "- If the selected destination fully answers the question, stop there immediately; do not move onward "
            "to occupation, instance of, subclass of, family name, description source, category, or other metadata "
            "unless the question asks for that.\n"
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
        max_parse_retries: int = 1,
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
        if max_parse_retries < 0:
            raise ValueError("max_parse_retries must be non-negative.")

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
            "max_parse_retries": max_parse_retries,
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

        def record_parse_error(
            step: int,
            stage: str,
            strategy: str,
            content: str,
            exc: Exception,
            attempt: int,
        ) -> dict:
            error_record = {
                "step": step,
                "stage": stage,
                "strategy": strategy,
                "attempt": attempt,
                "raw_output": content,
                "error": str(exc),
            }
            aggregate_status["parse_validation_errors"].append(error_record)
            return error_record

        def fail_parse(step: int, stage: str, strategy: str, content: str, exc: Exception):
            return finalize(
                status="error",
                termination_reason="invalid_output",
                final_entity=None,
                message=f"Invalid {strategy}/{stage} navigation response after retries: {exc}",
            )

        def make_correction_prompt(prompt: str, stage: str, raw_output: str, exc: Exception) -> str:
            return (
                f"{prompt}\n"
                "Your previous response for this exact navigation decision was invalid.\n"
                f"Validation error: {exc}\n"
                "Previous response:\n"
                f"{raw_output}\n\n"
                "Return only one corrected JSON object for the same decision. Do not include explanation, "
                "markdown, prose, or any keys outside the required schema.\n"
            )

        def call_parse_stage(prompt: str, stage: str, strategy: str, parser):
            current_prompt = prompt
            last_content = None
            last_exc = None
            for attempt in range(max_parse_retries + 1):
                content, status_info = self._call_navigation_stage(
                    current_prompt,
                    stage=stage,
                    strategy=strategy,
                    step=step,
                    current_entity=current_entity,
                    aggregate_status=aggregate_status,
                    trace=trace,
                )
                if status_info.get("status") != "success" or content is None:
                    return None, content, status_info, None
                try:
                    return parser(content), content, status_info, None
                except ValueError as exc:
                    last_content = content
                    last_exc = exc
                    record_parse_error(step, stage, strategy, content, exc, attempt)
                    if attempt >= max_parse_retries:
                        break
                    aggregate_status["api_retries"] += 1
                    current_prompt = make_correction_prompt(prompt, stage, content, exc)
            return None, last_content, {"status": "parse_error", "message": str(last_exc)}, last_exc

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
                decision, content, status_info, parse_exc = call_parse_stage(
                    prompt,
                    stage="tuple",
                    strategy=strategy,
                    parser=lambda raw: self.parse_navigation_decision(raw, len(actions)),
                )
                if status_info.get("status") != "success" or content is None:
                    if parse_exc is not None:
                        return fail_parse(step, "tuple", strategy, content or "", parse_exc)
                    return fail_stage(status_info, "api_error")

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
            relation_decision, relation_content, relation_status, parse_exc = call_parse_stage(
                relation_prompt,
                stage="relation",
                strategy=strategy,
                parser=lambda raw: self.parse_relation_decision(raw, len(relation_groups)),
            )
            if relation_status.get("status") != "success" or relation_content is None:
                if parse_exc is not None:
                    return fail_parse(step, "relation", strategy, relation_content or "", parse_exc)
                return fail_stage(relation_status, "api_error")

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
            entity_decision, entity_content, entity_status, parse_exc = call_parse_stage(
                entity_prompt,
                stage="entity",
                strategy=strategy,
                parser=lambda raw: self.parse_entity_decision(raw, len(relation_actions)),
            )
            if entity_status.get("status") != "success" or entity_content is None:
                if parse_exc is not None:
                    return fail_parse(step, "entity", strategy, entity_content or "", parse_exc)
                return fail_stage(entity_status, "api_error")

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

