import atexit
import signal

from pathlib import Path
import random
import threading

from model.constants import valid_models, has_instruct_versions, has_quantized_versions, context_window_limits
from utils.kgqa_utils import translate_path
from utils.api_utils import list_models, chat, extract_model_ids, pick_model, load_api_config, register_cleanup_handlers, unload_model

from typing import List, Sequence, Tuple

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
            "If multiple answers are supported by the triplets, return exactly one supported answer.\n"
            "Return only the final answer (no explanation, no reasoning, no extra text).\n"
            "Double-check the spelling of your answer.\n\n"
            f"Question: {question}\n"
            f"Starting Node: {start_node_str}\n"
            "Triplets (head, relation, tail):\n"
            f"{triplets_str}\n\n"
        )
        return template, triplets_str

    def prepare_grouped_prompt(
            self,
            question: str,
            start_node: str,
            grouped_triplets: Sequence[Sequence[Tuple[str, str, str]]],
            entity_title: dict,
            relation_title: dict,
        ) -> Tuple[str, str]:
        """
        Prepare a prompt for SG-RAG style grouped subgraphs.

        Args:
            question (str): The natural-language question.
            start_node (str): The starting node for the subgraph.
            grouped_triplets (Sequence[Sequence[Tuple[str, str, str]]]): Retrieved subgraph records where each inner
                sequence contains the triplets of one matched subgraph.
            entity_title (dict): Mapping of entity IDs to titles.
            relation_title (dict): Mapping of relation IDs to titles.

        Returns:
            Tuple[str, str]: The prompt and the rendered grouped subgraph text.
        """
        start_node_str = entity_title.get(start_node, start_node)
        rendered_subgraphs = []
        used_entities = set()
        used_relations = set()

        for idx, subgraph in enumerate(grouped_triplets, start=1):
            readable_triplets = translate_path(subgraph, entity_title, relation_title)
            lines = [f"Subgraph {idx}:"]
            for (head, relation, tail), (raw_head, raw_relation, raw_tail) in zip(readable_triplets, subgraph):
                used_entities.update((raw_head, raw_tail))
                used_relations.add(raw_relation)
                lines.append(f"\t({head}, {relation}, {tail})")
            rendered_subgraphs.append("\n".join(lines))

        grouped_text = "\n\n".join(rendered_subgraphs) if rendered_subgraphs else "Subgraph 1:\n\t()"

        template = (
            "You will be given a natural-language question, a starting node, and one or more retrieved "
            "knowledge-graph subgraphs.\n"
            "Each subgraph is written as ordered (subject, relation, object) triplets.\n"
            "Use ONLY the retrieved subgraphs to answer the question.\n"
            # "If the answer is not supported by the retrieved subgraphs, reply exactly: UNKNOWN.\n"
            "If multiple answers are supported by the retrieved subgraphs, return exactly one supported answer.\n"
            "Return only the final answer (no explanation, no reasoning, no extra text).\n"
            "Double-check the spelling of your answer.\n\n"
            f"Question: {question}\n"
            f"Starting Node: {start_node_str}\n"
            "Retrieved Subgraphs:\n"
            f"{grouped_text}"
        )
        return template, grouped_text

    def prepare_path_prompt(
            self,
            question: str,
            start_node: str,
            grouped_triplets: Sequence[Sequence[Tuple[str, str, str]]],
            entity_title: dict,
            relation_title: dict,
        ) -> Tuple[str, str]:
        """
        Prepare a prompt for PathRAG style retrieved relational paths.

        Args:
            question (str): The natural-language question.
            start_node (str): The starting node for the query.
            grouped_triplets (Sequence[Sequence[Tuple[str, str, str]]]): Retrieved paths where each inner
                sequence contains the triplets of one relational path.
            entity_title (dict): Mapping of entity IDs to titles.
            relation_title (dict): Mapping of relation IDs to titles.

        Returns:
            Tuple[str, str]: The prompt and rendered path text.
        """
        start_node_str = entity_title.get(start_node, start_node)
        rendered_paths = []
        used_entities = set()
        used_relations = set()

        for idx, path_triplets in enumerate(grouped_triplets, start=1):
            readable_triplets = translate_path(path_triplets, entity_title, relation_title)
            lines = [f"Path {idx}:"]
            for (head, relation, tail), (raw_head, raw_relation, raw_tail) in zip(readable_triplets, path_triplets):
                used_entities.update((raw_head, raw_tail))
                used_relations.add(raw_relation)
                lines.append(f"\t({head}, {relation}, {tail})")
            rendered_paths.append("\n".join(lines))

        rendered_text = "\n\n".join(rendered_paths) if rendered_paths else "Path 1:\n\t()"

        template = (
            "You will be given a natural-language question, a starting node, and one or more retrieved "
            "relational paths from a knowledge graph.\n"
            "Each path is written as an ordered sequence of (subject, relation, object) triplets.\n"
            "The paths are ordered from lower to higher retrieval reliability, so later paths are usually "
            "more reliable.\n"
            "Use ONLY the retrieved paths to answer the question.\n"
            # "If the answer is not supported by the retrieved paths, reply exactly: UNKNOWN.\n"
            "If multiple answers are supported by the retrieved paths, return exactly one supported answer.\n"
            "Return only the final answer (no explanation, no reasoning, no extra text).\n"
            "Double-check the spelling of your answer.\n\n"
            f"Question: {question}\n"
            f"Starting Node: {start_node_str}\n"
            "Retrieved Paths:\n"
            f"{rendered_text}"
        )
        return template, rendered_text

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

    def process_question_grouped(
        self,
        question: str,
        start_node: str,
        grouped_triplets: Sequence[Sequence[Tuple[str, str, str]]],
        entity_title: dict,
        relation_title: dict,
    ) -> str:
        """
        Process a question using SG-RAG style grouped subgraphs.
        """
        template, grouped_text = self.prepare_grouped_prompt(
            question=question,
            start_node=start_node,
            grouped_triplets=grouped_triplets,
            entity_title=entity_title,
            relation_title=relation_title,
        )
        out, status_info = self.chat(user_text=template)
        status_info.update(self.normalize_usage(out))

        if self.debug and status_info["status"] != "success":
            print(f"LLM response status: {status_info['status']}, message: {status_info.get('message', '')}")

        if status_info["status"] == "timeout":
            return "TIMEOUT", grouped_text, status_info
        elif status_info["status"] != "success":
            return "ERROR", grouped_text, status_info

        if out is None:
            return "UNKNOWN", grouped_text, status_info

        if type(out) != dict or "message" not in out or "content" not in out["message"]:
            return "UNKNOWN", grouped_text, status_info
        return out["message"]["content"], grouped_text, status_info

    def process_question_paths(
        self,
        question: str,
        start_node: str,
        grouped_triplets: Sequence[Sequence[Tuple[str, str, str]]],
        entity_title: dict,
        relation_title: dict,
    ) -> str:
        """
        Process a question using PathRAG style retrieved paths.
        """
        template, rendered_text = self.prepare_path_prompt(
            question=question,
            start_node=start_node,
            grouped_triplets=grouped_triplets,
            entity_title=entity_title,
            relation_title=relation_title,
        )
        out, status_info = self.chat(user_text=template)
        status_info.update(self.normalize_usage(out))

        if self.debug and status_info["status"] != "success":
            print(f"LLM response status: {status_info['status']}, message: {status_info.get('message', '')}")

        if status_info["status"] == "timeout":
            return "TIMEOUT", rendered_text, status_info
        elif status_info["status"] != "success":
            return "ERROR", rendered_text, status_info

        if out is None:
            return "UNKNOWN", rendered_text, status_info

        if type(out) != dict or "message" not in out or "content" not in out["message"]:
            return "UNKNOWN", rendered_text, status_info
        return out["message"]["content"], rendered_text, status_info

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
