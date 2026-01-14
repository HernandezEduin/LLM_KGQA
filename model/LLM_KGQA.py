from pathlib import Path
import random

from utils.kgqa_utils import translate_path
from utils.api_utils import list_models, chat, extract_model_ids, pick_model, load_api_config

from typing import List, Tuple

valid_models = [
    'gemma3', 
    'llama3', 
    'llama3.1', 
    'deepseek-r1', 
    'qwen2.5', 
    'gpt-oss', 
    'mixtral', 
    'vicuna', 
    'phi3'
]

context_window_limits = {
    'gemma3': 128*1024,
    'llama3': 8*1024,
    'llama3.1': 128*1024,
    'deepseek-r1': 128*1024,
    'qwen2.5': 32*1024,
    'gpt-oss': 128*1024,
    'mixtral': 32*1024,
    'vicuna': 4*1024,
    'phi3': 128*1024,
}

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
        context_window: int = 4096,
        seed: int | None = None, 
        timeout: int = 120, 
        debug: bool = False
    ):
        """
        Initialize the LLM_KGQA_Client with configuration.

        Args:
            config_path (Path): Path to the configuration file.
            model_choice (str): Default model to use for the LLM API.
            context_window (int): Context window size for the model.
            seed (int | None): Optional random seed for the requests.
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
        
        self.timeout = timeout
        self.context_window = context_window
        self.seed = seed
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

        self.change_llm(model_choice)

    def change_llm(self, model_name: str):
        """
        Change the current LLM model.

        Args:
            model_name (str): Name of the model to switch to.
        """
        self.model_choice = pick_model(self.model_ids, choice=model_name)
        # if self.debug:
        print("\nUsing model:", self.model_choice)

    def prepare_prompt(
            self, 
            question: str, 
            triplets: List[Tuple[str, str, str]], 
            entity_title: dict,
            relation_title: dict
        ) -> Tuple[str, str]:
        """
        Prepare the prompt for the LLM based on the question and triplets.

        Args:
            question (str): The natural-language question.
            triplets (List[Tuple[str, str, str]]): Knowledge-graph triplets.
            entity_title (dict): Mapping of entity IDs to titles.
            relation_title (dict): Mapping of relation IDs to titles.

        Returns:
            str: The formatted prompt string.
        """
        triplets_str = translate_path(triplets, entity_title, relation_title)
        triplets_str = "{\n" + "\n".join([f"\t({h}, {r}, {t})" for h, r, t in triplets_str]) + "\n}"
        template = (
            "You will be given a natural-language question and a set of knowledge-graph triplets.\n"
            "Answer the question using ONLY the information supported by the provided triplets.\n"
            # "If the answer is not entailed by the triplets, reply exactly: UNKNOWN.\n\n"
            "Each question contains a unique answer.\n"
            "Return only the final answer (no explanation, no reasoning, no extra text).\n"
            "Double-check the spelling of your answer.\n\n"
            f"Question: {question}\n"
            "Triplets (head, relation, tail):\n"
            f"{triplets_str}\n\n"
        )
        return template, triplets_str

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
            timeout=self.timeout
        )

    def process_question(
        self, 
        question: str, 
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
        template, triplets_str = self.prepare_prompt(question, sub_graph, entity_title, relation_title)
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