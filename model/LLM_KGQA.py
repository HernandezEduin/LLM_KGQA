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

class LLM_KGQA_Client:
    def __init__(
        self, 
        config_path: Path, 
        model_choice: str = 'gemma3', 
        seed: int | None = None, 
        timeout: int = 120, 
        debug: bool = False
    ):
        """
        Initialize the LLM_KGQA_Client with configuration.

        Args:
            config_path (Path): Path to the configuration file.
            model_choice (str): Default model to use for the LLM API.
            seed (int | None): Optional random seed for the requests.
            timeout (int): Timeout in seconds for LLM API requests.
            debug (bool): Enable debug mode for verbose output.
        """
        self.timeout = timeout
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
        if self.debug and status_info["status"] != "success":
            print(f"LLM response status: {status_info['status']}, message: {status_info.get('message', '')}")

        if status_info["status"] == "timeout":
            return "TIMEOUT", triplets_str, status_info['elapsed_time']
        elif status_info["status"] != "success":
            return "ERROR", triplets_str, status_info['elapsed_time']
        
        if out is None:
            return "UNKNOWN", triplets_str, status_info['elapsed_time']
        if type(out) != dict or "choices" not in out or len(out["choices"]) == 0:
            return "UNKNOWN", triplets_str, status_info['elapsed_time']
        return out["choices"][0]["message"]["content"], triplets_str, status_info['elapsed_time']