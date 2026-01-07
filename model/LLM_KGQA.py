from pathlib import Path

from utils.path_utils import translate_path
from utils.api_utils import list_models, chat, extract_model_ids, pick_model, load_api_config

from typing import List, Tuple

class LLM_KGQA_Client:
    def __init__(self, config_path: Path, model_choice: str = 'gemma3', debug: bool = False):
        """
        Initialize the LLM_KGQA_Client with configuration.

        Args:
            config_path (Path): Path to the configuration file.
            model_choice (str): Default model to use for the LLM API.
            debug (bool): Enable debug mode for verbose output.
        """
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
        if self.debug:
            print("\nUsing model:", self.model_choice)

    def prepare_prompt(self, question: str, triplets: List[Tuple[str, str, str]], entity_title: dict, relation_title: dict) -> str:
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
        # print(triplets_str)
        template = (
            "You will be given a natural-language question and a set of knowledge-graph triplets.\n"
            "Answer the question using ONLY the information supported by the provided triplets.\n"
            "If the answer is not entailed by the triplets, reply exactly: UNKNOWN.\n\n"
            "Return only the final answer (no explanation, no extra text).\n"
            f"Question: {question}\n\n"
            "Triplets (head, relation, tail):\n"
            f"{triplets_str}\n\n"
        )
        return template

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
        return chat(base_url=self.base_url, headers=self.headers, model=self.model_choice, user_text=user_text)

    def process_question(self, question: str, sub_graph: set, entity_title: dict, relation_title: dict) -> str:
        """
        Process a single question by preparing the prompt, sending it to the API, and extracting the prediction.

        Args:
            question (str): The natural-language question.
            sub_graph (set): The subgraph of triplets to use for the question.
            entity_title (dict): Mapping of entity IDs to titles.
            relation_title (dict): Mapping of relation IDs to titles.

        Returns:
            str: The predicted answer from the LLM.
        """
        template = self.prepare_prompt(question, list(sub_graph), entity_title, relation_title)
        out = self.chat(user_text=template)
        return out["choices"][0]["message"]["content"]