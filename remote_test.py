from pathlib import Path
from utils.api_utils import list_models, chat, extract_model_ids, pick_model, load_api_config

class RemoteTestClient:
    def __init__(self, config_path: Path):
        """
        Initialize the RemoteTestClient with configuration.

        Args:
            config_path (Path): Path to the configuration file.
        """
        self.base_url, self.api_key = load_api_config(config_path)
        self.headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

    def list_models(self):
        """
        Fetch the list of available models.

        Returns:
            dict: JSON response containing the list of models.
        """
        return list_models(base_url=self.base_url, headers=self.headers)

    def chat(self, model: str, user_text: str, seed: int | None = None):
        """
        Send a chat message to the API and get the response.

        Args:
            model (str): The model ID to use for the chat.
            user_text (str): The user's input text.

        Returns:
            dict: JSON response from the API.
        """
        return chat(base_url=self.base_url, headers=self.headers, model=model, user_text=user_text, seed=seed)

if __name__ == "__main__":
    CONFIG_PATH = Path(__file__).with_name("openwebui_config.json").parent / "configs" / "openwebui_config.json"

    client = RemoteTestClient(CONFIG_PATH)

    models_resp = client.list_models()
    print("Models response JSON:")
    # print(models_resp)
    model_ids = extract_model_ids(models_resp)

    if not model_ids:
        raise RuntimeError(f"Couldn't parse model list response: {models_resp}")

    print("Available models:")
    for name, mid in model_ids.items():
        print(f"- {name}: {mid}")

    chosen = pick_model(model_ids, choice="gemma3")
    print("\nUsing model:", chosen)

    out = client.chat(model=chosen, user_text="Write a short Python function that computes gcd(a,b).")
    print("\nResponse JSON:")
    print(out)
