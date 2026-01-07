from pathlib import Path
from utils.api_utils import load_api_config, list_models, chat, extract_model_ids, pick_model

CONFIG_PATH = Path(__file__).with_name("openwebui_config.json").parent / "configs" / "openwebui_config.json"

BASE_URL, API_KEY = load_api_config(CONFIG_PATH)

HEADERS = {
    "Authorization": f"Bearer {API_KEY}",
    "Content-Type": "application/json",
}

if __name__ == "__main__":
    models_resp = list_models(base_url=BASE_URL, headers=HEADERS)
    model_ids = extract_model_ids(models_resp)

    if not model_ids:
        raise RuntimeError(f"Couldn't parse model list response: {models_resp}")

    print("Available models:")
    for i, mid in enumerate(model_ids, start=1):
        print(f"  {i:>2}. {mid}")

    chosen = pick_model(model_ids)
    print("\nUsing model:", chosen)

    out = chat(base_url=BASE_URL, headers=HEADERS, model=chosen, user_text="Write a short Python function that computes gcd(a,b).")
    print("\nResponse JSON:")
    print(out["choices"][0]["message"]["content"])
