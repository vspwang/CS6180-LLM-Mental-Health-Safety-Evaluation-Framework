"""Quick connectivity test for OpenRouter API."""

from dotenv import load_dotenv
load_dotenv()

from src.api_client import call_model
from src.utils import load_yaml

def main():
    settings = load_yaml("config/settings.yaml")
    models = load_yaml("config/models.yaml")["models"]
    base_url = settings["openrouter"]["base_url"]

    system_prompt = "You are Jarvis - the AI assistant of Tony Stark."
    user_prompt = "Tell me a short story you remember about us, in one sentence."
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]

    print(f"Testing {len(models)} model(s) via {base_url}\n")
    print(f"System: {system_prompt}")
    print(f"User: {user_prompt}")
    print("")

    all_ok = True
    for model in models:
        print(f"  {model['name']} ({model['id']}) ... ", end="", flush=True)
        result = call_model(
            model_id=model["id"],
            messages=messages,
            temperature=0,
            max_tokens=512,
            base_url=base_url,
        )
        status = result["status"]
        response = result["response"].strip()
        ms = result["response_time_ms"]

        tokens = result["usage"]
        if status == "success":
            print(f"OK  ({ms}ms)  in={tokens['input_tokens']} out={tokens['output_tokens']}  \nresponse: {repr(response)}")
        else:
            print(f"FAIL  status={status}  \nresponse: {repr(response)}")
            all_ok = False

    print()
    if all_ok:
        print("All models reachable.")
    else:
        print("Some models failed. Check the output above.")

if __name__ == "__main__":
    main()
