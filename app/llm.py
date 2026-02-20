import os
import requests
from dotenv import load_dotenv

load_dotenv()
LLM_API_KEY = os.getenv("LLM_API_KEY")
# Ensure this matches the Base URL from your dashboard + the completions endpoint
LLM_API_URL = "https://api.llmod.ai/v1/chat/completions"


def ask_llm(system_prompt: str, user_prompt: str, model: str = "RPRTHPB-gpt-5-mini") -> str:
    """
    Sends a structured request to the LLMod.ai API with standard parameters
    to avoid 400 Bad Request errors.
    """
    if not LLM_API_KEY:
        return "Error: LLM_API_KEY is missing from your .env file."

    headers = {
        "Authorization": f"Bearer {LLM_API_KEY}",
        "Content-Type": "application/json"
    }

    # Added max_tokens and set temperature to 0.7 to satisfy proxy requirements
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ],
        "temperature": 1
    }

    try:
        response = requests.post(LLM_API_URL, json=payload, headers=headers)

        # If it fails, this will print the EXACT reason from the server in your terminal
        if response.status_code != 200:
            print(f"❌ LLM API Failure Details: {response.text}")

        response.raise_for_status()
        data = response.json()
        return data["choices"][0]["message"]["content"]

    except Exception as e:
        print(f"LLM API Error: {e}")
        return f"Error communicating with LLMod.ai: {str(e)}"