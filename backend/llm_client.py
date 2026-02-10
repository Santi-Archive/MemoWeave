
import os
import json
import requests
from typing import Dict, Any, Optional

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

# =========================
# Configuration (matches character.py)
# =========================

OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"
MODEL_NAME = "gpt-oss-120b"


def call_reasoning_model(system_prompt: str, user_prompt: str) -> str:
    """
    Call the LLM for reasoning tasks via OpenRouter API.
    Uses the same model and endpoint as character.py.
    """

    if not OPENROUTER_API_KEY:
        print("[LLM Client] WARNING: OPENROUTER_API_KEY not set.")
        print("[LLM Client] Returning empty reasoning graph.")
        return json.dumps({
            "temporal_relations": [],
            "causal_relations": []
        }, indent=2)

    headers = {
        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
        "Content-Type": "application/json",
    }

    payload = {
        "model": MODEL_NAME,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ],
        "temperature": 0.0
    }

    try:
        total_chars = len(system_prompt) + len(user_prompt)
        print(f"[LLM Client] Calling OpenRouter ({MODEL_NAME})...")
        print(f"[LLM Client] Total prompt size: {total_chars:,} characters")
        response = requests.post(OPENROUTER_URL, headers=headers, json=payload)
        if response.status_code != 200:
            print(f"[LLM Client] Response status: {response.status_code}")
            print(f"[LLM Client] Response body: {response.text}")
        response.raise_for_status()
        data = response.json()
        return data["choices"][0]["message"]["content"]
    except requests.exceptions.RequestException as e:
        print(f"[LLM Client] OpenRouter request failed: {e}")
        raise
