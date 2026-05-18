import os
from typing import Any, Dict, Optional, Tuple

import requests


DEFAULT_OLLAMA_URL = "http://127.0.0.1:11434"


def query_ollama_chat(
    model: str,
    system_prompt: str,
    user_prompt: str = "",
    temperature: float = 0,
    max_tokens: Optional[int] = None,
) -> Tuple[str, Dict[str, Any]]:
    """Query a local Ollama chat model and return response text plus metadata."""
    base_url = os.environ.get("OLLAMA_HOST", DEFAULT_OLLAMA_URL).rstrip("/")
    timeout = float(os.environ.get("OLLAMA_TIMEOUT", "300"))

    messages = [{"role": "system", "content": system_prompt}]
    if user_prompt:
        messages.append({"role": "user", "content": user_prompt})

    options: Dict[str, Any] = {"temperature": temperature}
    if max_tokens is not None:
        options["num_predict"] = max_tokens

    payload = {
        "model": model,
        "messages": messages,
        "stream": False,
        "options": options,
    }

    response = requests.post(f"{base_url}/api/chat", json=payload, timeout=timeout)
    response.raise_for_status()
    data = response.json()
    text = data.get("message", {}).get("content", "")
    if not text:
        raise RuntimeError("Ollama returned an empty chat response")
    return text, data
