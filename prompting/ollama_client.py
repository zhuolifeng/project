import os
import re
from typing import Any, Dict, Optional, Tuple

import requests


DEFAULT_OLLAMA_URL = "http://127.0.0.1:11434"


def _parse_ollama_think(value: str) -> Any:
    normalized = value.strip().lower()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False
    if normalized in {"low", "medium", "high"}:
        return normalized
    raise ValueError(
        "OLLAMA_THINK must be one of: true, false, low, medium, high"
    )


def strip_think_blocks(text: str) -> str:
    text = re.sub(r"(?is)<think>.*?</think>\s*", "", text or "")
    text = re.sub(
        r"(?is)\bThinking\.\.\..*?\.\.\.done thinking\.\s*",
        "",
        text,
    )
    return text.strip()


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
    think_value = os.environ.get("OLLAMA_THINK")
    think = _parse_ollama_think(think_value) if think_value is not None else None

    messages = [{"role": "system", "content": system_prompt}]
    if user_prompt:
        messages.append({"role": "user", "content": user_prompt})

    options: Dict[str, Any] = {"temperature": temperature}
    if max_tokens is not None:
        options["num_predict"] = max_tokens

    payload: Dict[str, Any] = {
        "model": model,
        "messages": messages,
        "stream": False,
        "options": options,
    }
    if think is not None:
        payload["think"] = think

    response = requests.post(f"{base_url}/api/chat", json=payload, timeout=timeout)
    response.raise_for_status()
    data = response.json()
    message = data.get("message", {})
    text = strip_think_blocks(message.get("content", ""))
    if not text:
        thinking = message.get("thinking", "")
        done_reason = data.get("done_reason", "unknown")
        eval_count = data.get("eval_count", "unknown")
        if thinking and max_tokens is not None:
            retry_options = dict(options)
            retry_options["num_predict"] = max_tokens
            retry_payload = dict(payload)
            retry_payload["options"] = retry_options
            response = requests.post(
                f"{base_url}/api/chat",
                json=retry_payload,
                timeout=timeout,
            )
            response.raise_for_status()
            data = response.json()
            message = data.get("message", {})
            text = strip_think_blocks(message.get("content", ""))
            if text:
                return text, data

        hint = ""
        if thinking:
            hint = (
                " The model produced thinking tokens but no final content; "
                "set OLLAMA_THINK=false to reserve the 2048 token budget for final content."
            )
        raise RuntimeError(
            "Ollama returned an empty chat response "
            f"(done_reason={done_reason}, eval_count={eval_count}).{hint}"
        )
    return text, data
