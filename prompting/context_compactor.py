import os
import re
from typing import Iterable, List


DEFAULT_ITEM_CHARS = 1200
DEFAULT_HISTORY_ITEMS = 12


def _limit() -> int:
    return int(os.environ.get("PROMPT_COMPACT_ITEM_CHARS", DEFAULT_ITEM_CHARS))


def _history_items() -> int:
    return int(os.environ.get("PROMPT_COMPACT_HISTORY_ITEMS", DEFAULT_HISTORY_ITEMS))


def strip_thinking(text: str) -> str:
    """Remove common Qwen/Ollama thinking blocks before reusing text as context."""
    if text is None:
        return ""

    text = re.sub(
        r"(?is)\bThinking\.\.\..*?\.\.\.done thinking\.\s*",
        "",
        text,
    )
    text = re.sub(r"(?is)<think>.*?</think>\s*", "", text)
    return text.strip()


def compact_text(text: str, max_chars: int = None) -> str:
    text = strip_thinking(text)
    if not text:
        return ""

    if "EXECUTE" in text:
        text = "EXECUTE" + text.split("EXECUTE", 1)[1]

    text = re.sub(r"\n{3,}", "\n\n", text).strip()
    max_chars = _limit() if max_chars is None else max_chars
    if len(text) <= max_chars:
        return text

    head = text[: max_chars // 3].rstrip()
    tail = text[-(max_chars - len(head) - 40) :].lstrip()
    return f"{head}\n...[context compacted]...\n{tail}"


def compact_items(items: Iterable[str], max_items: int = None) -> List[str]:
    compacted = [compact_text(item) for item in items]
    compacted = [item for item in compacted if item]
    max_items = _history_items() if max_items is None else max_items
    if len(compacted) <= max_items:
        return compacted
    omitted = len(compacted) - max_items
    return [f"[Context compacted: omitted {omitted} older item(s)]"] + compacted[-max_items:]
