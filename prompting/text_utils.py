import re

_THINK_TAG_RE = re.compile(r"<think>.*?</think>", re.DOTALL)
_THINK_BLOCK_RE = re.compile(r"Thinking\.\.\..*?\.\.\.done thinking\.", re.DOTALL)


def strip_think(text: str) -> str:
    if not text:
        return text or ""
    text = _THINK_TAG_RE.sub("", text)
    text = _THINK_BLOCK_RE.sub("", text)
    return text
