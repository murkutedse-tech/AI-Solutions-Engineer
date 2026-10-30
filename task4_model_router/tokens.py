"""Token estimation used for admission control before the real usage is known.

A ~4 characters/token heuristic plus per-message framing overhead. It is only an
estimate for the *reservation*; the reservation is reconciled to the provider's
reported ``usage.total_tokens`` afterwards, so estimation error never accumulates.
Swap in a real tokenizer (e.g. tiktoken) per model if tighter admission is needed.
"""

from __future__ import annotations

import math
from typing import Any

CHARS_PER_TOKEN = 4
PER_MESSAGE_OVERHEAD = 4
PER_REQUEST_OVERHEAD = 3
DEFAULT_MAX_COMPLETION_TOKENS = 1024


def estimate_text_tokens(text: str) -> int:
    return math.ceil(len(text) / CHARS_PER_TOKEN)


def estimate_prompt_tokens(messages: list[dict[str, Any]]) -> int:
    total = PER_REQUEST_OVERHEAD
    for message in messages:
        content = message.get("content")
        if isinstance(content, list):  # multi-part content: count the text parts
            content = " ".join(p.get("text", "") for p in content if isinstance(p, dict))
        total += PER_MESSAGE_OVERHEAD + estimate_text_tokens(content if isinstance(content, str) else "")
    return total


def reservation_tokens(payload: dict[str, Any]) -> int:
    """Worst-case cost to reserve: prompt estimate + the completion budget the caller asked for."""
    max_completion = payload.get("max_tokens") or payload.get("max_completion_tokens") or DEFAULT_MAX_COMPLETION_TOKENS
    return estimate_prompt_tokens(payload.get("messages", [])) + int(max_completion)


def actual_tokens(response: dict[str, Any], payload: dict[str, Any]) -> int:
    """Provider-reported usage, falling back to an estimate when the provider omits it."""
    usage = response.get("usage")
    if isinstance(usage, dict) and isinstance(usage.get("total_tokens"), int):
        return usage["total_tokens"]
    completion = "".join(
        (c.get("message") or {}).get("content") or "" for c in response.get("choices", []) if isinstance(c, dict)
    )
    return estimate_prompt_tokens(payload.get("messages", [])) + estimate_text_tokens(completion)
