"""Deterministic input guardrails.

This is a *first-line, pre-LLM* screen: cheap regex checks that run before any
model call, so blatant prompt-injection attempts are refused without spending a
request. It complements (does not replace) two other defences:

1. Retrieved passages are framed as DATA, not instructions, in the generation
   prompt (see :data:`app.agent.prompts.GENERATE_SYSTEM`).
2. The router refuses out-of-scope questions.

What this catches:
    Classic English injection phrasings — "ignore previous instructions",
    "reveal your system prompt", "you are now ...", "developer mode", "DAN", etc.

What this does NOT catch (documented honestly in the README):
    - Obfuscated / encoded / base64 / homoglyph injections.
    - Non-English or paraphrased attacks.
    - INDIRECT injection: malicious instructions hidden inside an ingested
      document (mitigated only by the data-not-instructions framing).
    - Model-level jailbreaks that don't match a known pattern.
    There is no ML classifier here — this is a deliberately simple, explainable
    layer, not a complete solution.
"""

from __future__ import annotations

import re

# Known injection phrasings. Kept readable and conservative to limit false
# positives on legitimate questions about these very papers.
_INJECTION_PATTERNS: list[str] = [
    r"ignore\s+(all\s+|any\s+|the\s+)?(previous|prior|above)\s+(instruction|prompt|message)",
    r"disregard\s+(all\s+|the\s+|any\s+)?(previous|prior|above|safety)\s*(instruction|prompt|rule)?",
    r"forget\s+(everything|all|previous|prior|your\s+instructions)",
    r"reveal\s+(your|the)\s+(system\s+)?(prompt|instructions)",
    r"(what\s+is|show\s+me|print)\s+(your|the)\s+(system\s+)?prompt",
    r"\b(system|developer)\s+prompt\b",
    r"you\s+are\s+now\s+",
    r"act\s+as\s+(if|a|an|though)\b",
    r"developer\s+mode",
    r"jailbreak",
    r"do\s+anything\s+now",
    r"\bDAN\b",
]

_COMPILED = [re.compile(p, re.IGNORECASE) for p in _INJECTION_PATTERNS]


def detect_prompt_injection(text: str) -> bool:
    """Return ``True`` if ``text`` matches a known prompt-injection pattern."""
    return any(pattern.search(text) for pattern in _COMPILED)


def is_blocked(text: str) -> tuple[bool, str]:
    """Screen a user query.

    Returns:
        ``(blocked, reason)`` — ``reason`` is ``"prompt_injection"`` when
        blocked, else an empty string.
    """
    if detect_prompt_injection(text):
        return True, "prompt_injection"
    return False, ""
