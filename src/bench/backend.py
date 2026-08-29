"""Local generation backend.

Every arm in this study runs on a local Ollama server. Nothing leaves the
machine, which is a requirement inherited from the earlier phases: the clinical
corpus used there carries a data-use agreement that forbids third-party
transmission, and keeping generation local makes the two settings comparable.

Decoding is pinned and seeded so that a rerun reproduces the reported numbers.
"""

from __future__ import annotations

import time

import ollama

TEMPERATURE = 0.3
TOP_P = 0.9
NUM_PREDICT = 512
SEED = 20260828

# Fluent declinations pass every lexical detector and would be scored as
# well-behaved therapeutic text. They are generation failures, not responses.
_REFUSAL_OPENERS = (
    "i cannot", "i can not", "i can't", "i am unable", "i'm unable",
    "i won't", "i will not", "i'm sorry, but i", "i am sorry, but i",
    "as an ai", "i'm not able to", "i am not able to",
    "unfortunately, i cannot", "unfortunately i cannot",
)


def is_refusal(text: str) -> bool:
    return (text or "").strip().lower().startswith(_REFUSAL_OPENERS)


def generate(
    system: str | None,
    user: str,
    model: str,
    num_predict: int = NUM_PREDICT,
) -> tuple[str, float]:
    """Return (response_text, wall_clock_seconds)."""
    messages = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": user})

    started = time.perf_counter()
    response = ollama.chat(
        model=model,
        messages=messages,
        options={
            "temperature": TEMPERATURE,
            "top_p": TOP_P,
            "num_predict": num_predict,
            "seed": SEED,
        },
    )
    elapsed = time.perf_counter() - started
    return response["message"]["content"].strip(), elapsed


def installed_models() -> set[str]:
    try:
        listing = ollama.list()
    except Exception:
        return set()
    names: set[str] = set()
    for entry in listing.get("models", []):
        name = entry.get("model") or entry.get("name") or ""
        if name:
            names.add(name)
            names.add(name.split(":")[0])
    return names
