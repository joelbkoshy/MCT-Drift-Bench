"""Corpus access with contamination control.

Both evaluation sets are drawn from the same public corpus that supplied the
supervised fine-tuning data for the adapted arm. Any narrative that entered that
set — train or holdout — is excluded here, so the adapted arm is never scored on
text it was tuned on and the comparison against the base arms stays honest.
"""

from __future__ import annotations

import json
import random
import re
from functools import lru_cache
from pathlib import Path

import pandas as pd

from src.core.safety_check import check_safety

BASE_DIR = Path(__file__).resolve().parents[2]
CORPUS_PATH = BASE_DIR / "data" / "narratives_public.csv"
EXCLUDED_PATH = BASE_DIR / "data" / "sft_excluded_ids.json"

SEED = 20260828
CAS_CATEGORIES = ("rumination", "threat_monitoring", "avoidance", "neutral")
MIN_WORDS, MAX_WORDS = 40, 120


def _clean(text: str) -> str:
    text = re.sub(r"http\S+", "", str(text))
    return re.sub(r"\s+", " ", text).strip()


@lru_cache(maxsize=1)
def excluded_ids() -> frozenset[int]:
    return frozenset(json.loads(EXCLUDED_PATH.read_text(encoding="utf-8"))["ids"])


@lru_cache(maxsize=1)
def eligible() -> pd.DataFrame:
    """Crisis-free, length-bounded, non-contaminated narratives by CAS category."""
    df = pd.read_csv(CORPUS_PATH)
    df["text"] = df["text"].map(_clean)
    df["n_words"] = df["text"].str.split().str.len()

    df = df[
        df["n_words"].between(MIN_WORDS, MAX_WORDS)
        & df["category"].isin(CAS_CATEGORIES)
        & ~df["id"].isin(excluded_ids())
        & (df["ground_truth_crisis"].astype(str).str.lower().isin(["false", "0", "nan"]))
    ]
    # A narrative that trips the pre-generation screener is escalated rather than
    # answered, so it can never reach the model and cannot be scored.
    keep = df["text"].map(lambda t: check_safety(t).status == "SAFE")
    return df[keep].reset_index(drop=True)


def stratified_sample(per_category: int, exclude_ids: set[int] | None = None) -> list[dict]:
    """Draw ``per_category`` narratives from each CAS category, seeded."""
    pool = eligible()
    if exclude_ids:
        pool = pool[~pool["id"].isin(exclude_ids)]

    rng = random.Random(SEED)
    picked: list[dict] = []
    for category in CAS_CATEGORIES:
        rows = pool[pool["category"] == category].reset_index(drop=True)
        for i in sorted(rng.sample(range(len(rows)), per_category)):
            r = rows.iloc[i]
            picked.append(
                {
                    "narrative_id": int(r["id"]),
                    "category": category,
                    "text": r["text"],
                    "n_words": int(r["n_words"]),
                }
            )
    return picked
