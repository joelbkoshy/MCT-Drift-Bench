"""Safety axis: recall and specificity of the deterministic pre-generation screener.

No model is involved. The screener is a fixed regex bank that runs before any
generation and, on a match, escalates instead of replying. This axis measures
how that bank behaves across the ways risk is actually phrased, and — through
the two control families — what it costs in false alarms.
"""

from __future__ import annotations

import csv
from pathlib import Path

from src.bench.safety_probes import CONTROL_FAMILIES, ESCALATION_FAMILIES, SAFETY_PROBES
from src.core.safety_check import check_safety

BASE_DIR = Path(__file__).resolve().parents[2]
OUT_PATH = BASE_DIR / "results" / "safety_runs.csv"

FIELDS = ["family", "expect_escalate", "escalated", "correct", "matched_patterns", "text"]


def run(out_path: Path = OUT_PATH) -> list[dict]:
    rows: list[dict] = []
    for probe in SAFETY_PROBES:
        result = check_safety(probe.text)
        escalated = result.status == "HIGH_RISK"
        rows.append(
            {
                "family": probe.family,
                "expect_escalate": probe.expect_escalate,
                "escalated": escalated,
                "correct": escalated == probe.expect_escalate,
                "matched_patterns": "; ".join(result.matched_keywords),
                "text": probe.text,
            }
        )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    return rows


def summarise(rows: list[dict]) -> dict:
    tp = sum(r["escalated"] and r["expect_escalate"] for r in rows)
    fn = sum(not r["escalated"] and r["expect_escalate"] for r in rows)
    fp = sum(r["escalated"] and not r["expect_escalate"] for r in rows)
    tn = sum(not r["escalated"] and not r["expect_escalate"] for r in rows)

    recall = tp / (tp + fn) if tp + fn else float("nan")
    precision = tp / (tp + fp) if tp + fp else float("nan")
    specificity = tn / (tn + fp) if tn + fp else float("nan")
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0

    per_family = {}
    for family in ESCALATION_FAMILIES + CONTROL_FAMILIES:
        subset = [r for r in rows if r["family"] == family]
        per_family[family] = {
            "n": len(subset),
            "correct": sum(r["correct"] for r in subset),
            "rate": sum(r["correct"] for r in subset) / len(subset) if subset else float("nan"),
        }

    return {
        "tp": tp, "fn": fn, "fp": fp, "tn": tn,
        "recall": recall, "precision": precision,
        "specificity": specificity, "f1": f1,
        "per_family": per_family,
    }
