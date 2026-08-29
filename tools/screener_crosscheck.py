"""Run any phase's crisis screener against the benchmark's safety axis.

The screener is the one component that has been carried unchanged through every
phase of this programme, and each phase has reported its performance on that
phase's own narratives. This script points the benchmark's phrasing families at
a screener implementation given by path, so the earlier reported figures can be
re-derived on a common instrument instead of being compared across different
test sets.

    python tools/screener_crosscheck.py <path to safety_check.py> [more paths...]
"""

from __future__ import annotations

import importlib.util
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.bench.safety_probes import (  # noqa: E402
    CONTROL_FAMILIES,
    ESCALATION_FAMILIES,
    SAFETY_PROBES,
)


def load_screener(path: Path):
    spec = importlib.util.spec_from_file_location(f"screener_{abs(hash(path))}", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def evaluate(module) -> dict:
    per_family: dict[str, list[int]] = defaultdict(lambda: [0, 0])
    tp = fn = fp = tn = 0

    for probe in SAFETY_PROBES:
        escalated = module.check_safety(probe.text).status == "HIGH_RISK"
        per_family[probe.family][1] += 1
        per_family[probe.family][0] += int(escalated == probe.expect_escalate)
        if probe.expect_escalate:
            tp += escalated
            fn += not escalated
        else:
            fp += escalated
            tn += not escalated

    recall = tp / (tp + fn) if tp + fn else float("nan")
    precision = tp / (tp + fp) if tp + fp else 0.0
    specificity = tn / (tn + fp) if tn + fp else float("nan")
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "per_family": dict(per_family),
        "tp": tp, "fn": fn, "fp": fp, "tn": tn,
        "recall": recall, "precision": precision,
        "specificity": specificity, "f1": f1,
    }


def n_patterns(module) -> int:
    return len(getattr(module, "CRISIS_PATTERNS", []))


def main(paths: list[str]) -> int:
    if not paths:
        print(__doc__)
        return 1

    for raw in paths:
        path = Path(raw).resolve()
        module = load_screener(path)
        result = evaluate(module)

        print(f"\n{path}")
        print(f"  {n_patterns(module)} crisis patterns")
        print(f"  {'family':<10} {'correct':>9}   expectation")
        for family in ESCALATION_FAMILIES + CONTROL_FAMILIES:
            correct, total = result["per_family"].get(family, (0, 0))
            expect = "escalate" if family in ESCALATION_FAMILIES else "proceed"
            print(f"  {family:<10} {correct:>4}/{total:<4}   {expect}")
        print(
            f"  recall={result['recall']:.3f}  precision={result['precision']:.3f}  "
            f"specificity={result['specificity']:.3f}  F1={result['f1']:.3f}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
