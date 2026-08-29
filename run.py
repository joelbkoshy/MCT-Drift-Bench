"""MCT-DriftBench command line entry point.

    python run.py probes      build the probe set from the public corpus
    python run.py safety      score the deterministic screener (no model needed)
    python run.py drift       run every arm against every drift probe
    python run.py judge       adjudicate the drift generations with the judge model
    python run.py fidelity    run every arm against unbaited real narratives
    python run.py analyse     build tables, statistics and figures
    python run.py all         everything above, in order
"""

from __future__ import annotations

import sys
from pathlib import Path

USAGE = __doc__

BASE_DIR = Path(__file__).resolve().parent
LOCK_PATH = BASE_DIR / "results" / ".run.lock"

# Stages that write generations; only these can corrupt data if run twice at once.
_GENERATING = {"drift", "fidelity", "judge", "all"}


def main(argv: list[str]) -> int:
    if len(argv) < 2 or argv[1] in {"-h", "--help"}:
        print(USAGE)
        return 0

    command = argv[1]
    no_reference = "--no-reference" in argv

    if command in _GENERATING:
        from src.bench.runner import single_instance

        with single_instance(LOCK_PATH):
            return _dispatch(command, no_reference)
    return _dispatch(command, no_reference)


def _dispatch(command: str, no_reference: bool) -> int:
    if command in {"probes", "all"}:
        from src.bench.build_probes import build
        print(f"probes: {len(build())}")

    if command in {"safety", "all"}:
        from src.bench.safety_eval import run, summarise
        rows = run()
        s = summarise(rows)
        print(f"safety: n={len(rows)} recall={s['recall']:.3f} "
              f"specificity={s['specificity']:.3f} F1={s['f1']:.3f}")

    if command in {"drift", "all"}:
        from src.bench.drift_eval import run
        run(include_reference=not no_reference)

    if command in {"fidelity", "all"}:
        from src.bench.ablation import run
        run(include_reference=not no_reference)

    if command in {"judge", "all"}:
        from src.bench.drift_eval import judge_run
        judge_run()

    if command in {"analyse", "analyze", "all"}:
        from src.bench.analysis import run
        summary = run()
        print("wrote results/summary.json and figures/")
        for row in summary["drift_by_arm"]:
            print(f"  {row['arm']:>18}  DRS={row['drs']:.3f}  fidelity={row['fidelity']:.2f}")

    if command not in {"probes", "safety", "drift", "fidelity", "judge",
                       "analyse", "analyze", "all"}:
        print(USAGE)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
