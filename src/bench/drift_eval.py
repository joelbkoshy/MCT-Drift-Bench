"""Drift axis: run every arm against every probe and score the outcome.

Each cell produces one generation, which is then scored three ways:

  * the family-specific lexical drift detector (the benchmark's own metric),
  * the general MCT constraint filter and fidelity scale carried over from the
    earlier phases, so drift can be read against overall adherence,
  * an independent model judge, run separately by ``judge_run`` so that
    adjudication can be repeated without regenerating.
"""

from __future__ import annotations

from pathlib import Path

from src.bench import rag
from src.bench.backend import generate, installed_models, is_refusal
from src.bench.build_probes import load as load_probes
from src.bench.build_probes import probe_hash
from src.bench.conditions import ARMS, REFERENCE_ARM, build_user_message
from src.bench.judge import adjudicate
from src.bench.runner import ResumableWriter, progress
from src.bench.taxonomy import detect_drift
from src.core.constraint_filter import check_constraints, score_mct_indicators
from src.core.fidelity_scorer import compute_fidelity

BASE_DIR = Path(__file__).resolve().parents[2]
DRIFT_PATH = BASE_DIR / "results" / "drift_runs.csv"
JUDGE_PATH = BASE_DIR / "results" / "drift_judgements.csv"
MANIFEST_PATH = BASE_DIR / "results" / "run_manifest.json"

FIELDS = [
    "arm", "model", "weights", "specification", "rag",
    "probe_id", "family", "family_name", "axis",
    "carrier_id", "carrier_category",
    "drifted", "drift_hits", "held", "hold_hits",
    "constraint_compliant", "violations",
    "fidelity_overall", "fidelity_grade",
    "refused", "latency_s", "n_words", "response",
]


def _check_manifest() -> None:
    """Refuse to append generations produced against a different probe set.

    A run resumed days later, after the taxonomy or carrier sampling has been
    touched, would otherwise pool rows that answered different probes under the
    same probe ids -- and nothing downstream could detect it.
    """
    import json

    current = probe_hash()
    if MANIFEST_PATH.exists():
        recorded = json.loads(MANIFEST_PATH.read_text(encoding="utf-8")).get("probe_hash")
        if recorded and recorded != current:
            raise RuntimeError(
                f"probe set has changed since this run began "
                f"(recorded {recorded}, current {current}). Move results/ aside "
                "and start a fresh run, or restore the previous probe set."
            )
    else:
        MANIFEST_PATH.parent.mkdir(parents=True, exist_ok=True)
        MANIFEST_PATH.write_text(
            json.dumps({"probe_hash": current}, indent=2), encoding="utf-8"
        )


def run(include_reference: bool = True, out_path: Path = DRIFT_PATH) -> None:
    _check_manifest()
    probes = load_probes()
    arms = list(ARMS) + ([REFERENCE_ARM] if include_reference else [])

    available = installed_models()
    missing = {a.model for a in arms} - available
    if missing:
        raise RuntimeError(f"models not installed in Ollama: {sorted(missing)}")

    total = len(arms) * len(probes)
    done = 0
    with ResumableWriter(out_path, FIELDS, key=("arm", "probe_id")) as writer:
        for arm in arms:
            for probe in probes:
                done += 1
                if writer.already_done(arm.name, probe["probe_id"]):
                    continue

                context = rag.retrieve(probe["probe_text"]) if arm.use_rag else None
                user = build_user_message(probe["patient_block"], context)
                try:
                    text, latency = generate(arm.system_prompt, user, arm.model)
                except Exception as exc:
                    text, latency = f"[GENERATION ERROR: {exc}]", 0.0

                refused = is_refusal(text)
                verdict = detect_drift(text, probe["family"])
                constraint = check_constraints(text)
                fidelity = compute_fidelity(text)

                writer.write({
                    "arm": arm.name, "model": arm.model, "weights": arm.weights,
                    "specification": arm.specification, "rag": arm.use_rag,
                    "probe_id": probe["probe_id"], "family": probe["family"],
                    "family_name": probe["family_name"], "axis": probe["axis"],
                    "carrier_id": probe["carrier_id"],
                    "carrier_category": probe["carrier_category"],
                    "drifted": verdict.drifted,
                    "drift_hits": "; ".join(verdict.drift_hits),
                    "held": verdict.held,
                    "hold_hits": "; ".join(verdict.hold_hits),
                    "constraint_compliant": constraint.is_compliant,
                    "violations": "; ".join(constraint.violations),
                    "fidelity_overall": fidelity.overall_score,
                    "fidelity_grade": fidelity.grade,
                    "refused": refused,
                    "latency_s": round(latency, 2),
                    "n_words": len(text.split()),
                    "response": text,
                })
                progress("drift", done, total, f"{arm.name} {probe['probe_id']} {latency:.1f}s")


JUDGE_FIELDS = ["arm", "probe_id", "family", "lexical_drifted", "judge_verdict", "judge_rationale"]


def judge_run(runs_path: Path = DRIFT_PATH, out_path: Path = JUDGE_PATH) -> None:
    """Adjudicate every generation with the independent judge model."""
    import csv

    probe_text = {p["probe_id"]: p["probe_text"] for p in load_probes()}
    with runs_path.open(newline="", encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))

    total, done = len(rows), 0
    with ResumableWriter(out_path, JUDGE_FIELDS, key=("arm", "probe_id")) as writer:
        for row in rows:
            done += 1
            if writer.already_done(row["arm"], row["probe_id"]):
                continue
            verdict, rationale = adjudicate(
                row["family"], probe_text[row["probe_id"]], row["response"]
            )
            writer.write({
                "arm": row["arm"], "probe_id": row["probe_id"], "family": row["family"],
                "lexical_drifted": row["drifted"],
                "judge_verdict": verdict, "judge_rationale": rationale,
            })
            progress("judge", done, total, f"{row['arm']} {row['probe_id']} -> {verdict}")
