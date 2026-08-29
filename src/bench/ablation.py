"""Fidelity axis: what does each intervention buy on ordinary clinical input?

The drift axis measures behaviour under pressure. This axis measures behaviour
without any: each arm answers unbaited, real narratives, and the outputs are
scored with the MCT adherence instruments carried over from the earlier phases.

Read together, the two axes separate two things that are usually conflated. An
intervention can raise average adherence on cooperative input while leaving the
system just as easy to talk out of the therapy, and that distinction only
becomes visible when both are measured on the same arms.
"""

from __future__ import annotations

from pathlib import Path

from src.bench import rag
from src.bench.backend import generate, installed_models, is_refusal
from src.bench.build_probes import carrier_ids
from src.bench.conditions import ARMS, REFERENCE_ARM, build_user_message
from src.bench.corpus import stratified_sample
from src.bench.runner import ResumableWriter, progress
from src.core.constraint_filter import check_constraints, score_mct_indicators
from src.core.fidelity_scorer import compute_fidelity

BASE_DIR = Path(__file__).resolve().parents[2]
OUT_PATH = BASE_DIR / "results" / "fidelity_runs.csv"

NARRATIVES_PER_CATEGORY = 10

FIELDS = [
    "arm", "model", "weights", "specification", "rag",
    "narrative_id", "category",
    "fidelity_overall", "fidelity_grade",
    "item_avoids_why_questions", "item_no_content_elaboration",
    "item_process_over_content", "item_promotes_mct_techniques",
    "item_no_rumination_validation",
    "constraint_compliant", "violation_count", "violations",
    "ind_process_labeling", "ind_detached_mindfulness", "ind_att_reference",
    "ind_postponement", "ind_metacognitive_question",
    "refused", "latency_s", "n_words", "response",
]


def evaluation_set() -> list[dict]:
    """Narratives disjoint from both the fine-tuning set and the probe carriers."""
    return stratified_sample(NARRATIVES_PER_CATEGORY, exclude_ids=carrier_ids())


def run(include_reference: bool = True, out_path: Path = OUT_PATH) -> None:
    narratives = evaluation_set()
    arms = list(ARMS) + ([REFERENCE_ARM] if include_reference else [])

    missing = {a.model for a in arms} - installed_models()
    if missing:
        raise RuntimeError(f"models not installed in Ollama: {sorted(missing)}")

    total = len(arms) * len(narratives)
    done = 0
    with ResumableWriter(out_path, FIELDS, key=("arm", "narrative_id")) as writer:
        for arm in arms:
            for item in narratives:
                done += 1
                nid = item["narrative_id"]
                if writer.already_done(arm.name, nid):
                    continue

                context = rag.retrieve(item["text"]) if arm.use_rag else None
                user = build_user_message(f'PATIENT INPUT:\n"{item["text"]}"', context)
                try:
                    text, latency = generate(arm.system_prompt, user, arm.model)
                except Exception as exc:
                    text, latency = f"[GENERATION ERROR: {exc}]", 0.0

                fidelity = compute_fidelity(text)
                constraint = check_constraints(text)
                indicators = score_mct_indicators(text)

                writer.write({
                    "arm": arm.name, "model": arm.model, "weights": arm.weights,
                    "specification": arm.specification, "rag": arm.use_rag,
                    "narrative_id": nid, "category": item["category"],
                    "fidelity_overall": fidelity.overall_score,
                    "fidelity_grade": fidelity.grade,
                    **{f"item_{k}": v for k, v in fidelity.item_scores.items()},
                    "constraint_compliant": constraint.is_compliant,
                    "violation_count": constraint.violation_count,
                    "violations": "; ".join(constraint.violations),
                    **{f"ind_{k}": v for k, v in indicators.items()},
                    "refused": is_refusal(text),
                    "latency_s": round(latency, 2),
                    "n_words": len(text.split()),
                    "response": text,
                })
                progress("fidelity", done, total, f"{arm.name} n{nid} {latency:.1f}s")
