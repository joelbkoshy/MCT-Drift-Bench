"""Probe construction.

A probe is a real narrative followed by one authored bait sentence.

The bait has to be authored: it is a controlled experimental manipulation, and
there is no ethical way to harvest genuine attempts to talk a therapy system out
of its therapeutic model. The narrative it is attached to is not authored. Each
bait is carried by a human-written first-person account drawn from the public
corpus used in the earlier phases, so the model always sees real clinical
language and the manipulation is the only constructed element.

Carriers are stratified across the four CAS categories and rotated across
families, so every family is probed against the same carrier pool and no family
is advantaged by an easier set of narratives. Selection is seeded and excludes
every narrative that entered the fine-tuning set.
"""

from __future__ import annotations

import hashlib
import json
import random
from pathlib import Path

from src.bench.corpus import SEED, stratified_sample
from src.bench.taxonomy import FAMILIES

BASE_DIR = Path(__file__).resolve().parents[2]
OUT_PATH = BASE_DIR / "data" / "probes" / "driftbench_v1.jsonl"

CARRIERS_PER_CATEGORY = 3


def _assign_carriers(n_families: int, n_baits: int, n_carriers: int) -> list[list[int]]:
    """Balanced carrier assignment: every carrier used equally often overall, and
    no carrier repeated within a family.

    A simple modular rotation looks balanced but is not: with six baits and
    twelve carriers it hands even-numbered families one half of the pool and
    odd-numbered families the other, so family and carrier become confounded and
    a family could look hard only because its narratives were.
    """
    total = n_families * n_baits
    if total % n_carriers:
        raise ValueError("carrier pool must divide the probe count evenly")

    slots = list(range(n_carriers)) * (total // n_carriers)
    rng = random.Random(SEED)
    for _ in range(10_000):
        rng.shuffle(slots)
        blocks = [slots[f * n_baits:(f + 1) * n_baits] for f in range(n_families)]
        if all(len(set(b)) == n_baits for b in blocks):
            return blocks
    raise RuntimeError("could not find a balanced carrier assignment")


def build(out_path: Path = OUT_PATH) -> list[dict]:
    carriers = stratified_sample(CARRIERS_PER_CATEGORY)
    blocks = _assign_carriers(len(FAMILIES), len(FAMILIES[0].baits), len(carriers))
    probes: list[dict] = []

    for fi, family in enumerate(FAMILIES):
        for bi, bait in enumerate(family.baits):
            carrier = carriers[blocks[fi][bi]]
            probes.append(
                {
                    "probe_id": f"{family.code}-{bi + 1:02d}",
                    "family": family.code,
                    "family_name": family.name,
                    "axis": family.axis,
                    "bait": bait,
                    "carrier_id": carrier["narrative_id"],
                    "carrier_category": carrier["category"],
                    "carrier_text": carrier["text"],
                    "probe_text": f"{carrier['text']}\n\n{bait}",
                    # The narrative is prior context and the bait is the turn
                    # being answered. Run together as one block the bait reads as
                    # one more clause of a hundred-word account, and a small model
                    # answers the account instead of the request -- which measures
                    # salience rather than adherence.
                    "patient_block": (
                        f'PATIENT (earlier in this session):\n"{carrier["text"]}"\n\n'
                        f'PATIENT (current message):\n"{bait}"'
                    ),
                }
            )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as fh:
        for p in probes:
            fh.write(json.dumps(p, ensure_ascii=False) + "\n")
    return probes


def carrier_ids(path: Path = OUT_PATH) -> set[int]:
    return {p["carrier_id"] for p in load(path)}


def probe_hash(path: Path = OUT_PATH) -> str:
    """Fingerprint of the probe set, so a resumed run can prove it is appending
    to generations produced against the same probes."""
    if not path.exists():
        build(path)
    return hashlib.sha256(path.read_bytes()).hexdigest()[:16]


def load(path: Path = OUT_PATH) -> list[dict]:
    if not path.exists():
        return build(path)
    with path.open(encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


if __name__ == "__main__":
    built = build()
    print(f"wrote {len(built)} probes to {OUT_PATH}")
