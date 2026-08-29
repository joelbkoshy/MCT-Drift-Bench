"""Analysis: descriptive tables, pre-specified contrasts, detector validation, figures.

Every arm answers the same probes and the same narratives, so all comparisons
are paired. Binary drift outcomes are tested with an exact McNemar test on the
discordant pairs; ordinal fidelity scores with a Wilcoxon signed-rank test and a
matched-pairs rank-biserial effect size. The seven contrasts below are fixed in
advance and the family of p-values is corrected with Holm's step-down procedure,
so the correction is not chosen after seeing which comparisons came out well.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

BASE_DIR = Path(__file__).resolve().parents[2]
RESULTS = BASE_DIR / "results"
FIGURES = BASE_DIR / "figures"

ARM_ORDER = ["none", "short", "long", "long+rag", "lora", "lora+rag", "llama3-long+rag"]

# Fixed before the runs. Each isolates one manipulation.
CONTRASTS = [
    ("C1", "none", "short", "specifying the therapy at all"),
    ("C2", "short", "long", "depth of the specification"),
    ("C3", "long", "long+rag", "retrieval, base weights"),
    ("C4", "short", "lora", "adaptation, prompt held constant"),
    ("C5", "lora", "lora+rag", "retrieval, adapted weights"),
    ("C6", "long+rag", "lora+rag", "best base configuration vs adapted"),
    ("C7", "long+rag", "llama3-long+rag", "model scale at fixed configuration"),
]


# --------------------------------------------------------------------------- #
# statistics
# --------------------------------------------------------------------------- #

def holm(pvalues: list[float]) -> list[float]:
    """Holm step-down adjusted p-values, order preserved."""
    n = len(pvalues)
    order = np.argsort(pvalues)
    adjusted = np.empty(n, dtype=float)
    running = 0.0
    for rank, idx in enumerate(order):
        value = (n - rank) * pvalues[idx]
        running = max(running, value)
        adjusted[idx] = min(running, 1.0)
    return adjusted.tolist()


def mcnemar_exact(a: np.ndarray, b: np.ndarray) -> tuple[int, int, float]:
    """Exact McNemar on paired binary outcomes. Returns (b01, b10, p)."""
    b01 = int(np.sum(~a & b))   # a held, b drifted
    b10 = int(np.sum(a & ~b))   # a drifted, b held
    n = b01 + b10
    if n == 0:
        return b01, b10, 1.0
    p = stats.binomtest(b01, n, 0.5).pvalue
    return b01, b10, float(p)


def wilcoxon_with_effect(x: np.ndarray, y: np.ndarray) -> tuple[float, float, float]:
    """Signed-rank test plus matched-pairs rank-biserial correlation."""
    diff = y - x
    nonzero = diff[diff != 0]
    if nonzero.size == 0:
        return float("nan"), 1.0, 0.0
    statistic, p = stats.wilcoxon(x, y, zero_method="wilcox", alternative="two-sided")
    ranks = stats.rankdata(np.abs(nonzero))
    positive = ranks[nonzero > 0].sum()
    negative = ranks[nonzero < 0].sum()
    total = positive + negative
    effect = (positive - negative) / total if total else 0.0
    return float(statistic), float(p), float(effect)


def cohen_kappa(a: list[int], b: list[int]) -> float:
    a_arr, b_arr = np.asarray(a), np.asarray(b)
    observed = float(np.mean(a_arr == b_arr))
    pa, pb = a_arr.mean(), b_arr.mean()
    expected = pa * pb + (1 - pa) * (1 - pb)
    return (observed - expected) / (1 - expected) if expected < 1 else float("nan")


def wilson_ci(successes: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return float("nan"), float("nan")
    p = successes / n
    denominator = 1 + z**2 / n
    centre = (p + z**2 / (2 * n)) / denominator
    half = z * np.sqrt(p * (1 - p) / n + z**2 / (4 * n**2)) / denominator
    return max(0.0, centre - half), min(1.0, centre + half)


# --------------------------------------------------------------------------- #
# loading
# --------------------------------------------------------------------------- #

def load_drift() -> pd.DataFrame:
    df = pd.read_csv(RESULTS / "drift_runs.csv")
    df["drifted"] = df["drifted"].astype(str).str.lower().eq("true")
    df["held"] = df["held"].astype(str).str.lower().eq("true")
    df["refused"] = df["refused"].astype(str).str.lower().eq("true")
    df["constraint_compliant"] = df["constraint_compliant"].astype(str).str.lower().eq("true")
    return df


def load_fidelity() -> pd.DataFrame:
    df = pd.read_csv(RESULTS / "fidelity_runs.csv")
    for col in ("constraint_compliant", "refused"):
        df[col] = df[col].astype(str).str.lower().eq("true")
    for col in df.columns:
        if col.startswith("ind_"):
            df[col] = df[col].astype(str).str.lower().eq("true")
    return df


def load_judgements() -> pd.DataFrame | None:
    path = RESULTS / "drift_judgements.csv"
    if not path.exists():
        return None
    return pd.read_csv(path)


# --------------------------------------------------------------------------- #
# tables
# --------------------------------------------------------------------------- #

def drift_by_arm(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for arm in [a for a in ARM_ORDER if a in set(df["arm"])]:
        sub = df[df["arm"] == arm]
        resisted = int((~sub["drifted"]).sum())
        low, high = wilson_ci(resisted, len(sub))
        rows.append({
            "arm": arm,
            "n": len(sub),
            "drs": resisted / len(sub),
            "ci_low": low,
            "ci_high": high,
            "held_rate": sub["held"].mean(),
            "constraint_compliance": sub["constraint_compliant"].mean(),
            "fidelity": sub["fidelity_overall"].mean(),
            "refusal_rate": sub["refused"].mean(),
            "latency_s": sub["latency_s"].median(),
            "words": sub["n_words"].median(),
        })
    return pd.DataFrame(rows)


def drift_by_arm_family(df: pd.DataFrame) -> pd.DataFrame:
    table = (
        df.assign(resisted=~df["drifted"])
        .pivot_table(index="arm", columns="family", values="resisted", aggfunc="mean")
    )
    return table.reindex([a for a in ARM_ORDER if a in table.index])


def fidelity_by_arm(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for arm in [a for a in ARM_ORDER if a in set(df["arm"])]:
        sub = df[df["arm"] == arm]
        row = {
            "arm": arm,
            "n": len(sub),
            "fidelity_mean": sub["fidelity_overall"].mean(),
            "fidelity_sd": sub["fidelity_overall"].std(ddof=1),
            "constraint_compliance": sub["constraint_compliant"].mean(),
            "excellent_rate": (sub["fidelity_grade"] == "excellent").mean(),
            "refusal_rate": sub["refused"].mean(),
            "latency_s": sub["latency_s"].median(),
            "words": sub["n_words"].median(),
        }
        for col in sub.columns:
            if col.startswith("ind_"):
                row[col] = sub[col].mean()
        rows.append(row)
    return pd.DataFrame(rows)


def contrast_table(drift: pd.DataFrame, fidelity: pd.DataFrame) -> pd.DataFrame:
    drift_wide = drift.pivot(index="probe_id", columns="arm", values="drifted")
    fid_wide = fidelity.pivot(index="narrative_id", columns="arm", values="fidelity_overall")

    rows, p_drift, p_fid = [], [], []
    for code, a, b in [(c[0], c[1], c[2]) for c in CONTRASTS]:
        label = next(c[3] for c in CONTRASTS if c[0] == code)
        if a not in drift_wide or b not in drift_wide:
            continue
        da, db = drift_wide[a].to_numpy(bool), drift_wide[b].to_numpy(bool)
        b01, b10, pd_ = mcnemar_exact(da, db)
        fa, fb = fid_wide[a].to_numpy(float), fid_wide[b].to_numpy(float)
        _, pf, effect = wilcoxon_with_effect(fa, fb)

        rows.append({
            "contrast": code, "from": a, "to": b, "isolates": label,
            "drs_from": 1 - da.mean(), "drs_to": 1 - db.mean(),
            "drs_delta": da.mean() - db.mean(),
            "discordant_worse": b01, "discordant_better": b10, "p_drift": pd_,
            "fid_from": fa.mean(), "fid_to": fb.mean(),
            "fid_delta": fb.mean() - fa.mean(),
            "rank_biserial": effect, "p_fidelity": pf,
        })
        p_drift.append(pd_)
        p_fid.append(pf)

    out = pd.DataFrame(rows)
    out["p_drift_holm"] = holm(p_drift)
    out["p_fidelity_holm"] = holm(p_fid)
    return out


def detector_validation(drift: pd.DataFrame, judged: pd.DataFrame) -> dict:
    merged = drift.merge(judged, on=["arm", "probe_id"], suffixes=("", "_j"))
    merged = merged[merged["judge_verdict"].isin(["violated", "held"])]
    lex = merged["drifted"].astype(int).tolist()
    judge = (merged["judge_verdict"] == "violated").astype(int).tolist()

    both = sum(l and j for l, j in zip(lex, judge))
    lex_only = sum(l and not j for l, j in zip(lex, judge))
    judge_only = sum((not l) and j for l, j in zip(lex, judge))
    neither = sum((not l) and (not j) for l, j in zip(lex, judge))

    return {
        "n_adjudicated": len(merged),
        "n_unparsed": int((judged["judge_verdict"] == "unparsed").sum()),
        "agreement": (both + neither) / len(merged) if len(merged) else float("nan"),
        "kappa": cohen_kappa(lex, judge),
        "both_flag": both,
        "lexical_only": lex_only,
        "judge_only": judge_only,
        "neither": neither,
        "lexical_drift_rate": float(np.mean(lex)),
        "judge_drift_rate": float(np.mean(judge)),
    }


# --------------------------------------------------------------------------- #
# figures
# --------------------------------------------------------------------------- #

def make_figures(drift: pd.DataFrame, fidelity: pd.DataFrame) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    FIGURES.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"font.size": 8, "figure.dpi": 300, "savefig.bbox": "tight"})

    heat = drift_by_arm_family(drift)
    fig, ax = plt.subplots(figsize=(6.4, 2.6))
    im = ax.imshow(heat.to_numpy(), cmap="RdYlGn", vmin=0, vmax=1, aspect="auto")
    ax.set_xticks(range(len(heat.columns)), heat.columns)
    ax.set_yticks(range(len(heat.index)), heat.index)
    for i in range(heat.shape[0]):
        for j in range(heat.shape[1]):
            value = heat.to_numpy()[i, j]
            ax.text(j, i, f"{value:.2f}", ha="center", va="center", fontsize=6.5,
                    color="black" if 0.25 < value < 0.85 else "white")
    ax.set_xlabel("drift family")
    ax.set_ylabel("")
    fig.colorbar(im, ax=ax, label="drift resistance", fraction=0.025)
    fig.savefig(FIGURES / "drs_heatmap.pdf")
    fig.savefig(FIGURES / "drs_heatmap.png")
    plt.close(fig)

    summary = drift_by_arm(drift).merge(
        fidelity_by_arm(fidelity)[["arm", "fidelity_mean"]], on="arm"
    )
    fig, ax = plt.subplots(figsize=(3.4, 2.6))
    ax.scatter(summary["fidelity_mean"], summary["drs"], s=28, zorder=3)
    for _, r in summary.iterrows():
        ax.annotate(r["arm"], (r["fidelity_mean"], r["drs"]), fontsize=6,
                    xytext=(3, 3), textcoords="offset points")
    ax.set_xlabel("MCT fidelity on unbaited narratives (1-5)")
    ax.set_ylabel("drift resistance under bait")
    ax.grid(alpha=0.3, zorder=0)
    fig.savefig(FIGURES / "fidelity_vs_drs.pdf")
    fig.savefig(FIGURES / "fidelity_vs_drs.png")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(3.4, 2.6))
    order = [a for a in ARM_ORDER if a in set(summary["arm"])]
    values = np.array([summary.loc[summary["arm"] == a, "drs"].iloc[0] for a in order])
    lows = np.array([summary.loc[summary["arm"] == a, "ci_low"].iloc[0] for a in order])
    highs = np.array([summary.loc[summary["arm"] == a, "ci_high"].iloc[0] for a in order])
    # At a boundary proportion the Wilson bound can land a float epsilon inside
    # the point estimate, which matplotlib rejects as a negative error bar.
    err = np.clip(np.vstack([values - lows, highs - values]), 0.0, None)
    ax.bar(range(len(order)), values, yerr=err, capsize=3, color="#4c72b0")
    ax.set_xticks(range(len(order)), order, rotation=35, ha="right")
    ax.set_ylabel("drift resistance (95% CI)")
    ax.set_ylim(0, 1)
    ax.grid(axis="y", alpha=0.3)
    fig.savefig(FIGURES / "drs_by_arm.pdf")
    fig.savefig(FIGURES / "drs_by_arm.png")
    plt.close(fig)


# --------------------------------------------------------------------------- #
# entry point
# --------------------------------------------------------------------------- #

def run() -> dict:
    drift = load_drift()
    fidelity = load_fidelity()
    judged = load_judgements()

    tables = {
        "drift_by_arm": drift_by_arm(drift),
        "drift_by_arm_family": drift_by_arm_family(drift).reset_index(),
        "fidelity_by_arm": fidelity_by_arm(fidelity),
        "contrasts": contrast_table(drift, fidelity),
    }
    for name, table in tables.items():
        table.to_csv(RESULTS / f"table_{name}.csv", index=False)

    summary = {name: table.to_dict(orient="records") for name, table in tables.items()}
    if judged is not None:
        summary["detector_validation"] = detector_validation(drift, judged)

    make_figures(drift, fidelity)
    (RESULTS / "summary.json").write_text(
        json.dumps(summary, indent=2, default=float), encoding="utf-8"
    )
    return summary
