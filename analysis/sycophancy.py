"""
Sycophancy deep-dive analysis.

Investigates:
  S1  — sycophancy rate by model & tier
  S2  — sycophancy rate by theme
  S3  — sycophancy rate by input length & scenario source
  S4  — THE PARADOX: sycophantic turns vs non-sycophantic turns on harm/help scores
  S5  — sycophancy × over_solving co-occurrence heatmap
  S6  — sycophancy rate by severity tier (does escalation change sycophancy?)
  S7  — sycophantic_accommodation sub-score distribution by model

Usage:
  python analysis/sycophancy.py
  python analysis/sycophancy.py --output analysis/figures/sycophancy
"""

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats

sns.set_theme(style="whitegrid", font_scale=1.05)
PALETTE = sns.color_palette("Set2")

SEVERITY_ORDER = ["baseline", "medium", "stress_test"]

# ── Load ───────────────────────────────────────────────────────────────────────

def load_all(eval_dir: Path) -> pd.DataFrame:
    records = []
    for path in sorted(eval_dir.rglob("eval_*.json")):
        if any(part.startswith("_") for part in path.parts):
            continue

        parts = path.parts
        if "human_checked" in str(path):
            scenario_src, input_len = "human_checked", "short"
        elif "long_input" in str(path):
            scenario_src, input_len = "llm_generated", "long"
        else:
            scenario_src, input_len = "llm_generated", "short"
        model_tier = "dev" if "dev_models" in str(path) else "prod"

        with open(path, encoding="utf-8") as f:
            d = json.load(f)

        model_name = d.get("model_name", "unknown")
        theme      = d.get("theme", "unknown")

        for t in d.get("turns", []):
            m    = t.get("evaluation_metrics") or {}
            tags = m.get("error_tags", [])

            def sc(key):
                v = m.get(key, {})
                s = v.get("score") if isinstance(v, dict) else v
                if s in ("NA", None):
                    return np.nan
                try:
                    return float(s)
                except (ValueError, TypeError):
                    return np.nan

            h1   = sc("response_attunement")
            h2   = sc("sycophantic_accommodation")
            ev   = sc("emotional_validation")
            as_  = sc("actionable_support")
            cr   = sc("cognitive_reframing")
            harm = h1 + h2 if not (np.isnan(h1) or np.isnan(h2)) else np.nan
            help_s = np.nanmean([ev, as_, cr]) if not all(np.isnan([ev, as_, cr])) else np.nan

            records.append({
                "model_name":               model_name,
                "model_tier":               model_tier,
                "theme":                    theme,
                "scenario_src":             scenario_src,
                "input_len":                input_len,
                "severity_tier":            t.get("severity_tier"),
                "sycophancy":               "sycophancy" in tags,
                "over_solving":             "over_solving" in tags,
                "over_medicalization":      "over_medicalization" in tags,
                "harm_score":               harm,
                "sycophantic_accommodation": h2,
                "help_score":               help_s,
                "emotional_validation":     ev,
                "actionable_support":       as_,
                "cognitive_reframing":      cr,
                "response_words":           len(t.get("model_response", "").split()),
            })

    df = pd.DataFrame(records)
    df["severity_tier"] = pd.Categorical(
        df["severity_tier"], categories=SEVERITY_ORDER, ordered=True
    )
    return df


def _save(fig, out_dir: Path, name: str):
    out_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_dir / name, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {name}")


MODEL_ORDER_DEV  = ["GPT-5.4 Nano", "Gemini 2.5 Flash Lite", "Mistral Small 3.2"]
MODEL_ORDER_PROD = ["Claude Haiku 4.5", "DeepSeek V3.2", "GPT-5.4 Mini", "Gemini 3 Flash Preview"]
ALL_MODELS = MODEL_ORDER_DEV + MODEL_ORDER_PROD


# ── S1: Sycophancy rate by model & tier ───────────────────────────────────────

def plot_s1_by_model(df: pd.DataFrame, out_dir: Path):
    agg = df.groupby(["model_name", "model_tier"])["sycophancy"].mean().reset_index()
    agg["pct"] = agg["sycophancy"] * 100

    models_present = [m for m in ALL_MODELS if m in agg["model_name"].values]
    agg = agg.set_index("model_name").reindex(models_present).reset_index()

    tier_colors = [PALETTE[0] if t == "dev" else PALETTE[1]
                   for t in agg["model_tier"]]

    fig, ax = plt.subplots(figsize=(10, 5))
    bars = ax.bar(agg["model_name"], agg["pct"], color=tier_colors,
                  edgecolor="white", linewidth=1.2)

    for bar, val in zip(bars, agg["pct"]):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                f"{val:.1f}%", ha="center", va="bottom", fontsize=9)

    ax.axhline(df["sycophancy"].mean() * 100, color="gray",
               linestyle="--", linewidth=1, label=f"Overall mean ({df['sycophancy'].mean()*100:.1f}%)")
    ax.set_ylabel("Sycophancy Rate (%)", fontsize=11)
    ax.set_title("Sycophancy Rate by Model", fontsize=13, fontweight="bold")
    ax.tick_params(axis="x", rotation=20, labelsize=9)
    ax.set_ylim(0, 100)

    patches = [
        mpatches.Patch(color=PALETTE[0], label="Dev"),
        mpatches.Patch(color=PALETTE[1], label="Prod"),
    ]
    ax.legend(handles=patches + [plt.Line2D([0], [0], color="gray",
              linestyle="--", label=f"Overall mean ({df['sycophancy'].mean()*100:.1f}%)")],
              fontsize=9)

    plt.tight_layout()
    _save(fig, out_dir, "S1_sycophancy_by_model.png")


# ── S2: Sycophancy rate by theme ──────────────────────────────────────────────

def plot_s2_by_theme(df: pd.DataFrame, out_dir: Path):
    agg = df.groupby(["theme", "model_tier"])["sycophancy"].mean().unstack("model_tier") * 100
    agg = agg.sort_values("prod", ascending=False) if "prod" in agg.columns else agg.sort_values(agg.columns[0], ascending=False)

    fig, ax = plt.subplots(figsize=(11, 5))
    x = np.arange(len(agg))
    w = 0.35

    if "dev" in agg.columns:
        ax.bar(x - w/2, agg["dev"], w, color=PALETTE[0], label="Dev", edgecolor="white")
    if "prod" in agg.columns:
        ax.bar(x + w/2, agg["prod"], w, color=PALETTE[1], label="Prod", edgecolor="white")

    ax.set_xticks(x)
    ax.set_xticklabels(agg.index, rotation=20, ha="right", fontsize=9)
    ax.set_ylabel("Sycophancy Rate (%)", fontsize=11)
    ax.set_title("Sycophancy Rate by Theme", fontsize=13, fontweight="bold")
    ax.set_ylim(0, 100)
    ax.legend(fontsize=9)

    plt.tight_layout()
    _save(fig, out_dir, "S2_sycophancy_by_theme.png")


# ── S3: Sycophancy by input length & source ───────────────────────────────────

def plot_s3_by_input(df: pd.DataFrame, out_dir: Path):
    groups = [
        ("human_checked", "short", "HC · short"),
        ("llm_generated",  "short", "Gen · short"),
        ("llm_generated",  "long",  "Gen · long"),
    ]
    fig, ax = plt.subplots(figsize=(9, 5))
    x = np.arange(len(groups))
    w = 0.35

    for tier_i, (tier, color) in enumerate([("dev", PALETTE[0]), ("prod", PALETTE[1])]):
        vals = []
        for src, ilen, _ in groups:
            sub = df[(df["scenario_src"] == src) & (df["input_len"] == ilen) & (df["model_tier"] == tier)]
            vals.append(sub["sycophancy"].mean() * 100 if len(sub) else np.nan)
        offset = -w/2 if tier == "dev" else w/2
        bars = ax.bar(x + offset, vals, w, color=color, label=tier.capitalize(),
                      edgecolor="white", linewidth=1.2)
        for bar, val in zip(bars, vals):
            if not np.isnan(val):
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                        f"{val:.1f}%", ha="center", va="bottom", fontsize=8)

    ax.set_xticks(x)
    ax.set_xticklabels([g[2] for g in groups], fontsize=10)
    ax.set_ylabel("Sycophancy Rate (%)", fontsize=11)
    ax.set_title("Sycophancy Rate by Input Length & Source", fontsize=13, fontweight="bold")
    ax.set_ylim(0, 100)
    ax.legend(fontsize=9)

    plt.tight_layout()
    _save(fig, out_dir, "S3_sycophancy_by_input.png")


# ── S4: THE PARADOX ───────────────────────────────────────────────────────────

def plot_s4_paradox(df: pd.DataFrame, out_dir: Path):
    """
    Core finding: sycophantic turns vs non-sycophantic turns.
    Compare harm_score AND help_score split by sycophancy flag, per model.
    """
    models = [m for m in ALL_MODELS if m in df["model_name"].unique()]
    x = np.arange(len(models))
    w = 0.35

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle(
        "The Sycophancy Paradox: Do Sycophantic Responses Appear More Helpful?",
        fontsize=12, fontweight="bold"
    )

    syco_color    = "#e41a1c"
    nonsyco_color = "#377eb8"

    for ax, col, title, ylim in [
        (axes[0], "harm_score", "Harm Score (lower = better, 0–4)", (0, 4)),
        (axes[1], "help_score", "Help Score (higher = better, 1–5)", (0, 5)),
    ]:
        syco_vals, non_vals = [], []
        syco_errs, non_errs = [], []

        for m in models:
            sub = df[df["model_name"] == m]
            s  = sub[sub["sycophancy"]][col].dropna()
            ns = sub[~sub["sycophancy"]][col].dropna()
            syco_vals.append(s.mean()  if len(s)  else np.nan)
            non_vals.append(ns.mean()  if len(ns) else np.nan)
            syco_errs.append(s.sem()   if len(s)  else 0)
            non_errs.append(ns.sem()   if len(ns) else 0)

        b1 = ax.bar(x - w/2, syco_vals, w, yerr=syco_errs,
                    color=syco_color,    label="Sycophantic", capsize=4,
                    edgecolor="white", linewidth=1.2, alpha=0.85)
        b2 = ax.bar(x + w/2, non_vals,  w, yerr=non_errs,
                    color=nonsyco_color, label="Non-sycophantic", capsize=4,
                    edgecolor="white", linewidth=1.2, alpha=0.85)

        for bar, val in zip(list(b1) + list(b2), syco_vals + non_vals):
            if not np.isnan(val):
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.04,
                        f"{val:.2f}", ha="center", va="bottom", fontsize=7.5)

        ax.set_xticks(x)
        ax.set_xticklabels(models, rotation=18, ha="right", fontsize=8)
        ax.set_title(title, fontsize=10)
        ax.set_ylim(*ylim)
        ax.legend(fontsize=9)

    # Annotation box
    axes[1].text(0.98, 0.04,
                 "If sycophantic turns show HIGHER help scores\n"
                 "but also HIGHER harm scores → safety paradox",
                 transform=axes[1].transAxes, fontsize=8,
                 ha="right", va="bottom",
                 bbox=dict(boxstyle="round,pad=0.4", facecolor="#fff9c4", alpha=0.8))

    plt.tight_layout()
    _save(fig, out_dir, "S4_sycophancy_paradox.png")

    # Stats summary
    print("\n=== S4 Paradox Stats (all models pooled) ===")
    for col, label in [("harm_score", "Harm"), ("help_score", "Help")]:
        s  = df[df["sycophancy"]][col].dropna()
        ns = df[~df["sycophancy"]][col].dropna()
        t_stat, p_val = stats.ttest_ind(s, ns)
        print(f"{label}: syco={s.mean():.3f}  non-syco={ns.mean():.3f}  "
              f"Δ={s.mean()-ns.mean():.3f}  p={p_val:.2e}")


# ── S5: Co-occurrence heatmap ─────────────────────────────────────────────────

def plot_s5_cooccurrence(df: pd.DataFrame, out_dir: Path):
    error_cols = ["sycophancy", "over_solving", "over_medicalization"]
    labels     = ["Sycophancy", "Over-Solving", "Over-Medicalization"]

    # Per-model stacked bar: fraction with each tag combination
    models = [m for m in ALL_MODELS if m in df["model_name"].unique()]
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("Error Tag Co-occurrence & Distribution", fontsize=13, fontweight="bold")

    # Left: stacked bar of tag rates per model
    ax = axes[0]
    colors = ["#e41a1c", "#ff7f00", "#4daf4a"]
    bottom = np.zeros(len(models))
    for col, label, color in zip(error_cols, labels, colors):
        vals = [df[df["model_name"] == m][col].mean() * 100 for m in models]
        ax.bar(models, vals, bottom=bottom, label=label, color=color,
               edgecolor="white", linewidth=0.8, alpha=0.85)
        bottom += np.array(vals)
    ax.set_ylabel("% of turns", fontsize=10)
    ax.set_title("Error Tag Rates per Model", fontsize=10)
    ax.tick_params(axis="x", rotation=20, labelsize=8)
    ax.set_ylim(0, 110)
    ax.legend(fontsize=8)

    # Right: co-occurrence matrix (% of turns with BOTH tags)
    ax = axes[1]
    n = len(error_cols)
    matrix = np.zeros((n, n))
    for i, c1 in enumerate(error_cols):
        for j, c2 in enumerate(error_cols):
            if i == j:
                matrix[i][j] = df[c1].mean() * 100
            else:
                matrix[i][j] = (df[c1] & df[c2]).mean() * 100

    sns.heatmap(matrix, ax=ax, annot=True, fmt=".1f",
                xticklabels=labels, yticklabels=labels,
                cmap="YlOrRd", vmin=0, vmax=70,
                linewidths=0.5, cbar_kws={"label": "% of turns"})
    ax.set_title("Co-occurrence Matrix (%)", fontsize=10)
    ax.tick_params(axis="x", rotation=20, labelsize=8)
    ax.tick_params(axis="y", rotation=0,  labelsize=8)

    plt.tight_layout()
    _save(fig, out_dir, "S5_error_cooccurrence.png")


# ── S6: Sycophancy by severity tier ───────────────────────────────────────────

def plot_s6_severity(df: pd.DataFrame, out_dir: Path):
    valid = df[df["severity_tier"].isin(SEVERITY_ORDER)].copy()
    if valid.empty:
        print("  No severity data — skipping S6.")
        return

    models = [m for m in ALL_MODELS if m in df["model_name"].unique()]
    colors = {m: PALETTE[i % len(PALETTE)] for i, m in enumerate(models)}

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle("Sycophancy Rate Across Severity Tiers", fontsize=13, fontweight="bold")

    tier_labels = {"baseline": "Baseline", "medium": "Medium", "stress_test": "Stress Test"}

    for ax, tier_group, title in [
        (axes[0], "dev",  "Dev Models"),
        (axes[1], "prod", "Prod Models"),
    ]:
        tier_models = [m for m in (MODEL_ORDER_DEV if tier_group == "dev" else MODEL_ORDER_PROD)
                       if m in df["model_name"].unique()]
        for model in tier_models:
            sub = valid[valid["model_name"] == model]
            grp = sub.groupby("severity_tier", observed=True)["sycophancy"].mean() * 100
            x_labels = [tier_labels.get(t, t) for t in grp.index]
            ax.plot(x_labels, grp.values, marker="o", label=model,
                    color=colors[model], linewidth=2, markersize=7)

        ax.set_xlabel("Severity Tier", fontsize=10)
        ax.set_ylabel("Sycophancy Rate (%)", fontsize=10)
        ax.set_title(title, fontsize=10)
        ax.set_ylim(0, 100)
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    _save(fig, out_dir, "S6_sycophancy_by_severity.png")


# ── S7: Sycophantic_accommodation sub-score distribution ──────────────────────

def plot_s7_subscore(df: pd.DataFrame, out_dir: Path):
    """Distribution of the sycophantic_accommodation raw score (0-4) per model."""
    models  = [m for m in ALL_MODELS if m in df["model_name"].unique()]
    palette = {m: PALETTE[i % len(PALETTE)] for i, m in enumerate(ALL_MODELS)}

    fig, ax = plt.subplots(figsize=(11, 5))
    sub = df[["model_name", "sycophantic_accommodation"]].dropna()
    sub = sub[sub["model_name"].isin(models)]

    sns.boxplot(data=sub, x="model_name", y="sycophantic_accommodation",
                hue="model_name", order=models, palette=palette,
                ax=ax, width=0.5, linewidth=1.2, legend=False)

    ax.set_title("Sycophantic Accommodation Sub-score Distribution (0–4)",
                 fontsize=12, fontweight="bold")
    ax.set_xlabel("")
    ax.set_ylabel("Score (0 = none, 4 = severe)", fontsize=10)
    ax.tick_params(axis="x", rotation=20, labelsize=9)

    # Annotate means
    means = sub.groupby("model_name")["sycophantic_accommodation"].mean()
    for i, m in enumerate(models):
        if m in means.index:
            ax.text(i, means[m] + 0.05, f"{means[m]:.2f}",
                    ha="center", va="bottom", fontsize=8, color="black")

    plt.tight_layout()
    _save(fig, out_dir, "S7_syco_accommodation_dist.png")


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--eval-dir", default="data/eval_results")
    parser.add_argument("--output",   default="analysis/figures/sycophancy")
    args = parser.parse_args()

    eval_dir = Path(args.eval_dir)
    out_dir  = Path(args.output)

    print(f"Loading data from {eval_dir} ...")
    df = load_all(eval_dir)
    print(f"Loaded {len(df)} turns | sycophancy rate: {df['sycophancy'].mean()*100:.1f}%\n")

    print("Generating sycophancy figures...")
    plot_s1_by_model(df, out_dir)
    plot_s2_by_theme(df, out_dir)
    plot_s3_by_input(df, out_dir)
    plot_s4_paradox(df, out_dir)
    plot_s5_cooccurrence(df, out_dir)
    plot_s6_severity(df, out_dir)
    plot_s7_subscore(df, out_dir)

    print(f"\nDone. Saved to: {out_dir}")


if __name__ == "__main__":
    main()
