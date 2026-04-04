"""
Data analysis and visualization for LLM mental health evaluation results.

Reads all eval result files from data/eval_results/ and generates:
  01_composite_scores.png       — bar chart: harm / help / AI quality per model
  02_safety_empathy_tradeoff.png — scatter: harm vs help with quadrant labels
  03_radar_model_profiles.png   — radar chart: all dimensions normalized
  04_error_tags.png             — error tag frequency by model
  05_scores_by_escalation.png   — line chart: scores across escalation levels
  06_metric_distributions.png   — box plots: per-metric distributions
  summary_table.csv             — aggregate stats per model

Usage:
  python analysis/analyze.py
  python analysis/analyze.py --eval-dir data/eval_results --output analysis/figures
"""

import argparse
import json
from collections import Counter
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

sns.set_theme(style="whitegrid", font_scale=1.05)
PALETTE = sns.color_palette("Set2")


# ── Data loading ───────────────────────────────────────────────────────────────

def load_eval_results(eval_dir: Path) -> pd.DataFrame:
    """Load all eval_*.json files and flatten to one row per turn."""
    records = []

    for path in sorted(eval_dir.rglob("eval_*.json")):
        with open(path, encoding="utf-8") as f:
            data = json.load(f)

        model_name = data.get("model_name", "unknown")
        stimulus_id = data.get("stimulus_id", "unknown")

        for turn in data.get("turns", []):
            metrics = turn.get("evaluation_metrics", {})

            def score(key):
                m = metrics.get(key, {})
                if isinstance(m, dict):
                    s = m.get("score")
                    if s == "NA" or s is None:
                        return np.nan
                    try:
                        return float(s)
                    except (ValueError, TypeError):
                        return np.nan
                return np.nan

            ref_raw = None
            ref_field = metrics.get("professional_referral", {})
            if isinstance(ref_field, dict):
                ref_raw = ref_field.get("score")

            ei = score("emotional_invalidation")
            mr = score("maladaptive_reinforcement")
            harm = ei + mr if not (np.isnan(ei) or np.isnan(mr)) else np.nan

            records.append({
                "stimulus_id": stimulus_id,
                "model_name": model_name,
                "turn": turn.get("turn"),
                "escalation_level": turn.get("escalation_level"),
                "expected_risk": turn.get("expected_risk"),
                "judge_status": turn.get("judge_status"),
                # Harm
                "emotional_invalidation": ei,
                "maladaptive_reinforcement": mr,
                "harm_score": harm,
                # Help
                "emotional_validation": score("emotional_validation"),
                "actionable_support": score("actionable_support"),
                "cognitive_reframing": score("cognitive_reframing"),
                # AI Quality
                "boundary_appropriateness": score("boundary_appropriateness"),
                "professional_referral_raw": ref_raw,
                # Error tags
                "error_tags": metrics.get("error_tags", []),
            })

    if not records:
        return pd.DataFrame()

    df = pd.DataFrame(records)
    df["help_score"] = df[["emotional_validation", "actionable_support", "cognitive_reframing"]].mean(axis=1)
    df["any_harm"] = df["harm_score"].fillna(0) > 0
    df["referral_correct"] = df["professional_referral_raw"].apply(
        lambda x: str(x) in ("2", "NA") if pd.notna(x) else False
    )
    return df


def _model_colors(models):
    return {m: PALETTE[i % len(PALETTE)] for i, m in enumerate(sorted(models))}


# ── Figure 1: Composite score comparison ──────────────────────────────────────

def plot_composite_scores(df: pd.DataFrame, out_dir: Path):
    dims = [
        ("harm_score",             "Harm Score\n(lower = better, 0–4)"),
        ("help_score",             "Help Score\n(higher = better, 1–5)"),
        ("boundary_appropriateness", "AI Quality\n(higher = better, 1–5)"),
    ]

    agg = df.groupby("model_name")[[c for c, _ in dims]].mean().reset_index()
    models = sorted(agg["model_name"].tolist())
    colors = _model_colors(models)

    fig, axes = plt.subplots(1, 3, figsize=(14, 5))
    fig.suptitle("Composite Score Comparison Across Models", fontsize=14, fontweight="bold")

    ylims = [4, 5, 5]
    for ax, (col, label), ylim in zip(axes, dims, ylims):
        vals = [agg.loc[agg["model_name"] == m, col].values[0] for m in models]
        bars = ax.bar(models, vals, color=[colors[m] for m in models],
                      width=0.5, edgecolor="white", linewidth=1.5)
        ax.set_title(label, fontsize=10)
        ax.set_ylim(0, ylim)
        ax.tick_params(axis="x", rotation=20, labelsize=8)
        for bar, val in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.05,
                    f"{val:.2f}", ha="center", va="bottom", fontsize=9)

    plt.tight_layout()
    _save(fig, out_dir, "01_composite_scores.png")


# ── Figure 2: Safety–Empathy tradeoff scatter ──────────────────────────────────

def plot_safety_empathy_tradeoff(df: pd.DataFrame, out_dir: Path):
    agg = df.groupby("model_name").agg(
        harm_mean=("harm_score", "mean"),
        help_mean=("help_score", "mean"),
    ).reset_index()

    colors = _model_colors(agg["model_name"].tolist())

    fig, ax = plt.subplots(figsize=(7, 6))

    for _, row in agg.iterrows():
        ax.scatter(row["harm_mean"], row["help_mean"],
                   color=colors[row["model_name"]], s=180, zorder=3,
                   edgecolors="white", linewidths=1.5)
        ax.annotate(row["model_name"], (row["harm_mean"], row["help_mean"]),
                    textcoords="offset points", xytext=(8, 4), fontsize=9)

    ax.axvline(x=1.0, color="gray", linestyle="--", alpha=0.4, linewidth=1)
    ax.axhline(y=3.0, color="gray", linestyle="--", alpha=0.4, linewidth=1)

    kw = dict(fontsize=8, alpha=0.6)
    ax.text(0.05, 4.8, "Safe & Helpful",       color="green",     **kw)
    ax.text(1.8,  4.8, "Helpful but Risky",    color="darkorange", **kw)
    ax.text(0.05, 1.1, "Safe but Unhelpful",   color="steelblue", **kw)
    ax.text(1.8,  1.1, "Risky & Unhelpful",    color="red",       **kw)

    ax.set_xlabel("Harm Score (lower = better)", fontsize=11)
    ax.set_ylabel("Help Score (higher = better)", fontsize=11)
    ax.set_title("Safety–Empathy Tradeoff", fontsize=13, fontweight="bold")
    ax.set_xlim(-0.2, 4.2)
    ax.set_ylim(0.8, 5.2)

    plt.tight_layout()
    _save(fig, out_dir, "02_safety_empathy_tradeoff.png")


# ── Figure 3: Radar chart ──────────────────────────────────────────────────────

def plot_radar(df: pd.DataFrame, out_dir: Path):
    # (column, min, max, invert?)
    metric_map = [
        ("Emotional\nValidation",  "emotional_validation",     1, 5, False),
        ("Actionable\nSupport",    "actionable_support",       1, 5, False),
        ("Cognitive\nReframing",   "cognitive_reframing",      1, 5, False),
        ("Boundary\nApprop.",      "boundary_appropriateness", 1, 5, False),
        ("Low Harm\n(inverted)",   "harm_score",               0, 4, True),
    ]

    labels = [m[0] for m in metric_map]
    N = len(labels)
    angles = np.linspace(0, 2 * np.pi, N, endpoint=False).tolist()
    angles += angles[:1]

    models = sorted(df["model_name"].unique())
    colors = _model_colors(models)

    fig, ax = plt.subplots(figsize=(7, 7), subplot_kw=dict(polar=True))

    for model in models:
        sub = df[df["model_name"] == model]
        vals = []
        for _, col, lo, hi, invert in metric_map:
            mean = sub[col].mean()
            norm = (mean - lo) / (hi - lo)
            if invert:
                norm = 1 - norm
            vals.append(float(np.clip(norm, 0, 1)))
        vals += vals[:1]

        ax.plot(angles, vals, color=colors[model], linewidth=2, label=model)
        ax.fill(angles, vals, color=colors[model], alpha=0.12)

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(labels, fontsize=9)
    ax.set_ylim(0, 1)
    ax.set_yticks([0.25, 0.5, 0.75, 1.0])
    ax.set_yticklabels(["0.25", "0.50", "0.75", "1.00"], fontsize=7)
    ax.set_title("Model Profile Comparison (normalized to 0–1)",
                 fontsize=12, fontweight="bold", pad=20)
    ax.legend(loc="upper right", bbox_to_anchor=(1.35, 1.1), fontsize=9)

    plt.tight_layout()
    _save(fig, out_dir, "03_radar_model_profiles.png")


# ── Figure 4: Error tag frequency ─────────────────────────────────────────────

def plot_error_tags(df: pd.DataFrame, out_dir: Path):
    rows = [
        {"model_name": r["model_name"], "tag": tag}
        for _, r in df.iterrows()
        for tag in r["error_tags"]
    ]

    if not rows:
        print("  No error tags found — skipping Figure 4.")
        return

    tag_df = pd.DataFrame(rows)
    pivot = tag_df.groupby(["tag", "model_name"]).size().unstack(fill_value=0)
    models = sorted(pivot.columns.tolist())
    colors = _model_colors(models)

    fig, ax = plt.subplots(figsize=(10, max(4, len(pivot) * 0.7 + 2)))
    pivot[models].plot(kind="barh", ax=ax,
                       color=[colors[m] for m in models],
                       edgecolor="white", linewidth=1)
    ax.set_xlabel("Count", fontsize=11)
    ax.set_title("Error Tag Frequency by Model", fontsize=13, fontweight="bold")
    ax.legend(title="Model", fontsize=9)

    plt.tight_layout()
    _save(fig, out_dir, "04_error_tags.png")


# ── Figure 5: Scores by escalation level ──────────────────────────────────────

def plot_by_escalation(df: pd.DataFrame, out_dir: Path):
    if df["escalation_level"].isna().all():
        print("  No escalation_level data — skipping Figure 5.")
        return

    models = sorted(df["model_name"].unique())
    colors = _model_colors(models)

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle("Scores Across Escalation Levels", fontsize=13, fontweight="bold")

    for ax, col, label, ylim in [
        (axes[0], "harm_score", "Harm Score (lower = better)", (0, 4)),
        (axes[1], "help_score", "Help Score (higher = better)", (1, 5)),
    ]:
        for model in models:
            sub = (df[df["model_name"] == model]
                   .groupby("escalation_level")[col].mean())
            ax.plot(sub.index, sub.values, marker="o", label=model,
                    color=colors[model], linewidth=2, markersize=6)
        ax.set_xlabel("Escalation Level", fontsize=10)
        ax.set_ylabel(label, fontsize=10)
        ax.set_ylim(*ylim)
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    _save(fig, out_dir, "05_scores_by_escalation.png")


# ── Figure 6: Per-metric box plots ────────────────────────────────────────────

def plot_metric_distributions(df: pd.DataFrame, out_dir: Path):
    metrics = [
        ("emotional_invalidation",   "Emotional Invalidation (0–2)"),
        ("maladaptive_reinforcement", "Maladaptive Reinforcement (0–2)"),
        ("emotional_validation",     "Emotional Validation (1–5)"),
        ("actionable_support",       "Actionable Support (1–5)"),
        ("cognitive_reframing",      "Cognitive Reframing (1–5)"),
        ("boundary_appropriateness", "Boundary Appropriateness (1–5)"),
    ]

    models = sorted(df["model_name"].unique())
    palette = {m: PALETTE[i % len(PALETTE)] for i, m in enumerate(models)}

    fig, axes = plt.subplots(2, 3, figsize=(14, 9))
    fig.suptitle("Per-Metric Score Distributions by Model",
                 fontsize=13, fontweight="bold")

    for ax, (col, title) in zip(axes.flat, metrics):
        sub = df[["model_name", col]].dropna()
        sns.boxplot(data=sub, x="model_name", y=col, hue="model_name", ax=ax,
                    palette=palette, width=0.5, linewidth=1.2, order=models, legend=False)
        ax.set_title(title, fontsize=9)
        ax.set_xlabel("")
        ax.set_ylabel("Score", fontsize=8)
        ax.tick_params(axis="x", rotation=20, labelsize=8)

    plt.tight_layout()
    _save(fig, out_dir, "06_metric_distributions.png")


# ── Summary table ──────────────────────────────────────────────────────────────

def print_and_save_summary(df: pd.DataFrame, out_dir: Path) -> pd.DataFrame:
    agg = df.groupby("model_name").agg(
        n_turns=("turn", "count"),
        harm_score=("harm_score", "mean"),
        harm_rate=("any_harm", "mean"),
        help_score=("help_score", "mean"),
        emotional_validation=("emotional_validation", "mean"),
        actionable_support=("actionable_support", "mean"),
        cognitive_reframing=("cognitive_reframing", "mean"),
        ai_quality=("boundary_appropriateness", "mean"),
        referral_accuracy=("referral_correct", "mean"),
    ).round(3).reset_index()

    print("\n=== Summary Table ===")
    print(agg.to_string(index=False))

    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / "summary_table.csv"
    agg.to_csv(csv_path, index=False)
    print(f"Summary saved to: {csv_path}")
    return agg


# ── Helpers ────────────────────────────────────────────────────────────────────

def _save(fig, out_dir: Path, name: str):
    out_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_dir / name, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {name}")


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Analyze LLM evaluation results")
    parser.add_argument("--eval-dir", default="data/eval_results",
                        help="Path to eval results directory (default: data/eval_results)")
    parser.add_argument("--output", default="analysis/figures",
                        help="Output directory for figures (default: analysis/figures)")
    args = parser.parse_args()

    eval_dir = Path(args.eval_dir)
    out_dir  = Path(args.output)

    print(f"Loading eval results from: {eval_dir}")
    df = load_eval_results(eval_dir)

    if df.empty:
        print("No eval results found. Run the evaluation pipeline first.")
        return

    print(f"Loaded {len(df)} turns | "
          f"{df['model_name'].nunique()} models | "
          f"{df['stimulus_id'].nunique()} stimuli")

    print_and_save_summary(df, out_dir)

    print("\nGenerating figures...")
    plot_composite_scores(df, out_dir)
    plot_safety_empathy_tradeoff(df, out_dir)
    plot_radar(df, out_dir)
    plot_error_tags(df, out_dir)
    plot_by_escalation(df, out_dir)
    plot_metric_distributions(df, out_dir)

    print(f"\nDone. All outputs saved to: {out_dir}")


if __name__ == "__main__":
    main()
