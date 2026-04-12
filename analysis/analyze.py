"""
Data analysis and visualization for LLM mental health evaluation results.

Reads all eval_*.json files from data/eval_results/ and generates:
  01_composite_scores.png       — bar chart: harm / help / AI quality per model
  02_safety_empathy_tradeoff.png — scatter: harm vs help with quadrant labels
  03_radar_model_profiles.png   — radar chart: all dimensions normalized
  04_error_tags.png             — error tag frequency by model
  05_scores_by_severity.png     — line chart: scores across severity tiers
  06_metric_distributions.png   — box plots: per-metric distributions
  07_scores_by_theme.png        — heatmap: help/harm scores per theme × model
  summary_table.csv             — aggregate stats per model

Usage:
  python analysis/analyze.py
  python analysis/analyze.py --eval-dir data/eval_results --output analysis/figures
"""

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

sns.set_theme(style="whitegrid", font_scale=1.05)
PALETTE = sns.color_palette("Set2")

SEVERITY_ORDER = ["baseline", "medium", "stress_test"]


# ── Data loading ───────────────────────────────────────────────────────────────

def _detect_version(paths) -> str:
    """Auto-detect rubric version by inspecting the first available eval file.
    Accepts either a Path (directory) or an iterable of file Paths.
    """
    if isinstance(paths, Path):
        paths = (p for p in paths.rglob("eval_*.json")
                 if not any(part.startswith("_") for part in p.parts))
    for path in paths:
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        for turn in data.get("turns", []):
            m = turn.get("evaluation_metrics") or {}
            if "response_attunement" in m:
                return "v2"
            if "emotional_invalidation" in m:
                return "v1"
    return "v1"


def load_eval_results(eval_dirs) -> pd.DataFrame:
    """Load all eval_*.json files and flatten to one row per turn.

    eval_dirs may be a single Path or a list of Paths.

    Auto-detects rubric version (v1 / v2) from field names.
    v1 harm fields: emotional_invalidation, maladaptive_reinforcement (0-2 each)
    v2 harm fields: response_attunement, sycophantic_accommodation  (0-4 each)
    """
    if isinstance(eval_dirs, Path):
        eval_dirs = [eval_dirs]

    all_paths = sorted(
        p
        for d in eval_dirs
        for p in d.rglob("eval_*.json")
        if not any(part.startswith("_") for part in p.parts)
    )

    version = _detect_version(iter(all_paths))
    print(f"Detected rubric version: {version}")

    records = []

    for path in all_paths:
        with open(path, encoding="utf-8") as f:
            data = json.load(f)

        model_name  = data.get("model_name", "unknown")
        stimulus_id = data.get("stimulus_id", "unknown")
        theme       = data.get("theme", "unknown")
        variant     = data.get("variant", "unknown")

        for turn in data.get("turns", []):
            metrics      = turn.get("evaluation_metrics") or {}
            judge_status = turn.get("judge_status", "unknown")

            def score(key):
                m = metrics.get(key, {})
                if isinstance(m, dict):
                    s = m.get("score")
                else:
                    s = m
                if s == "NA" or s is None:
                    return np.nan
                try:
                    return float(s)
                except (ValueError, TypeError):
                    return np.nan

            ref_raw = None
            ref_field = metrics.get("professional_referral", {})
            if isinstance(ref_field, dict):
                ref_raw = ref_field.get("score")

            # Harm fields differ between v1 and v2
            if version == "v2":
                h1 = score("response_attunement")
                h2 = score("sycophantic_accommodation")
            else:
                h1 = score("emotional_invalidation")
                h2 = score("maladaptive_reinforcement")

            harm = h1 + h2 if not (np.isnan(h1) or np.isnan(h2)) else np.nan

            records.append({
                "stimulus_id":   stimulus_id,
                "theme":         theme,
                "variant":       variant,
                "model_name":    model_name,
                "turn":          turn.get("turn"),
                "severity_tier": turn.get("severity_tier"),
                "data_source":   turn.get("data_source"),
                "word_count":    turn.get("word_count"),
                "judge_status":  judge_status,
                "rubric_version": version,
                # Harm (unified names for downstream code)
                "harm_1":     h1,
                "harm_2":     h2,
                "harm_score": harm,
                # Help
                "emotional_validation": score("emotional_validation"),
                "actionable_support":   score("actionable_support"),
                "cognitive_reframing":  score("cognitive_reframing"),
                # AI Quality
                "boundary_appropriateness":  score("boundary_appropriateness"),
                "professional_referral_raw": ref_raw,
                # Error tags
                "error_tags": metrics.get("error_tags", []),
            })

    if not records:
        return pd.DataFrame()

    df = pd.DataFrame(records)
    df["help_score"] = df[["emotional_validation", "actionable_support", "cognitive_reframing"]].mean(axis=1)
    df["any_harm"]   = df["harm_score"].fillna(0) > 0
    df["referral_correct"] = df["professional_referral_raw"].apply(
        lambda x: str(x) in ("2", "NA") if pd.notna(x) else False
    )
    # Ordered categorical for severity_tier
    df["severity_tier"] = pd.Categorical(
        df["severity_tier"], categories=SEVERITY_ORDER, ordered=True
    )
    return df


def _model_colors(models):
    return {m: PALETTE[i % len(PALETTE)] for i, m in enumerate(sorted(models))}


def _save(fig, out_dir: Path, name: str):
    out_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_dir / name, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {name}")


# ── Figure 1: Composite score comparison ──────────────────────────────────────

def plot_composite_scores(df: pd.DataFrame, out_dir: Path):
    # Only use turns with valid judge output (skip partial if harm fields missing)
    dims = [
        ("harm_score",               "Harm Score\n(lower = better, 0–4)"),
        ("help_score",               "Help Score\n(higher = better, 1–5)"),
        ("boundary_appropriateness", "AI Quality\n(higher = better, 1–5)"),
    ]

    agg    = df.groupby("model_name")[[c for c, _ in dims]].mean().reset_index()
    models = sorted(agg["model_name"].tolist())
    colors = _model_colors(models)

    fig, axes = plt.subplots(1, 3, figsize=(14, 5))
    fig.suptitle("Composite Score Comparison Across Models", fontsize=14, fontweight="bold")

    for ax, (col, label), ylim in zip(axes, dims, [4, 5, 5]):
        vals = [agg.loc[agg["model_name"] == m, col].values[0] for m in models]
        bars = ax.bar(models, vals, color=[colors[m] for m in models],
                      width=0.5, edgecolor="white", linewidth=1.5)
        ax.set_title(label, fontsize=10)
        ax.set_ylim(0, ylim)
        ax.tick_params(axis="x", rotation=20, labelsize=8)
        for bar, val in zip(bars, vals):
            if not np.isnan(val):
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
    ax.text(0.05, 4.8, "Safe & Helpful",      color="green",      **kw)
    ax.text(1.8,  4.8, "Helpful but Risky",   color="darkorange", **kw)
    ax.text(0.05, 1.1, "Safe but Unhelpful",  color="steelblue",  **kw)
    ax.text(1.8,  1.1, "Risky & Unhelpful",   color="red",        **kw)

    ax.set_xlabel("Harm Score (lower = better)", fontsize=11)
    ax.set_ylabel("Help Score (higher = better)", fontsize=11)
    ax.set_title("Safety–Empathy Tradeoff", fontsize=13, fontweight="bold")
    ax.set_xlim(-0.2, 4.2)
    ax.set_ylim(0.8, 5.2)

    plt.tight_layout()
    _save(fig, out_dir, "02_safety_empathy_tradeoff.png")


# ── Figure 3: Radar chart ──────────────────────────────────────────────────────

def plot_radar(df: pd.DataFrame, out_dir: Path):
    metric_map = [
        ("Emotional\nValidation",  "emotional_validation",     1, 5, False),
        ("Actionable\nSupport",    "actionable_support",       1, 5, False),
        ("Cognitive\nReframing",   "cognitive_reframing",      1, 5, False),
        ("Boundary\nApprop.",      "boundary_appropriateness", 1, 5, False),
        ("Low Harm\n(inverted)",   "harm_score",               0, 4, True),
    ]

    labels = [m[0] for m in metric_map]
    N      = len(labels)
    angles = np.linspace(0, 2 * np.pi, N, endpoint=False).tolist()
    angles += angles[:1]

    models = sorted(df["model_name"].unique())
    colors = _model_colors(models)

    fig, ax = plt.subplots(figsize=(7, 7), subplot_kw=dict(polar=True))

    for model in models:
        sub  = df[df["model_name"] == model]
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
    models = sorted(tag_df["model_name"].unique())
    colors = _model_colors(models)
    pivot  = tag_df.groupby(["tag", "model_name"]).size().unstack(fill_value=0)
    pivot  = pivot.reindex(columns=models, fill_value=0)

    fig, ax = plt.subplots(figsize=(10, max(4, len(pivot) * 0.7 + 2)))
    pivot.plot(kind="barh", ax=ax,
               color=[colors[m] for m in models],
               edgecolor="white", linewidth=1)
    ax.set_xlabel("Count", fontsize=11)
    ax.set_title("Error Tag Frequency by Model", fontsize=13, fontweight="bold")
    ax.legend(title="Model", fontsize=9)

    plt.tight_layout()
    _save(fig, out_dir, "04_error_tags.png")


# ── Figure 5: Scores by severity tier ─────────────────────────────────────────

def plot_by_severity(df: pd.DataFrame, out_dir: Path):
    valid_tiers = [t for t in SEVERITY_ORDER if t in df["severity_tier"].cat.categories]
    sub = df[df["severity_tier"].isin(valid_tiers)]

    if sub.empty:
        print("  No severity_tier data — skipping Figure 5.")
        return

    models = sorted(df["model_name"].unique())
    colors = _model_colors(models)

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle("Scores Across Severity Tiers", fontsize=13, fontweight="bold")

    tier_labels = {"baseline": "Baseline", "medium": "Medium", "stress_test": "Stress Test"}

    for ax, col, label, ylim in [
        (axes[0], "harm_score", "Harm Score (lower = better)", (0, 4)),
        (axes[1], "help_score", "Help Score (higher = better)", (1, 5)),
    ]:
        for model in models:
            grp = (sub[sub["model_name"] == model]
                   .groupby("severity_tier", observed=True)[col].mean())
            x_labels = [tier_labels.get(t, t) for t in grp.index]
            ax.plot(x_labels, grp.values, marker="o", label=model,
                    color=colors[model], linewidth=2, markersize=7)

        ax.set_xlabel("Severity Tier", fontsize=10)
        ax.set_ylabel(label, fontsize=10)
        ax.set_ylim(*ylim)
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    _save(fig, out_dir, "05_scores_by_severity.png")


# ── Figure 6: Per-metric box plots ────────────────────────────────────────────

def plot_metric_distributions(df: pd.DataFrame, out_dir: Path):
    version = df["rubric_version"].iloc[0] if "rubric_version" in df.columns else "v1"
    if version == "v2":
        harm_metrics = [
            ("harm_1", "Response Attunement (0–4)"),
            ("harm_2", "Sycophantic Accommodation (0–4)"),
        ]
    else:
        harm_metrics = [
            ("harm_1", "Emotional Invalidation (0–2)"),
            ("harm_2", "Maladaptive Reinforcement (0–2)"),
        ]
    metrics = harm_metrics + [
        ("emotional_validation",     "Emotional Validation (1–5)"),
        ("actionable_support",       "Actionable Support (1–5)"),
        ("cognitive_reframing",      "Cognitive Reframing (1–5)"),
        ("boundary_appropriateness", "Boundary Appropriateness (1–5)"),
    ]

    models  = sorted(df["model_name"].unique())
    palette = {m: PALETTE[i % len(PALETTE)] for i, m in enumerate(models)}

    fig, axes = plt.subplots(2, 3, figsize=(14, 9))
    fig.suptitle("Per-Metric Score Distributions by Model",
                 fontsize=13, fontweight="bold")

    for ax, (col, title) in zip(axes.flat, metrics):
        sub = df[["model_name", col]].dropna()
        sns.boxplot(data=sub, x="model_name", y=col, hue="model_name", ax=ax,
                    palette=palette, width=0.5, linewidth=1.2,
                    order=models, legend=False)
        ax.set_title(title, fontsize=9)
        ax.set_xlabel("")
        ax.set_ylabel("Score", fontsize=8)
        ax.tick_params(axis="x", rotation=20, labelsize=8)

    plt.tight_layout()
    _save(fig, out_dir, "06_metric_distributions.png")


# ── Figure 7: Heatmap — scores by theme × model ───────────────────────────────

def plot_theme_heatmap(df: pd.DataFrame, out_dir: Path):
    if df["theme"].nunique() < 2:
        print("  Only one theme — skipping Figure 7.")
        return

    fig, axes = plt.subplots(1, 2, figsize=(14, max(5, df["theme"].nunique() * 0.5 + 2)))
    fig.suptitle("Mean Scores by Theme × Model", fontsize=13, fontweight="bold")

    for ax, col, title, fmt, vmin, vmax, cmap in [
        (axes[0], "harm_score", "Harm Score (lower = better)",
         ".2f", 0, 4, "YlOrRd"),
        (axes[1], "help_score", "Help Score (higher = better)",
         ".2f", 1, 5, "YlGn"),
    ]:
        pivot = (df.groupby(["theme", "model_name"])[col]
                 .mean()
                 .unstack("model_name")
                 .sort_index())
        sns.heatmap(pivot, ax=ax, annot=True, fmt=fmt,
                    vmin=vmin, vmax=vmax, cmap=cmap,
                    linewidths=0.5, cbar_kws={"shrink": 0.8})
        ax.set_title(title, fontsize=10)
        ax.set_xlabel("")
        ax.set_ylabel("Theme", fontsize=9)
        ax.tick_params(axis="x", rotation=20, labelsize=8)
        ax.tick_params(axis="y", rotation=0,  labelsize=8)

    plt.tight_layout()
    _save(fig, out_dir, "07_scores_by_theme.png")


# ── Summary table ──────────────────────────────────────────────────────────────

def print_and_save_summary(df: pd.DataFrame, out_dir: Path) -> pd.DataFrame:
    version = df["rubric_version"].iloc[0] if "rubric_version" in df.columns else "v1"
    h1_name = "response_attunement" if version == "v2" else "emotional_invalidation"
    h2_name = "sycophantic_accommodation" if version == "v2" else "maladaptive_reinforcement"

    agg = df.groupby("model_name").agg(
        n_turns=("turn", "count"),
        n_stimuli=("stimulus_id", "nunique"),
        harm_score=("harm_score", "mean"),
        harm_rate=("any_harm", "mean"),
        help_score=("help_score", "mean"),
        emotional_validation=("emotional_validation", "mean"),
        actionable_support=("actionable_support", "mean"),
        cognitive_reframing=("cognitive_reframing", "mean"),
        ai_quality=("boundary_appropriateness", "mean"),
        referral_accuracy=("referral_correct", "mean"),
    ).round(3).reset_index()
    agg = agg.rename(columns={"harm_score": f"harm_score ({h1_name}+{h2_name})"})

    print("\n=== Summary Table ===")
    print(agg.to_string(index=False))

    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / "summary_table.csv"
    agg.to_csv(csv_path, index=False)
    print(f"Summary saved to: {csv_path}")
    return agg


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Analyze LLM evaluation results")
    parser.add_argument("--eval-dir", nargs="+", default=["data/eval_results"],
                        help="One or more eval results directories to merge")
    parser.add_argument("--output", default="analysis/figures",
                        help="Output directory for figures (default: analysis/figures)")
    args = parser.parse_args()

    eval_dirs = [Path(d) for d in args.eval_dir]
    out_dir   = Path(args.output)

    print(f"Loading eval results from: {[str(d) for d in eval_dirs]}")
    df = load_eval_results(eval_dirs)

    if df.empty:
        print("No eval results found. Run the evaluation pipeline first.")
        return

    print(f"Loaded {len(df)} turns | "
          f"{df['model_name'].nunique()} models | "
          f"{df['stimulus_id'].nunique()} stimuli | "
          f"{df['theme'].nunique()} themes")

    # Report judge_status breakdown
    status_counts = df["judge_status"].value_counts()
    print(f"Judge status: {status_counts.to_dict()}")

    print_and_save_summary(df, out_dir)

    print("\nGenerating figures...")
    plot_composite_scores(df, out_dir)
    plot_safety_empathy_tradeoff(df, out_dir)
    plot_radar(df, out_dir)
    plot_error_tags(df, out_dir)
    plot_by_severity(df, out_dir)
    plot_metric_distributions(df, out_dir)
    plot_theme_heatmap(df, out_dir)

    print(f"\nDone. All outputs saved to: {out_dir}")


if __name__ == "__main__":
    main()
