"""
Cross-group comparison analysis for LLM mental health evaluation.

Loads all 6 subsets and compares across two axes:
  - Model tier:   dev vs prod
  - Input length: human_checked (short) / llm_gen short / llm_gen long

Produces:
  C1_dev_vs_prod.png          — harm/help by model tier, faceted by scenario group
  C2_input_length.png         — harm/help by input length (prod models only)
  C3_scenario_source.png      — human-checked vs LLM-generated (prod, short only)
  C4_full_heatmap.png         — all 6 groups × 3 key metrics heatmap
  C5_per_model_tier.png       — per-model breakdown within dev / prod tiers

Usage:
  python analysis/compare.py
  python analysis/compare.py --eval-dir data/eval_results --output analysis/figures/compare
"""

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd
import seaborn as sns

sns.set_theme(style="whitegrid", font_scale=1.05)
PALETTE = sns.color_palette("Set2")

# ── Group definitions ──────────────────────────────────────────────────────────

GROUPS = [
    {
        "label":        "human_checked · short · dev",
        "scenario_src": "human_checked",
        "input_len":    "short",
        "model_tier":   "dev",
        "rel_path":     "human_checked_scenarios/dev_models",
    },
    {
        "label":        "human_checked · short · prod",
        "scenario_src": "human_checked",
        "input_len":    "short",
        "model_tier":   "prod",
        "rel_path":     "human_checked_scenarios/prod_models",
    },
    {
        "label":        "llm_gen · short · dev",
        "scenario_src": "llm_generated",
        "input_len":    "short",
        "model_tier":   "dev",
        "rel_path":     "llm_generated_scenarios/short_input/dev_models",
    },
    {
        "label":        "llm_gen · short · prod",
        "scenario_src": "llm_generated",
        "input_len":    "short",
        "model_tier":   "prod",
        "rel_path":     "llm_generated_scenarios/short_input/prod_models",
    },
    {
        "label":        "llm_gen · long · dev",
        "scenario_src": "llm_generated",
        "input_len":    "long",
        "model_tier":   "dev",
        "rel_path":     "llm_generated_scenarios/long_input/dev_models",
    },
    {
        "label":        "llm_gen · long · prod",
        "scenario_src": "llm_generated",
        "input_len":    "long",
        "model_tier":   "prod",
        "rel_path":     "llm_generated_scenarios/long_input/prod_models",
    },
]


# ── Data loading ───────────────────────────────────────────────────────────────

def _score(metrics, key):
    m = metrics.get(key, {})
    s = m.get("score") if isinstance(m, dict) else m
    if s in ("NA", None):
        return np.nan
    try:
        return float(s)
    except (ValueError, TypeError):
        return np.nan


def _detect_version(paths):
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


def load_group(eval_dir: Path, group: dict) -> pd.DataFrame:
    group_dir = eval_dir / group["rel_path"]
    paths = sorted(group_dir.rglob("eval_*.json"))
    if not paths:
        print(f"  [WARN] No files found in {group_dir}")
        return pd.DataFrame()

    version = _detect_version(paths)
    records = []

    for path in paths:
        with open(path, encoding="utf-8") as f:
            data = json.load(f)

        model_name  = data.get("model_name", "unknown")
        stimulus_id = data.get("stimulus_id", "unknown")
        theme       = data.get("theme", "unknown")

        for turn in data.get("turns", []):
            m = turn.get("evaluation_metrics") or {}

            if version == "v2":
                h1 = _score(m, "response_attunement")
                h2 = _score(m, "sycophantic_accommodation")
            else:
                h1 = _score(m, "emotional_invalidation")
                h2 = _score(m, "maladaptive_reinforcement")

            harm = h1 + h2 if not (np.isnan(h1) or np.isnan(h2)) else np.nan

            ev  = _score(m, "emotional_validation")
            as_ = _score(m, "actionable_support")
            cr  = _score(m, "cognitive_reframing")
            ba  = _score(m, "boundary_appropriateness")

            help_score = np.nanmean([ev, as_, cr]) if not all(np.isnan([ev, as_, cr])) else np.nan

            records.append({
                "group":        group["label"],
                "scenario_src": group["scenario_src"],
                "input_len":    group["input_len"],
                "model_tier":   group["model_tier"],
                "model_name":   model_name,
                "stimulus_id":  stimulus_id,
                "theme":        theme,
                "severity_tier":turn.get("severity_tier"),
                "word_count":   turn.get("word_count"),
                "harm_score":   harm,
                "help_score":   help_score,
                "ai_quality":   ba,
                "rubric_version": version,
            })

    return pd.DataFrame(records)


def load_all(eval_dir: Path) -> pd.DataFrame:
    frames = []
    for g in GROUPS:
        print(f"  Loading: {g['label']} ...")
        df = load_group(eval_dir, g)
        if not df.empty:
            frames.append(df)
            print(f"           {len(df)} turns, {df['model_name'].nunique()} models")
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


# ── Helpers ────────────────────────────────────────────────────────────────────

def _save(fig, out_dir: Path, name: str):
    out_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_dir / name, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {name}")


TIER_COLORS = {"dev": "#7fc97f", "prod": "#386cb0"}
LEN_COLORS  = {"short": "#fdc086", "long": "#beaed4"}
SRC_COLORS  = {"human_checked": "#f0027f", "llm_generated": "#bf5b17"}


# ── C1: Dev vs Prod ────────────────────────────────────────────────────────────

def plot_dev_vs_prod(df: pd.DataFrame, out_dir: Path):
    """Side-by-side bar: harm & help scores, grouped by scenario group, split dev/prod."""
    scenario_groups = [
        ("human_checked", "short", "Human-Checked\n(Short)"),
        ("llm_generated",  "short", "LLM-Generated\n(Short)"),
        ("llm_generated",  "long",  "LLM-Generated\n(Long)"),
    ]

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle("Dev vs Prod: Harm & Help Scores by Scenario Group",
                 fontsize=13, fontweight="bold")

    bar_width = 0.35
    x = np.arange(len(scenario_groups))

    for ax, col, title, ylim in [
        (axes[0], "harm_score", "Harm Score (lower = better)", (0, 4)),
        (axes[1], "help_score", "Help Score (higher = better)", (0, 5)),
    ]:
        dev_vals, prod_vals = [], []
        dev_errs, prod_errs = [], []

        for src, ilen, _ in scenario_groups:
            sub_dev  = df[(df["scenario_src"] == src) & (df["input_len"] == ilen) & (df["model_tier"] == "dev")][col].dropna()
            sub_prod = df[(df["scenario_src"] == src) & (df["input_len"] == ilen) & (df["model_tier"] == "prod")][col].dropna()
            dev_vals.append(sub_dev.mean() if len(sub_dev) else np.nan)
            prod_vals.append(sub_prod.mean() if len(sub_prod) else np.nan)
            dev_errs.append(sub_dev.sem() if len(sub_dev) else 0)
            prod_errs.append(sub_prod.sem() if len(sub_prod) else 0)

        bars_dev  = ax.bar(x - bar_width/2, dev_vals,  bar_width, yerr=dev_errs,
                           color=TIER_COLORS["dev"],  label="Dev",  capsize=4,
                           edgecolor="white", linewidth=1.2)
        bars_prod = ax.bar(x + bar_width/2, prod_vals, bar_width, yerr=prod_errs,
                           color=TIER_COLORS["prod"], label="Prod", capsize=4,
                           edgecolor="white", linewidth=1.2)

        for bar, val in zip(list(bars_dev) + list(bars_prod),
                            dev_vals + prod_vals):
            if not np.isnan(val):
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.05,
                        f"{val:.2f}", ha="center", va="bottom", fontsize=8)

        ax.set_xticks(x)
        ax.set_xticklabels([g[2] for g in scenario_groups], fontsize=9)
        ax.set_title(title, fontsize=10)
        ax.set_ylim(*ylim)
        ax.legend(fontsize=9)

    plt.tight_layout()
    _save(fig, out_dir, "C1_dev_vs_prod.png")


# ── C2: Input length comparison (prod only) ───────────────────────────────────

def plot_input_length(df: pd.DataFrame, out_dir: Path):
    """For prod models + llm_generated stimuli: short vs long input."""
    prod = df[(df["model_tier"] == "prod") & (df["scenario_src"] == "llm_generated")]

    if prod.empty:
        print("  No prod/llm_generated data — skipping C2.")
        return

    models = sorted(prod["model_name"].unique())
    x = np.arange(len(models))
    bar_width = 0.35

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle("Short vs Long Input: Prod Models (LLM-Generated Stimuli)",
                 fontsize=13, fontweight="bold")

    for ax, col, title, ylim in [
        (axes[0], "harm_score", "Harm Score (lower = better)", (0, 4)),
        (axes[1], "help_score", "Help Score (higher = better)", (0, 5)),
    ]:
        short_vals, long_vals = [], []
        short_errs, long_errs = [], []

        for m in models:
            s = prod[(prod["model_name"] == m) & (prod["input_len"] == "short")][col].dropna()
            l = prod[(prod["model_name"] == m) & (prod["input_len"] == "long")][col].dropna()
            short_vals.append(s.mean() if len(s) else np.nan)
            long_vals.append(l.mean()  if len(l) else np.nan)
            short_errs.append(s.sem()  if len(s) else 0)
            long_errs.append(l.sem()   if len(l) else 0)

        bars_s = ax.bar(x - bar_width/2, short_vals, bar_width, yerr=short_errs,
                        color=LEN_COLORS["short"], label="Short", capsize=4,
                        edgecolor="white", linewidth=1.2)
        bars_l = ax.bar(x + bar_width/2, long_vals,  bar_width, yerr=long_errs,
                        color=LEN_COLORS["long"],  label="Long",  capsize=4,
                        edgecolor="white", linewidth=1.2)

        for bar, val in zip(list(bars_s) + list(bars_l),
                            short_vals + long_vals):
            if not np.isnan(val):
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.05,
                        f"{val:.2f}", ha="center", va="bottom", fontsize=8)

        ax.set_xticks(x)
        ax.set_xticklabels(models, rotation=15, ha="right", fontsize=9)
        ax.set_title(title, fontsize=10)
        ax.set_ylim(*ylim)
        ax.legend(fontsize=9)

    plt.tight_layout()
    _save(fig, out_dir, "C2_input_length.png")


# ── C3: Human-checked vs LLM-generated (prod, short) ─────────────────────────

def plot_scenario_source(df: pd.DataFrame, out_dir: Path):
    """Prod models: human-checked stimuli vs LLM-generated short stimuli."""
    prod = df[(df["model_tier"] == "prod") & (df["input_len"] == "short")]

    if prod.empty:
        print("  No prod/short data — skipping C3.")
        return

    models = sorted(prod["model_name"].unique())
    x = np.arange(len(models))
    bar_width = 0.35

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle("Human-Checked vs LLM-Generated Stimuli: Prod Models (Short Input)",
                 fontsize=13, fontweight="bold")

    for ax, col, title, ylim in [
        (axes[0], "harm_score", "Harm Score (lower = better)", (0, 4)),
        (axes[1], "help_score", "Help Score (higher = better)", (0, 5)),
    ]:
        hc_vals,  llm_vals  = [], []
        hc_errs,  llm_errs  = [], []

        for m in models:
            hc  = prod[(prod["model_name"] == m) & (prod["scenario_src"] == "human_checked")][col].dropna()
            llm = prod[(prod["model_name"] == m) & (prod["scenario_src"] == "llm_generated")][col].dropna()
            hc_vals.append(hc.mean()   if len(hc)  else np.nan)
            llm_vals.append(llm.mean() if len(llm) else np.nan)
            hc_errs.append(hc.sem()    if len(hc)  else 0)
            llm_errs.append(llm.sem()  if len(llm) else 0)

        bars_hc  = ax.bar(x - bar_width/2, hc_vals,  bar_width, yerr=hc_errs,
                          color=SRC_COLORS["human_checked"], label="Human-Checked", capsize=4,
                          edgecolor="white", linewidth=1.2)
        bars_llm = ax.bar(x + bar_width/2, llm_vals, bar_width, yerr=llm_errs,
                          color=SRC_COLORS["llm_generated"], label="LLM-Generated",  capsize=4,
                          edgecolor="white", linewidth=1.2)

        for bar, val in zip(list(bars_hc) + list(bars_llm),
                            hc_vals + llm_vals):
            if not np.isnan(val):
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.05,
                        f"{val:.2f}", ha="center", va="bottom", fontsize=8)

        ax.set_xticks(x)
        ax.set_xticklabels(models, rotation=15, ha="right", fontsize=9)
        ax.set_title(title, fontsize=10)
        ax.set_ylim(*ylim)
        ax.legend(fontsize=9)

    plt.tight_layout()
    _save(fig, out_dir, "C3_scenario_source.png")


# ── C4: Full heatmap — all 6 groups × 3 metrics ───────────────────────────────

def plot_full_heatmap(df: pd.DataFrame, out_dir: Path):
    """Heatmap: rows = group label, cols = harm / help / ai_quality."""
    agg = df.groupby("group").agg(
        harm_score=("harm_score", "mean"),
        help_score=("help_score", "mean"),
        ai_quality=("ai_quality", "mean"),
    ).round(3)

    # Reorder rows by GROUPS definition order
    order = [g["label"] for g in GROUPS if g["label"] in agg.index]
    agg = agg.reindex(order)

    fig, axes = plt.subplots(1, 3, figsize=(12, 5))
    fig.suptitle("All 6 Groups × Key Metrics (mean scores)",
                 fontsize=13, fontweight="bold")

    configs = [
        ("harm_score", "Harm Score\n(lower = better, 0–4)", "YlOrRd", 0, 4),
        ("help_score", "Help Score\n(higher = better, 1–5)", "YlGn",   1, 5),
        ("ai_quality", "AI Quality\n(higher = better, 1–5)", "Blues",  1, 5),
    ]

    for ax, (col, title, cmap, vmin, vmax) in zip(axes, configs):
        data = agg[[col]]
        sns.heatmap(data, ax=ax, annot=True, fmt=".2f",
                    cmap=cmap, vmin=vmin, vmax=vmax,
                    linewidths=0.5, cbar_kws={"shrink": 0.8},
                    yticklabels=(ax == axes[0]))
        ax.set_title(title, fontsize=9)
        ax.set_xlabel("")
        ax.set_ylabel("")
        if ax != axes[0]:
            ax.set_yticklabels([])

    plt.tight_layout()
    _save(fig, out_dir, "C4_full_heatmap.png")


# ── C5: Per-model breakdown within tier ───────────────────────────────────────

def plot_per_model_tier(df: pd.DataFrame, out_dir: Path):
    """For each tier, show individual model scores across all 3 scenario groups."""
    scenario_order = [
        ("human_checked", "short", "HC·short"),
        ("llm_generated",  "short", "Gen·short"),
        ("llm_generated",  "long",  "Gen·long"),
    ]

    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle("Per-Model Scores Across Scenario Groups by Tier",
                 fontsize=13, fontweight="bold")

    for row_idx, tier in enumerate(["dev", "prod"]):
        tier_df = df[df["model_tier"] == tier]
        models  = sorted(tier_df["model_name"].unique())
        colors  = {m: PALETTE[i % len(PALETTE)] for i, m in enumerate(models)}
        x_labels = [s[2] for s in scenario_order]
        x = np.arange(len(scenario_order))

        for col_idx, (col, ylabel, ylim) in enumerate([
            ("harm_score", "Harm Score (lower = better)", (0, 4)),
            ("help_score", "Help Score (higher = better)", (0, 5)),
        ]):
            ax = axes[row_idx][col_idx]
            bar_width = 0.8 / max(len(models), 1)

            for m_idx, model in enumerate(models):
                vals = []
                for src, ilen, _ in scenario_order:
                    sub = tier_df[
                        (tier_df["model_name"] == model) &
                        (tier_df["scenario_src"] == src) &
                        (tier_df["input_len"] == ilen)
                    ][col].dropna()
                    vals.append(sub.mean() if len(sub) else np.nan)

                offset = (m_idx - len(models)/2 + 0.5) * bar_width
                ax.bar(x + offset, vals, bar_width,
                       color=colors[model], label=model,
                       edgecolor="white", linewidth=0.8)

            ax.set_xticks(x)
            ax.set_xticklabels(x_labels, fontsize=9)
            ax.set_title(f"{tier.upper()} — {ylabel}", fontsize=9)
            ax.set_ylim(*ylim)
            ax.set_ylabel(ylabel, fontsize=8)
            if col_idx == 1:
                ax.legend(fontsize=7, loc="upper right")

    plt.tight_layout()
    _save(fig, out_dir, "C5_per_model_tier.png")


# ── Summary CSV ────────────────────────────────────────────────────────────────

def save_summary(df: pd.DataFrame, out_dir: Path):
    agg = df.groupby(["group", "scenario_src", "input_len", "model_tier", "model_name"]).agg(
        n_turns=("harm_score", "count"),
        harm_score=("harm_score", "mean"),
        help_score=("help_score", "mean"),
        ai_quality=("ai_quality", "mean"),
    ).round(3).reset_index()

    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "comparison_summary.csv"
    agg.to_csv(path, index=False)
    print(f"\nSummary CSV saved: {path}")
    print(agg.to_string(index=False))


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--eval-dir", default="data/eval_results")
    parser.add_argument("--output",   default="analysis/figures/compare")
    args = parser.parse_args()

    eval_dir = Path(args.eval_dir)
    out_dir  = Path(args.output)

    print(f"Loading all groups from: {eval_dir}\n")
    df = load_all(eval_dir)

    if df.empty:
        print("No data loaded. Exiting.")
        return

    print(f"\nTotal: {len(df)} turns | "
          f"{df['model_name'].nunique()} models | "
          f"{df['group'].nunique()} groups\n")

    save_summary(df, out_dir)

    print("\nGenerating figures...")
    plot_dev_vs_prod(df, out_dir)
    plot_input_length(df, out_dir)
    plot_scenario_source(df, out_dir)
    plot_full_heatmap(df, out_dir)
    plot_per_model_tier(df, out_dir)

    print(f"\nDone. Figures saved to: {out_dir}")


if __name__ == "__main__":
    main()
