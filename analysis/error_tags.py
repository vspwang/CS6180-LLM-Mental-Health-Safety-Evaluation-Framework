"""
Error Tag Cross-Analysis.

Investigates patterns across all 8 rubric-defined error tags, including
the 3 tags that never appear in data (minimization, hallucinated_resource,
toxic_positivity) — their absence is itself a finding.

Produces:
  T0_all_tags_overview.png        — all 8 tags: prevalence + absent tag analysis
  T1_model_error_heatmap.png      — error tag rates per model (heatmap, all 8 tags)
  T2_theme_error_heatmap.png      — error tag rates per theme (heatmap)
  T3_failure_mode_profiles.png    — two distinct failure modes: sycophancy vs over-solving
  T4_error_tags_by_severity.png   — tag trajectory across baseline/medium/stress_test
  T5_error_by_input_length.png    — error tag rates: short vs long input per model
  T6_harm_by_error_profile.png    — harm score for each error tag combination

Usage:
  python analysis/error_tags.py
  python analysis/error_tags.py --output analysis/figures/error_tags
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

SEVERITY_ORDER   = ["baseline", "medium", "stress_test"]
SEVERITY_LABELS  = {"baseline": "Baseline", "medium": "Medium", "stress_test": "Stress Test"}
MODEL_ORDER_DEV  = ["GPT-5.4 Nano", "Gemini 2.5 Flash Lite", "Mistral Small 3.2"]
MODEL_ORDER_PROD = ["Claude Haiku 4.5", "DeepSeek V3.2", "GPT-5.4 Mini", "Gemini 3 Flash Preview"]
ALL_MODELS       = MODEL_ORDER_DEV + MODEL_ORDER_PROD

# Active tags (appear in data)
TAG_COLS    = ["sycophancy", "over_solving", "over_medicalization", "dependency_fostering", "pain_romanticization"]
TAG_LABELS  = ["Sycophancy", "Over-Solving", "Over-Medicalization", "Dependency\nFostering", "Pain\nRomanticization"]
TAG_COLORS  = ["#e41a1c", "#ff7f00", "#4daf4a", "#984ea3", "#a65628"]

# All 8 rubric-defined tags (including absent ones)
ALL_TAG_COLS   = TAG_COLS + ["minimization", "hallucinated_resource", "toxic_positivity"]
ALL_TAG_LABELS = TAG_LABELS + ["Minimization", "Hallucinated\nResource", "Toxic\nPositivity"]
ALL_TAG_COLORS = TAG_COLORS + ["#377eb8", "#f781bf", "#999999"]


# ── Load ───────────────────────────────────────────────────────────────────────

def load_all(eval_dir: Path) -> pd.DataFrame:
    records = []
    for path in sorted(eval_dir.rglob("eval_*.json")):
        if any(p.startswith("_") for p in path.parts):
            continue
        model_tier = "dev" if "dev_models" in str(path) else "prod"
        if "human_checked" in str(path):
            input_len = "short"
        elif "long_input" in str(path):
            input_len = "long"
        else:
            input_len = "short"

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
                if s in ("NA", None): return np.nan
                try: return float(s)
                except: return np.nan

            h1, h2 = sc("response_attunement"), sc("sycophantic_accommodation")
            ev, as_, cr = sc("emotional_validation"), sc("actionable_support"), sc("cognitive_reframing")
            harm   = h1 + h2 if not (np.isnan(h1) or np.isnan(h2)) else np.nan
            help_s = np.nanmean([ev, as_, cr]) if not all(np.isnan([ev, as_, cr])) else np.nan

            records.append({
                "model_name":            model_name,
                "model_tier":            model_tier,
                "theme":                 theme,
                "severity_tier":         t.get("severity_tier"),
                "input_len":             input_len,
                # Active tags
                "sycophancy":            "sycophancy"            in tags,
                "over_solving":          "over_solving"           in tags,
                "over_medicalization":   "over_medicalization"    in tags,
                "dependency_fostering":  "dependency_fostering"   in tags,
                "pain_romanticization":  "pain_romanticization"   in tags,
                # Absent tags (tracked to confirm zero prevalence)
                "minimization":          "minimization"           in tags,
                "hallucinated_resource": "hallucinated_resource"  in tags,
                "toxic_positivity":      "toxic_positivity"       in tags,
                "any_tag":               len(tags) > 0,
                "harm_score":            harm,
                "help_score":            help_s,
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


# ── T0: All 8 tags overview ────────────────────────────────────────────────────

def plot_t0_overview(df: pd.DataFrame, out_dir: Path):
    """
    Show all 8 rubric-defined error tags in one figure.
    Left: overall prevalence bar chart (active vs absent tags).
    Right: explanation panel for the 3 absent tags.
    """
    rates = {col: df[col].mean() * 100 for col in ALL_TAG_COLS}

    fig, axes = plt.subplots(1, 2, figsize=(14, 5),
                             gridspec_kw={"width_ratios": [2, 1]})
    fig.suptitle("All 8 Rubric-Defined Error Tags: Prevalence Overview",
                 fontsize=13, fontweight="bold")

    # Left: bar chart
    ax = axes[0]
    labels = [l.replace("\n", " ") for l in ALL_TAG_LABELS]
    values = [rates[c] for c in ALL_TAG_COLS]
    bar_colors = [c if rates[col] > 0 else "#cccccc"
                  for col, c in zip(ALL_TAG_COLS, ALL_TAG_COLORS)]

    bars = ax.barh(labels[::-1], values[::-1], color=bar_colors[::-1],
                   edgecolor="white", linewidth=1.2)

    for bar, val in zip(bars, values[::-1]):
        label = f"{val:.1f}%" if val > 0 else "0%  (not observed)"
        ax.text(max(val + 0.5, 1), bar.get_y() + bar.get_height() / 2,
                label, va="center", fontsize=9,
                color="black" if val > 0 else "#888888")

    ax.axvline(0, color="black", linewidth=0.8)
    ax.set_xlabel("% of all turns", fontsize=10)
    ax.set_xlim(0, 75)
    ax.set_title("Observed Prevalence (4,032 turns)", fontsize=10)

    # Shade absent tags
    for i, (col, label) in enumerate(zip(ALL_TAG_COLS[::-1], labels[::-1])):
        if rates[ALL_TAG_COLS[::-1][i]] == 0:
            ax.axhspan(i - 0.4, i + 0.4, color="#f5f5f5", zorder=0)

    # Right: text explanation of absent tags
    ax2 = axes[1]
    ax2.axis("off")
    explanation = (
        "Three tags were never triggered:\n\n"
        "Minimization (0%)\n"
        "Under-responding to meaningful distress.\n"
        "These models consistently over-respond\n"
        "(sycophancy / over-solving) rather than\n"
        "under-respond — the opposite failure mode.\n\n"
        "Hallucinated Resource (0%)\n"
        "Fabricating hotlines or therapies.\n"
        "Models either gave no referral or gave\n"
        "plausible generic advice. Absence may\n"
        "reflect safety training, or that judge\n"
        "did not verify resource accuracy.\n\n"
        "Toxic Positivity (0%)\n"
        "Forced optimism denying negative emotion.\n"
        "Likely absorbed by 'sycophancy' — both\n"
        "involve uncritical accommodation, making\n"
        "them hard to distinguish in practice.\n"
        "May indicate rubric overlap."
    )
    ax2.text(0.05, 0.97, explanation, transform=ax2.transAxes,
             fontsize=8.5, va="top", ha="left",
             bbox=dict(boxstyle="round,pad=0.5", facecolor="#fff9c4", alpha=0.8),
             linespacing=1.5)

    plt.tight_layout()
    _save(fig, out_dir, "T0_all_tags_overview.png")

    print("\n=== T0 All tag rates ===")
    for col, label in zip(ALL_TAG_COLS, ALL_TAG_LABELS):
        rate = rates[col]
        status = "ABSENT" if rate == 0 else f"{rate:.2f}%"
        print(f"  {label.replace(chr(10),' '):25s}  {status}")


# ── T1: Model × Error Tag heatmap (all 8 tags) ────────────────────────────────

def plot_t1_model_heatmap(df: pd.DataFrame, out_dir: Path):
    models = [m for m in ALL_MODELS if m in df["model_name"].unique()]
    pivot  = df.groupby("model_name")[ALL_TAG_COLS].mean().mul(100).reindex(models)
    pivot.columns = [l.replace("\n", " ") for l in ALL_TAG_LABELS]

    fig, ax = plt.subplots(figsize=(13, 6))
    sns.heatmap(pivot, ax=ax, annot=True, fmt=".1f",
                cmap="YlOrRd", vmin=0, vmax=90,
                linewidths=0.5, cbar_kws={"label": "% of turns"})

    # Shade absent-tag columns
    absent_cols = ["Minimization", "Hallucinated Resource", "Toxic Positivity"]
    for i, col in enumerate(pivot.columns):
        if col in absent_cols:
            ax.add_patch(plt.Rectangle((i, 0), 1, len(models),
                         fill=True, color="#eeeeee", alpha=0.5, zorder=0))

    # Add tier divider
    ax.axhline(3, color="white", linewidth=3)
    ax.text(-0.4, 1.5, "Dev",  fontsize=9, color=PALETTE[0],
            fontweight="bold", ha="right", va="center", rotation=90)
    ax.text(-0.4, 5.0, "Prod", fontsize=9, color=PALETTE[1],
            fontweight="bold", ha="right", va="center", rotation=90)

    ax.set_title("Error Tag Rates by Model — All 8 Tags (%)", fontsize=13, fontweight="bold")
    ax.set_xlabel("")
    ax.set_ylabel("")
    ax.tick_params(axis="x", rotation=15, labelsize=9)
    ax.tick_params(axis="y", rotation=0,  labelsize=9)

    plt.tight_layout()
    _save(fig, out_dir, "T1_model_error_heatmap.png")


# ── T2: Theme × Error Tag heatmap ─────────────────────────────────────────────

def plot_t2_theme_heatmap(df: pd.DataFrame, out_dir: Path):
    pivot = df.groupby("theme")[TAG_COLS].mean().mul(100)
    pivot.columns = TAG_LABELS
    pivot = pivot.sort_values("Sycophancy", ascending=False)

    fig, ax = plt.subplots(figsize=(9, 5))
    sns.heatmap(pivot, ax=ax, annot=True, fmt=".1f",
                cmap="YlOrRd", vmin=0, vmax=75,
                linewidths=0.5, cbar_kws={"label": "% of turns"})

    ax.set_title("Error Tag Rates by Theme (%)", fontsize=13, fontweight="bold")
    ax.set_xlabel("")
    ax.set_ylabel("")
    ax.tick_params(axis="x", rotation=15, labelsize=9)
    ax.tick_params(axis="y", rotation=0,  labelsize=9)

    plt.tight_layout()
    _save(fig, out_dir, "T2_theme_error_heatmap.png")


# ── T3: Two failure modes — sycophancy vs over-solving scatter ────────────────

def plot_t3_failure_modes(df: pd.DataFrame, out_dir: Path):
    """
    Each model as a point: x=sycophancy rate, y=over_solving rate.
    Reveals two distinct failure profiles: high-sycophancy vs high-over-solving.
    """
    models = [m for m in ALL_MODELS if m in df["model_name"].unique()]
    palette = {m: PALETTE[i % len(PALETTE)] for i, m in enumerate(ALL_MODELS)}

    agg = df.groupby(["model_name", "model_tier"])[TAG_COLS].mean().mul(100).reset_index()
    agg = agg[agg["model_name"].isin(models)]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("Failure Mode Profiles: Sycophancy vs Over-Solving",
                 fontsize=13, fontweight="bold")

    tier_markers = {"dev": "^", "prod": "o"}

    # Left: sycophancy vs over_solving
    ax = axes[0]
    for _, row in agg.iterrows():
        marker = tier_markers.get(row["model_tier"], "o")
        ax.scatter(row["sycophancy"], row["over_solving"],
                   color=palette[row["model_name"]], s=220, zorder=3,
                   marker=marker, edgecolors="white", linewidths=1.5)
        ax.annotate(row["model_name"],
                    (row["sycophancy"], row["over_solving"]),
                    textcoords="offset points", xytext=(7, 3), fontsize=8)

    # Quadrant lines
    ax.axvline(50, color="gray", linestyle="--", alpha=0.4, linewidth=1)
    ax.axhline(20, color="gray", linestyle="--", alpha=0.4, linewidth=1)
    ax.text(5,  75, "Low syco\nHigh over-solve", fontsize=7.5, color="steelblue", alpha=0.7)
    ax.text(55, 75, "High syco\nHigh over-solve", fontsize=7.5, color="darkorange", alpha=0.7)
    ax.text(5,   2, "Low syco\nLow over-solve",  fontsize=7.5, color="green",     alpha=0.7)
    ax.text(55,  2, "High syco\nLow over-solve", fontsize=7.5, color="red",       alpha=0.7)

    ax.set_xlabel("Sycophancy Rate (%)", fontsize=10)
    ax.set_ylabel("Over-Solving Rate (%)", fontsize=10)
    ax.set_title("Primary Failure Mode Clusters", fontsize=10)
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)

    # Right: all 4 tags as grouped bars
    ax = axes[1]
    x  = np.arange(len(models))
    w  = 0.18
    for i, (col, label, color) in enumerate(zip(TAG_COLS, TAG_LABELS, TAG_COLORS)):
        vals = [agg[agg["model_name"] == m][col].values[0]
                if m in agg["model_name"].values else 0
                for m in models]
        offset = (i - 1.5) * w
        bars = ax.bar(x + offset, vals, w, label=label.replace("\n", " "),
                      color=color, edgecolor="white", linewidth=0.8, alpha=0.85)

    ax.set_xticks(x)
    ax.set_xticklabels(models, rotation=18, ha="right", fontsize=8)
    ax.set_ylabel("Error Tag Rate (%)", fontsize=10)
    ax.set_title("All Error Tags per Model", fontsize=10)
    ax.set_ylim(0, 100)
    ax.legend(fontsize=8, loc="upper right")

    # tier legend
    dev_m  = plt.Line2D([0],[0], marker="^", color="gray", linestyle="None",
                         markersize=9, label="Dev")
    prod_m = plt.Line2D([0],[0], marker="o", color="gray", linestyle="None",
                         markersize=9, label="Prod")
    axes[0].legend(handles=[dev_m, prod_m], fontsize=8, loc="upper left")

    plt.tight_layout()
    _save(fig, out_dir, "T3_failure_mode_profiles.png")

    # Print summary
    print("\n=== T3 Failure Mode Summary ===")
    print(agg[["model_name", "model_tier"] + TAG_COLS]
          .round(1)
          .sort_values("sycophancy")
          .to_string(index=False))


# ── T4: Error tag trajectory by severity ──────────────────────────────────────

def plot_t4_severity_trajectory(df: pd.DataFrame, out_dir: Path):
    valid    = df[df["severity_tier"].isin(SEVERITY_ORDER)].copy()
    x_labels = [SEVERITY_LABELS[t] for t in SEVERITY_ORDER]

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle("Error Tag Rates Across Severity Tiers (All Models)",
                 fontsize=13, fontweight="bold")

    for ax, tier, title in [
        (axes[0], "dev",  "Dev Models"),
        (axes[1], "prod", "Prod Models"),
    ]:
        sub = valid[valid["model_tier"] == tier]
        for col, label, color in zip(TAG_COLS, TAG_LABELS, TAG_COLORS):
            grp  = sub.groupby("severity_tier", observed=True)[col].mean() * 100
            vals = [grp.get(t, np.nan) for t in SEVERITY_ORDER]
            ax.plot(x_labels, vals, marker="o", label=label.replace("\n", " "),
                    color=color, linewidth=2.2, markersize=8)
            if not np.isnan(vals[-1]):
                ax.annotate(f"{vals[-1]:.1f}%",
                            (x_labels[-1], vals[-1]),
                            textcoords="offset points", xytext=(6, 0),
                            fontsize=8, color=color)

        ax.set_title(title, fontsize=10)
        ax.set_ylim(0, 95)
        ax.set_ylabel("Error Tag Rate (%)", fontsize=9)
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    _save(fig, out_dir, "T4_error_tags_by_severity.png")

    print("\n=== T4 Error tag rates by severity (all models) ===")
    print(valid.groupby("severity_tier", observed=True)[TAG_COLS]
          .mean().mul(100).round(1).to_string())


# ── T5: Error tags by input length per model ──────────────────────────────────

def plot_t5_input_length(df: pd.DataFrame, out_dir: Path):
    llm = df[df["input_len"].isin(["short", "long"])].copy()
    models = [m for m in ALL_MODELS if m in llm["model_name"].unique()]

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("Error Tag Rates: Short vs Long Input per Model",
                 fontsize=13, fontweight="bold")

    for ax, col, label, color in zip(axes.flat, TAG_COLS, TAG_LABELS, TAG_COLORS):
        x   = np.arange(len(models))
        w   = 0.35
        s_vals = [llm[(llm["model_name"] == m) & (llm["input_len"] == "short")][col].mean() * 100
                  for m in models]
        l_vals = [llm[(llm["model_name"] == m) & (llm["input_len"] == "long")][col].mean() * 100
                  for m in models]

        b1 = ax.bar(x - w/2, s_vals, w, label="Short", color="#fdc086",
                    edgecolor="white", linewidth=1.2)
        b2 = ax.bar(x + w/2, l_vals, w, label="Long",  color="#beaed4",
                    edgecolor="white", linewidth=1.2)

        for bar, val in zip(list(b1) + list(b2), s_vals + l_vals):
            if val > 1:
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                        f"{val:.0f}%", ha="center", va="bottom", fontsize=7.5)

        ax.set_xticks(x)
        ax.set_xticklabels(models, rotation=18, ha="right", fontsize=8)
        ax.set_title(label.replace("\n", " "), fontsize=10)
        ax.set_ylabel("Rate (%)", fontsize=8)
        ax.set_ylim(0, 100)
        ax.legend(fontsize=8)

    plt.tight_layout()
    _save(fig, out_dir, "T5_error_by_input_length.png")


# ── T6: Harm score by error tag combination ───────────────────────────────────

def plot_t6_harm_by_profile(df: pd.DataFrame, out_dir: Path):
    """
    What is the mean harm for each error tag pattern?
    Clean / syco only / over-solve only / both / etc.
    """
    df2 = df.copy()
    df2["profile"] = "clean (no tags)"
    df2.loc[df2["sycophancy"] & ~df2["over_solving"],  "profile"] = "sycophancy only"
    df2.loc[~df2["sycophancy"] & df2["over_solving"],  "profile"] = "over-solving only"
    df2.loc[df2["sycophancy"] & df2["over_solving"],   "profile"] = "both"
    df2.loc[df2["over_medicalization"],                "profile"] = df2.loc[df2["over_medicalization"], "profile"] + " + over-med"

    profile_order = [
        "clean (no tags)",
        "sycophancy only",
        "over-solving only",
        "both",
        "sycophancy only + over-med",
        "both + over-med",
    ]
    profile_order = [p for p in profile_order if p in df2["profile"].unique()]

    agg = df2.groupby("profile").agg(
        harm_mean=("harm_score", "mean"),
        help_mean=("help_score", "mean"),
        count=("harm_score",     "count"),
    ).reindex(profile_order).dropna()

    colors = ["#4daf4a", "#e41a1c", "#ff7f00", "#984ea3",
              "#a65628", "#f781bf"][:len(agg)]

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle("Mean Harm & Help Score by Error Tag Combination",
                 fontsize=13, fontweight="bold")

    for ax, col, title, ylim in [
        (axes[0], "harm_mean", "Harm Score (lower = better, 0–4)", (0, 4)),
        (axes[1], "help_mean", "Help Score (higher = better, 1–5)", (1, 5)),
    ]:
        bars = ax.barh(agg.index, agg[col], color=colors,
                       edgecolor="white", linewidth=1.2)
        for bar, (idx, row) in zip(bars, agg.iterrows()):
            ax.text(bar.get_width() + 0.03, bar.get_y() + bar.get_height()/2,
                    f"{row[col]:.2f}  (n={row['count']})",
                    va="center", fontsize=8)
        ax.set_xlim(*ylim)
        ax.set_title(title, fontsize=10)
        ax.axvline(df[col.replace("_mean","_score")].mean(), color="gray",
                   linestyle="--", alpha=0.5, linewidth=1)

    plt.tight_layout()
    _save(fig, out_dir, "T6_harm_by_error_profile.png")

    print("\n=== T6 Harm/Help by error profile ===")
    print(agg.round(3).to_string())


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--eval-dir", default="data/eval_results")
    parser.add_argument("--output",   default="analysis/figures/error_tags")
    args = parser.parse_args()

    eval_dir = Path(args.eval_dir)
    out_dir  = Path(args.output)

    print(f"Loading data from {eval_dir} ...")
    df = load_all(eval_dir)
    print(f"Loaded {len(df)} turns\n")

    print("Generating figures...")
    plot_t0_overview(df, out_dir)
    plot_t1_model_heatmap(df, out_dir)
    plot_t2_theme_heatmap(df, out_dir)
    plot_t3_failure_modes(df, out_dir)
    plot_t4_severity_trajectory(df, out_dir)
    plot_t5_input_length(df, out_dir)
    plot_t6_harm_by_profile(df, out_dir)

    print(f"\nDone. Saved to: {out_dir}")


if __name__ == "__main__":
    main()
