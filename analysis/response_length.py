"""
Response Length Analysis.

Investigates whether response verbosity correlates with safety and quality.

Produces:
  L1_length_distribution.png     — box plots of response word count per model
  L2_length_vs_harm.png          — scatter + regression: length vs harm score
  L3_length_vs_help.png          — scatter + regression: length vs help score
  L4_length_by_severity.png      — does response length change with severity?
  L5_length_vs_sycophancy.png    — longer responses → more sycophantic?
  L6_consistency_vs_harm.png     — response length std (consistency) vs mean harm

Usage:
  python analysis/response_length.py
  python analysis/response_length.py --output analysis/figures/response_length
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

SEVERITY_ORDER  = ["baseline", "medium", "stress_test"]
SEVERITY_LABELS = {"baseline": "Baseline", "medium": "Medium", "stress_test": "Stress Test"}
MODEL_ORDER_DEV  = ["GPT-5.4 Nano", "Gemini 2.5 Flash Lite", "Mistral Small 3.2"]
MODEL_ORDER_PROD = ["Claude Haiku 4.5", "DeepSeek V3.2", "GPT-5.4 Mini", "Gemini 3 Flash Preview"]
ALL_MODELS = MODEL_ORDER_DEV + MODEL_ORDER_PROD


# ── Load ───────────────────────────────────────────────────────────────────────

def load_all(eval_dir: Path) -> pd.DataFrame:
    records = []
    for path in sorted(eval_dir.rglob("eval_*.json")):
        if any(p.startswith("_") for p in path.parts):
            continue
        model_tier = "dev" if "dev_models" in str(path) else "prod"

        with open(path, encoding="utf-8") as f:
            d = json.load(f)

        model_name = d.get("model_name", "unknown")

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
                "model_name":    model_name,
                "model_tier":    model_tier,
                "severity_tier": t.get("severity_tier"),
                "response_words": len(t.get("model_response", "").split()),
                "harm_score":    harm,
                "help_score":    help_s,
                "sycophancy":    "sycophancy" in tags,
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


# ── L1: Length distribution per model ─────────────────────────────────────────

def plot_l1_distribution(df: pd.DataFrame, out_dir: Path):
    models  = [m for m in ALL_MODELS if m in df["model_name"].unique()]
    palette = {m: PALETTE[i % len(PALETTE)] for i, m in enumerate(ALL_MODELS)}

    fig, ax = plt.subplots(figsize=(12, 5))
    sub = df[df["model_name"].isin(models)]

    sns.boxplot(data=sub, x="model_name", y="response_words",
                hue="model_name", order=models, palette=palette,
                ax=ax, width=0.5, linewidth=1.2, legend=False)

    # Annotate mean and std
    stats_df = sub.groupby("model_name")["response_words"].agg(["mean", "std"])
    for i, model in enumerate(models):
        if model in stats_df.index:
            mean = stats_df.loc[model, "mean"]
            std  = stats_df.loc[model, "std"]
            ax.text(i, sub[sub["model_name"] == model]["response_words"].max() + 10,
                    f"μ={mean:.0f}\nσ={std:.0f}",
                    ha="center", va="bottom", fontsize=7.5, color="#333")

    # Dev/prod separator
    ax.axvline(2.5, color="gray", linestyle="--", alpha=0.5, linewidth=1)
    ax.text(1.0, ax.get_ylim()[1] * 0.97, "Dev", ha="center", fontsize=9,
            color=PALETTE[0], fontweight="bold")
    ax.text(4.5, ax.get_ylim()[1] * 0.97, "Prod", ha="center", fontsize=9,
            color=PALETTE[1], fontweight="bold")

    ax.set_title("Response Length Distribution by Model", fontsize=13, fontweight="bold")
    ax.set_xlabel("")
    ax.set_ylabel("Response Word Count", fontsize=10)
    ax.tick_params(axis="x", rotation=18, labelsize=9)

    plt.tight_layout()
    _save(fig, out_dir, "L1_length_distribution.png")


# ── L2 & L3: Length vs Harm / Help scatter ────────────────────────────────────

def plot_l2_l3_scatter(df: pd.DataFrame, out_dir: Path):
    models  = [m for m in ALL_MODELS if m in df["model_name"].unique()]
    palette = {m: PALETTE[i % len(PALETTE)] for i, m in enumerate(ALL_MODELS)}
    sub     = df[df["model_name"].isin(models)].dropna(subset=["response_words", "harm_score", "help_score"])

    # Bin responses into 50-word buckets for cleaner visualization
    sub = sub.copy()
    sub["length_bin"] = (sub["response_words"] // 50) * 50

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("Response Length vs Harm & Help Scores", fontsize=13, fontweight="bold")

    for ax, col, ylabel, ylim, color in [
        (axes[0], "harm_score", "Harm Score (lower = better, 0–4)", (0, 4),   "#e41a1c"),
        (axes[1], "help_score", "Help Score (higher = better, 1–5)", (1, 5.2), "#377eb8"),
    ]:
        # Per-model mean per bin
        for model in models:
            msub = sub[sub["model_name"] == model]
            grp  = msub.groupby("length_bin")[col].mean().reset_index()
            ax.plot(grp["length_bin"], grp[col], "o-",
                    color=palette[model], alpha=0.7, linewidth=1.5,
                    markersize=5, label=model)

        # Overall regression line
        x_all = sub["response_words"].values
        y_all = sub[col].values
        mask  = ~np.isnan(y_all)
        slope, intercept, r, p, _ = stats.linregress(x_all[mask], y_all[mask])
        x_line = np.linspace(x_all.min(), x_all.max(), 100)
        ax.plot(x_line, slope * x_line + intercept,
                "--", color="black", linewidth=1.5, alpha=0.6,
                label=f"Overall trend (r={r:.2f}, p={p:.1e})")

        ax.set_xlabel("Response Word Count", fontsize=10)
        ax.set_ylabel(ylabel, fontsize=9)
        ax.set_ylim(*ylim)
        ax.legend(fontsize=7.5, loc="upper right" if col == "harm_score" else "lower right")

    plt.tight_layout()
    _save(fig, out_dir, "L2_L3_length_vs_scores.png")

    # Print stats
    print("\n=== Length vs Harm correlation (per model) ===")
    for model in models:
        msub = sub[sub["model_name"] == model].dropna(subset=["harm_score"])
        r, p = stats.pearsonr(msub["response_words"], msub["harm_score"])
        print(f"  {model:30s}  r={r:+.3f}  p={p:.2e}  mean_words={msub['response_words'].mean():.0f}")

    print("\n=== Length vs Help correlation (per model) ===")
    for model in models:
        msub = sub[sub["model_name"] == model].dropna(subset=["help_score"])
        r, p = stats.pearsonr(msub["response_words"], msub["help_score"])
        print(f"  {model:30s}  r={r:+.3f}  p={p:.2e}")


# ── L4: Response length by severity tier ──────────────────────────────────────

def plot_l4_severity(df: pd.DataFrame, out_dir: Path):
    valid    = df[df["severity_tier"].isin(SEVERITY_ORDER)].copy()
    x_labels = [SEVERITY_LABELS[t] for t in SEVERITY_ORDER]
    models   = [m for m in ALL_MODELS if m in df["model_name"].unique()]
    palette  = {m: PALETTE[i % len(PALETTE)] for i, m in enumerate(ALL_MODELS)}

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle("Response Length Across Severity Tiers",
                 fontsize=13, fontweight="bold")

    for ax, model_order, title in [
        (axes[0], MODEL_ORDER_DEV,  "Dev Models"),
        (axes[1], MODEL_ORDER_PROD, "Prod Models"),
    ]:
        tier_models = [m for m in model_order if m in df["model_name"].unique()]
        for model in tier_models:
            sub  = valid[valid["model_name"] == model]
            grp  = sub.groupby("severity_tier", observed=True)["response_words"].mean()
            vals = [grp.get(t, np.nan) for t in SEVERITY_ORDER]
            ax.plot(x_labels, vals, marker="o", label=model,
                    color=palette[model], linewidth=2.2, markersize=8)
            if not np.isnan(vals[-1]):
                ax.annotate(f"{vals[-1]:.0f}w",
                            (x_labels[-1], vals[-1]),
                            textcoords="offset points", xytext=(6, 0),
                            fontsize=8, color=palette[model])

        ax.set_title(title, fontsize=10)
        ax.set_ylabel("Mean Word Count", fontsize=9)
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    _save(fig, out_dir, "L4_length_by_severity.png")


# ── L5: Length vs Sycophancy ──────────────────────────────────────────────────

def plot_l5_sycophancy(df: pd.DataFrame, out_dir: Path):
    """Do longer responses have higher sycophancy rates?"""
    models  = [m for m in ALL_MODELS if m in df["model_name"].unique()]
    palette = {m: PALETTE[i % len(PALETTE)] for i, m in enumerate(ALL_MODELS)}

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle("Response Length vs Sycophancy", fontsize=13, fontweight="bold")

    # Left: sycophantic vs non-sycophantic word count distribution per model
    ax = axes[0]
    syco_means    = []
    nonsyco_means = []
    x = np.arange(len(models))
    w = 0.35

    for model in models:
        sub = df[df["model_name"] == model]
        syco_means.append(sub[sub["sycophancy"]]["response_words"].mean())
        nonsyco_means.append(sub[~sub["sycophancy"]]["response_words"].mean())

    b1 = ax.bar(x - w/2, syco_means,    w, color="#e41a1c", label="Sycophantic",
                edgecolor="white", linewidth=1.2, alpha=0.85)
    b2 = ax.bar(x + w/2, nonsyco_means, w, color="#377eb8", label="Non-sycophantic",
                edgecolor="white", linewidth=1.2, alpha=0.85)

    for bar, val in zip(list(b1) + list(b2), syco_means + nonsyco_means):
        if not np.isnan(val):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 3,
                    f"{val:.0f}", ha="center", va="bottom", fontsize=8)

    ax.set_xticks(x)
    ax.set_xticklabels(models, rotation=18, ha="right", fontsize=8)
    ax.set_ylabel("Mean Word Count", fontsize=10)
    ax.set_title("Word Count: Sycophantic vs Non-sycophantic", fontsize=10)
    ax.legend(fontsize=9)

    # Right: overall sycophancy rate binned by response length
    ax = axes[1]
    sub = df.copy()
    sub["length_bin"] = pd.cut(sub["response_words"],
                                bins=[0, 100, 150, 200, 250, 300, 400, 800],
                                labels=["<100", "100–150", "150–200",
                                        "200–250", "250–300", "300–400", ">400"])
    grp = sub.groupby("length_bin", observed=True)["sycophancy"].agg(["mean", "count"]).reset_index()
    grp["pct"] = grp["mean"] * 100

    colors = [PALETTE[0] if p < 55 else PALETTE[2] if p < 65 else "#e41a1c"
              for p in grp["pct"]]
    bars = ax.bar(grp["length_bin"].astype(str), grp["pct"],
                  color=colors, edgecolor="white", linewidth=1.2)

    for bar, row in zip(bars, grp.itertuples()):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                f"{row.pct:.1f}%\n(n={row.count})",
                ha="center", va="bottom", fontsize=7.5)

    ax.set_xlabel("Response Word Count (binned)", fontsize=10)
    ax.set_ylabel("Sycophancy Rate (%)", fontsize=10)
    ax.set_title("Sycophancy Rate by Response Length Bin", fontsize=10)
    ax.set_ylim(0, 95)

    plt.tight_layout()
    _save(fig, out_dir, "L5_length_vs_sycophancy.png")


# ── L6: Consistency (std) vs mean harm ────────────────────────────────────────

def plot_l6_consistency(df: pd.DataFrame, out_dir: Path):
    """Is response length consistency a proxy for safety consistency?"""
    models  = [m for m in ALL_MODELS if m in df["model_name"].unique()]
    palette = {m: PALETTE[i % len(PALETTE)] for i, m in enumerate(ALL_MODELS)}

    agg = df[df["model_name"].isin(models)].groupby("model_name").agg(
        mean_words=("response_words", "mean"),
        std_words =("response_words", "std"),
        mean_harm =("harm_score",     "mean"),
        model_tier=("model_tier",     "first"),
    ).reset_index()

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle("Response Length Characteristics vs Mean Harm Score",
                 fontsize=13, fontweight="bold")

    tier_markers = {"dev": "^", "prod": "o"}

    for ax, x_col, xlabel in [
        (axes[0], "mean_words", "Mean Response Word Count"),
        (axes[1], "std_words",  "Response Length Std Dev (consistency)"),
    ]:
        for _, row in agg.iterrows():
            marker = tier_markers.get(row["model_tier"], "o")
            ax.scatter(row[x_col], row["mean_harm"],
                       color=palette[row["model_name"]], s=180, zorder=3,
                       marker=marker, edgecolors="white", linewidths=1.5)
            ax.annotate(row["model_name"],
                        (row[x_col], row["mean_harm"]),
                        textcoords="offset points", xytext=(7, 3), fontsize=8)

        # Regression
        x_vals = agg[x_col].values
        y_vals = agg["mean_harm"].values
        slope, intercept, r, p, _ = stats.linregress(x_vals, y_vals)
        x_line = np.linspace(x_vals.min(), x_vals.max(), 100)
        ax.plot(x_line, slope * x_line + intercept,
                "--", color="gray", linewidth=1.5, alpha=0.7,
                label=f"r={r:.2f}, p={p:.2f}")

        ax.set_xlabel(xlabel, fontsize=10)
        ax.set_ylabel("Mean Harm Score", fontsize=10)
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)

    # Legend for tier markers
    dev_patch  = plt.Line2D([0], [0], marker="^", color="gray", linestyle="None",
                             markersize=9, label="Dev")
    prod_patch = plt.Line2D([0], [0], marker="o", color="gray", linestyle="None",
                             markersize=9, label="Prod")
    axes[1].legend(handles=[dev_patch, prod_patch] +
                   [plt.Line2D([0],[0],linestyle="--",color="gray",
                                label=f"r={stats.linregress(agg['std_words'].values, agg['mean_harm'].values)[2]:.2f}")],
                   fontsize=8)

    plt.tight_layout()
    _save(fig, out_dir, "L6_consistency_vs_harm.png")

    print("\n=== L6 Per-model length stats ===")
    print(agg[["model_name", "model_tier", "mean_words", "std_words", "mean_harm"]]
          .sort_values("mean_harm")
          .round(1)
          .to_string(index=False))


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--eval-dir", default="data/eval_results")
    parser.add_argument("--output",   default="analysis/figures/response_length")
    args = parser.parse_args()

    eval_dir = Path(args.eval_dir)
    out_dir  = Path(args.output)

    print(f"Loading data from {eval_dir} ...")
    df = load_all(eval_dir)
    print(f"Loaded {len(df)} turns\n")

    print("Generating figures...")
    plot_l1_distribution(df, out_dir)
    plot_l2_l3_scatter(df, out_dir)
    plot_l4_severity(df, out_dir)
    plot_l5_sycophancy(df, out_dir)
    plot_l6_consistency(df, out_dir)

    print(f"\nDone. Saved to: {out_dir}")


if __name__ == "__main__":
    main()
