"""
Sycophancy Threshold Sensitivity Analysis.

Tests whether the sycophancy paradox (simultaneous Harm↑ Help↑) is robust
to different definitions of "sycophantic":

  T1: binary sycophancy error tag (original, prevalence ~59%)
  T2: sycophantic_accommodation >= 1 (any nonzero score, ~75%)
  T3: sycophantic_accommodation >= 2 (moderate-severe, ~0.2%)

Produces:
  ST1_threshold_comparison.png   — harm/help delta and prevalence across thresholds
  ST2_score_distribution.png     — sycophantic_accommodation score distribution
  ST3_moderate_zone.png          — harm/help profile broken down by exact score value

Usage:
  python analysis/sycophancy_threshold.py
"""

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


# ── Data loading ───────────────────────────────────────────────────────────────

def load_all(eval_dir: Path) -> pd.DataFrame:
    records = []
    for path in sorted(eval_dir.rglob("eval_*.json")):
        if any(p.startswith("_") for p in path.parts):
            continue
        with open(path, encoding="utf-8") as f:
            d = json.load(f)
        model = d.get("model_name", "unknown")
        tier  = "dev" if "dev_models" in str(path) else "prod"

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
            help_s = np.nanmean([ev, as_, cr])

            records.append({
                "model":    model,
                "tier":     tier,
                "syco_tag": "sycophancy" in tags,
                "syco_acc": h2,
                "harm":     harm,
                "help":     help_s,
            })

    return pd.DataFrame(records).dropna(subset=["harm", "help", "syco_acc"])


def _save(fig, out_dir: Path, name: str):
    out_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_dir / name, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {name}")


# ── ST1: Threshold comparison bar chart ───────────────────────────────────────

def plot_st1(df: pd.DataFrame, out_dir: Path):
    thresholds = [
        ("T1: Error tag\n(59.2%)",      df["syco_tag"] == True),
        ("T2: Score ≥ 1\n(74.9%)",      df["syco_acc"] >= 1),
        ("T3: Score ≥ 2\n(0.2%, n=8)",  df["syco_acc"] >= 2),
    ]

    results = []
    for label, mask in thresholds:
        syco = df[mask]
        non  = df[~mask]
        harm_t, harm_p = stats.ttest_ind(syco["harm"].dropna(), non["harm"].dropna())
        help_t, help_p = stats.ttest_ind(syco["help"].dropna(), non["help"].dropna())
        results.append({
            "label":       label,
            "n_syco":      len(syco),
            "prevalence":  len(syco) / len(df) * 100,
            "harm_syco":   syco["harm"].mean(),
            "harm_non":    non["harm"].mean(),
            "harm_delta":  syco["harm"].mean() - non["harm"].mean(),
            "harm_p":      harm_p,
            "help_syco":   syco["help"].mean(),
            "help_non":    non["help"].mean(),
            "help_delta":  syco["help"].mean() - non["help"].mean(),
            "help_p":      help_p,
        })

    rdf = pd.DataFrame(results)
    labels = rdf["label"].tolist()
    x = np.arange(len(labels))
    w = 0.35

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle("ST1 — Sycophancy Paradox: Robustness to Threshold Choice",
                 fontsize=12, fontweight="bold")

    # Panel 1: Harm delta
    colors_harm = ["#e74c3c" if d > 0 else "#2ecc71" for d in rdf["harm_delta"]]
    bars = axes[0].bar(labels, rdf["harm_delta"], color=colors_harm,
                       edgecolor="white", alpha=0.85)
    for bar, row in zip(bars, results):
        sig = "***" if row["harm_p"] < 0.001 else ("**" if row["harm_p"] < 0.01 else
              ("*" if row["harm_p"] < 0.05 else "n.s."))
        axes[0].text(bar.get_x() + bar.get_width()/2,
                     bar.get_height() + 0.01,
                     f"Δ={row['harm_delta']:+.3f}\np={row['harm_p']:.1e}\n{sig}",
                     ha="center", va="bottom", fontsize=7.5)
    axes[0].axhline(0, color="gray", linewidth=0.8, linestyle="--")
    axes[0].set_ylabel("Harm Score Δ (sycophantic − non)", fontsize=9)
    axes[0].set_title("Harm elevation across thresholds", fontsize=10)

    # Panel 2: Help delta
    colors_help = ["#e74c3c" if d > 0 else "#2ecc71" for d in rdf["help_delta"]]
    bars2 = axes[1].bar(labels, rdf["help_delta"], color=colors_help,
                        edgecolor="white", alpha=0.85)
    for bar, row in zip(bars2, results):
        sig = "***" if row["help_p"] < 0.001 else ("**" if row["help_p"] < 0.01 else
              ("*" if row["help_p"] < 0.05 else "n.s."))
        ypos = bar.get_height() + 0.002 if row["help_delta"] >= 0 else bar.get_height() - 0.012
        axes[1].text(bar.get_x() + bar.get_width()/2,
                     ypos,
                     f"Δ={row['help_delta']:+.3f}\np={row['help_p']:.1e}\n{sig}",
                     ha="center", va="bottom", fontsize=7.5)
    axes[1].axhline(0, color="gray", linewidth=0.8, linestyle="--")
    axes[1].set_ylabel("Help Score Δ (sycophantic − non)", fontsize=9)
    axes[1].set_title("Help elevation across thresholds\n(paradox = both Harm↑ AND Help↑)",
                      fontsize=10)

    # Panel 3: Prevalence + mean scores side by side
    axes[2].barh(labels, rdf["prevalence"], color=PALETTE[0], alpha=0.75,
                 edgecolor="white")
    for i, row in enumerate(results):
        axes[2].text(row["prevalence"] + 0.5, i,
                     f"{row['prevalence']:.1f}% (n={row['n_syco']})",
                     va="center", fontsize=8)
    axes[2].set_xlabel("Prevalence (%)", fontsize=9)
    axes[2].set_title("Sycophancy prevalence by threshold", fontsize=10)
    axes[2].set_xlim(0, 105)

    plt.tight_layout()
    _save(fig, out_dir, "ST1_threshold_comparison.png")

    print("\n=== Threshold summary ===")
    for row in results:
        print(f"  {row['label'].replace(chr(10),' ')}")
        print(f"    Prevalence: {row['prevalence']:.1f}% (n={row['n_syco']})")
        print(f"    Harm: syco={row['harm_syco']:.3f}  non={row['harm_non']:.3f}  "
              f"Δ={row['harm_delta']:+.3f}  p={row['harm_p']:.3e}")
        print(f"    Help: syco={row['help_syco']:.3f}  non={row['help_non']:.3f}  "
              f"Δ={row['help_delta']:+.3f}  p={row['help_p']:.3e}")


# ── ST2: Score distribution ────────────────────────────────────────────────────

def plot_st2(df: pd.DataFrame, out_dir: Path):
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle("ST2 — sycophantic_accommodation Score Distribution",
                 fontsize=12, fontweight="bold")

    # Overall distribution
    counts = df["syco_acc"].value_counts().sort_index()
    axes[0].bar(counts.index.astype(str), counts.values,
                color=PALETTE[3], edgecolor="white", alpha=0.85)
    for x, v in zip(range(len(counts)), counts.values):
        axes[0].text(x, v + 5, f"{v}\n({v/len(df)*100:.1f}%)",
                     ha="center", va="bottom", fontsize=9)
    axes[0].set_xlabel("sycophantic_accommodation score", fontsize=9)
    axes[0].set_ylabel("Number of turns", fontsize=9)
    axes[0].set_title(f"Full distribution (n={len(df)})", fontsize=10)

    # By sycophancy tag
    tag_counts = df.groupby(["syco_acc", "syco_tag"]).size().unstack(fill_value=0)
    tag_counts.plot(kind="bar", ax=axes[1], color=[PALETTE[1], PALETTE[3]],
                   edgecolor="white", alpha=0.85)
    axes[1].set_xlabel("sycophantic_accommodation score", fontsize=9)
    axes[1].set_ylabel("Number of turns", fontsize=9)
    axes[1].set_title("Score × sycophancy error tag", fontsize=10)
    axes[1].legend(["No tag", "Sycophancy tag"], fontsize=8)
    axes[1].tick_params(axis="x", rotation=0)

    plt.tight_layout()
    _save(fig, out_dir, "ST2_score_distribution.png")

    print("\n=== Score × tag cross-tab ===")
    print(df.groupby(["syco_acc", "syco_tag"]).size().to_string())


# ── ST3: Harm/help profile by exact score value ───────────────────────────────

def plot_st3(df: pd.DataFrame, out_dir: Path):
    score_vals = sorted(df["syco_acc"].unique())

    harm_means, harm_sems = [], []
    help_means, help_sems = [], []
    ns = []

    for v in score_vals:
        sub = df[df["syco_acc"] == v]
        harm_means.append(sub["harm"].mean())
        harm_sems.append(sub["harm"].sem())
        help_means.append(sub["help"].mean())
        help_sems.append(sub["help"].sem())
        ns.append(len(sub))

    fig, axes = plt.subplots(1, 2, figsize=(11, 5))
    fig.suptitle("ST3 — Harm & Help by Exact sycophantic_accommodation Score",
                 fontsize=12, fontweight="bold")

    colors = [PALETTE[1], PALETTE[2], PALETTE[3]]

    for ax, means, sems, ylabel, title in [
        (axes[0], harm_means, harm_sems, "Harm Score (0–8)", "Mean Harm by Score"),
        (axes[1], help_means, help_sems, "Help Score (1–5)", "Mean Help by Score"),
    ]:
        x = np.arange(len(score_vals))
        bars = ax.bar(x, means, yerr=sems, capsize=6,
                      color=colors[:len(score_vals)], edgecolor="white",
                      alpha=0.85)
        for bar, mean, n in zip(bars, means, ns):
            ax.text(bar.get_x() + bar.get_width()/2,
                    bar.get_height() + max(sems) + 0.02,
                    f"{mean:.3f}\n(n={n})",
                    ha="center", va="bottom", fontsize=8)
        ax.set_xticks(x)
        ax.set_xticklabels([f"Score = {int(v)}" for v in score_vals], fontsize=9)
        ax.set_ylabel(ylabel, fontsize=9)
        ax.set_title(title, fontsize=10)

    # Annotate the paradox interpretation
    axes[1].annotate(
        "Score=1: Help↑ (paradox zone)\nScore=2: Help↓ (n=8, n.s.)",
        xy=(0.5, 0.05), xycoords="axes fraction",
        fontsize=8, color="gray", ha="center",
        bbox=dict(boxstyle="round,pad=0.3", facecolor="lightyellow", alpha=0.7)
    )

    plt.tight_layout()
    _save(fig, out_dir, "ST3_moderate_zone.png")

    print("\n=== Harm/Help by exact score ===")
    for v, hm, hp, n in zip(score_vals, harm_means, help_means, ns):
        print(f"  Score={int(v)}: harm={hm:.3f}  help={hp:.3f}  n={n}")


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--eval-dir", default="data/eval_results")
    parser.add_argument("--output",   default="analysis/figures/sycophancy_threshold")
    args = parser.parse_args()

    eval_dir = Path(args.eval_dir)
    out_dir  = Path(args.output)

    print("Loading data...")
    df = load_all(eval_dir)
    print(f"  {len(df)} turns loaded")
    print(f"  syco_acc distribution: {df['syco_acc'].value_counts().sort_index().to_dict()}")
    print(f"  sycophancy tag rate: {df['syco_tag'].mean()*100:.1f}%\n")

    print("Generating figures...")
    plot_st1(df, out_dir)
    plot_st2(df, out_dir)
    plot_st3(df, out_dir)

    print(f"\nDone. Saved to: {out_dir}")


if __name__ == "__main__":
    main()
