"""
Severity Escalation Analysis.

Investigates how model behavior changes as mental health scenario severity
escalates from baseline → medium → stress_test.

Produces:
  E1_harm_help_escalation.png   — harm & help line charts per model, dev/prod split
  E2_harm_delta.png             — harm increase (stress_test - baseline) per model
  E3_sycophancy_escalation.png  — sycophancy rate escalation per model
  E4_referral_at_stress.png     — professional referral rate at stress_test
  E5_help_submetics.png         — help sub-metric composition across severity tiers
  E6_error_tags_by_severity.png — error tag stacked bars per severity tier

Usage:
  python analysis/severity.py
  python analysis/severity.py --output analysis/figures/severity
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

SEVERITY_ORDER  = ["baseline", "medium", "stress_test"]
SEVERITY_LABELS = {"baseline": "Baseline", "medium": "Medium", "stress_test": "Stress Test"}

MODEL_ORDER_DEV  = ["GPT-5.4 Nano", "Gemini 2.5 Flash Lite", "Mistral Small 3.2"]
MODEL_ORDER_PROD = ["Claude Haiku 4.5", "DeepSeek V3.2", "GPT-5.4 Mini", "Gemini 3 Flash Preview"]
ALL_MODELS       = MODEL_ORDER_DEV + MODEL_ORDER_PROD


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

            h1, h2 = sc("response_attunement"), sc("sycophantic_accommodation")
            ev, as_, cr = sc("emotional_validation"), sc("actionable_support"), sc("cognitive_reframing")
            harm   = h1 + h2 if not (np.isnan(h1) or np.isnan(h2)) else np.nan
            help_s = np.nanmean([ev, as_, cr]) if not all(np.isnan([ev, as_, cr])) else np.nan

            ref     = m.get("professional_referral", {})
            ref_val = ref.get("score") if isinstance(ref, dict) else ref

            records.append({
                "model_name":            model_name,
                "model_tier":            model_tier,
                "theme":                 theme,
                "severity_tier":         t.get("severity_tier"),
                "harm_score":            harm,
                "help_score":            help_s,
                "emotional_validation":  ev,
                "actionable_support":    as_,
                "cognitive_reframing":   cr,
                "boundary_appropriateness": sc("boundary_appropriateness"),
                "sycophancy":            "sycophancy" in tags,
                "over_solving":          "over_solving" in tags,
                "over_medicalization":   "over_medicalization" in tags,
                "referral_score":        str(ref_val) if ref_val is not None else "NA",
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


# ── E1: Harm & Help escalation lines, dev/prod split ─────────────────────────

def plot_e1_escalation(df: pd.DataFrame, out_dir: Path):
    valid = df[df["severity_tier"].isin(SEVERITY_ORDER)].copy()
    x_labels = [SEVERITY_LABELS[t] for t in SEVERITY_ORDER]

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("Harm & Help Score Escalation Across Severity Tiers",
                 fontsize=13, fontweight="bold")

    configs = [
        (axes[0][0], MODEL_ORDER_DEV,  "harm_score", "Dev — Harm Score (lower = better)", (0, 4)),
        (axes[0][1], MODEL_ORDER_PROD, "harm_score", "Prod — Harm Score (lower = better)", (0, 4)),
        (axes[1][0], MODEL_ORDER_DEV,  "help_score", "Dev — Help Score (higher = better)", (2, 5)),
        (axes[1][1], MODEL_ORDER_PROD, "help_score", "Prod — Help Score (higher = better)", (2, 5)),
    ]

    for ax, model_order, col, title, ylim in configs:
        models = [m for m in model_order if m in df["model_name"].unique()]
        colors = {m: PALETTE[i % len(PALETTE)] for i, m in enumerate(models)}

        for model in models:
            sub  = valid[valid["model_name"] == model]
            grp  = sub.groupby("severity_tier", observed=True)[col].mean()
            vals = [grp.get(t, np.nan) for t in SEVERITY_ORDER]
            ax.plot(x_labels, vals, marker="o", label=model,
                    color=colors[model], linewidth=2.2, markersize=8)

            # annotate last point
            if not np.isnan(vals[-1]):
                ax.annotate(f"{vals[-1]:.2f}", (x_labels[-1], vals[-1]),
                            textcoords="offset points", xytext=(6, 0),
                            fontsize=8, color=colors[model])

        ax.set_title(title, fontsize=10)
        ax.set_ylim(*ylim)
        ax.set_xlabel("")
        ax.set_ylabel("Score", fontsize=9)
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    _save(fig, out_dir, "E1_harm_help_escalation.png")


# ── E2: Harm delta (stress_test - baseline) ───────────────────────────────────

def plot_e2_harm_delta(df: pd.DataFrame, out_dir: Path):
    """Who degrades most from baseline to stress_test?"""
    models = [m for m in ALL_MODELS if m in df["model_name"].unique()]

    deltas = []
    for model in models:
        sub  = df[df["model_name"] == model]
        base = sub[sub["severity_tier"] == "baseline"]["harm_score"].mean()
        st   = sub[sub["severity_tier"] == "stress_test"]["harm_score"].mean()
        tier = df[df["model_name"] == model]["model_tier"].iloc[0]
        deltas.append({"model": model, "tier": tier, "delta": st - base,
                       "base": base, "stress": st})

    delta_df = pd.DataFrame(deltas).sort_values("delta", ascending=False)

    colors = [PALETTE[0] if t == "dev" else PALETTE[1] for t in delta_df["tier"]]

    fig, ax = plt.subplots(figsize=(10, 5))
    bars = ax.barh(delta_df["model"], delta_df["delta"],
                   color=colors, edgecolor="white", linewidth=1.2)

    for bar, row in zip(bars, delta_df.itertuples()):
        label = f"+{row.delta:.2f}" if row.delta >= 0 else f"{row.delta:.2f}"
        ax.text(row.delta + 0.01, bar.get_y() + bar.get_height()/2,
                label, va="center", fontsize=9)

    ax.axvline(0, color="black", linewidth=0.8)
    ax.set_xlabel("Harm Score Increase (Stress Test − Baseline)", fontsize=10)
    ax.set_title("Harm Degradation from Baseline to Stress Test",
                 fontsize=12, fontweight="bold")

    import matplotlib.patches as mpatches
    patches = [mpatches.Patch(color=PALETTE[0], label="Dev"),
               mpatches.Patch(color=PALETTE[1], label="Prod")]
    ax.legend(handles=patches, fontsize=9)

    plt.tight_layout()
    _save(fig, out_dir, "E2_harm_delta.png")

    print("\n=== E2 Harm Delta (stress_test - baseline) ===")
    print(delta_df[["model", "tier", "base", "stress", "delta"]].round(3).to_string(index=False))


# ── E3: Sycophancy escalation per model ───────────────────────────────────────

def plot_e3_sycophancy(df: pd.DataFrame, out_dir: Path):
    valid    = df[df["severity_tier"].isin(SEVERITY_ORDER)].copy()
    x_labels = [SEVERITY_LABELS[t] for t in SEVERITY_ORDER]

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle("Sycophancy Rate Escalation Across Severity Tiers",
                 fontsize=13, fontweight="bold")

    for ax, model_order, title in [
        (axes[0], MODEL_ORDER_DEV,  "Dev Models"),
        (axes[1], MODEL_ORDER_PROD, "Prod Models"),
    ]:
        models = [m for m in model_order if m in df["model_name"].unique()]
        colors = {m: PALETTE[i % len(PALETTE)] for i, m in enumerate(models)}

        for model in models:
            sub  = valid[valid["model_name"] == model]
            grp  = sub.groupby("severity_tier", observed=True)["sycophancy"].mean() * 100
            vals = [grp.get(t, np.nan) for t in SEVERITY_ORDER]
            ax.plot(x_labels, vals, marker="o", label=model,
                    color=colors[model], linewidth=2.2, markersize=8)

        ax.set_title(title, fontsize=10)
        ax.set_ylim(0, 100)
        ax.set_ylabel("Sycophancy Rate (%)", fontsize=9)
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    # Annotate overall trend
    overall = df[df["severity_tier"].isin(SEVERITY_ORDER)].groupby(
        "severity_tier", observed=True)["sycophancy"].mean() * 100
    axes[1].text(0.98, 0.05,
                 f"Overall: {overall.get('baseline',0):.1f}% → "
                 f"{overall.get('medium',0):.1f}% → "
                 f"{overall.get('stress_test',0):.1f}%",
                 transform=axes[1].transAxes, ha="right", va="bottom",
                 fontsize=8, bbox=dict(boxstyle="round,pad=0.3",
                                       facecolor="#fff9c4", alpha=0.8))

    plt.tight_layout()
    _save(fig, out_dir, "E3_sycophancy_escalation.png")


# ── E4: Professional referral at stress_test ──────────────────────────────────

def plot_e4_referral(df: pd.DataFrame, out_dir: Path):
    """At stress_test: what fraction of models recommend professional help?"""
    st     = df[df["severity_tier"] == "stress_test"].copy()
    models = [m for m in ALL_MODELS if m in st["model_name"].unique()]

    referral_2   = []   # correctly referred
    referral_na  = []   # didn't refer (NA = not recommended)
    referral_miss = []  # score 0 or 1 = missed/partial

    for model in models:
        sub   = st[st["model_name"] == model]
        total = len(sub)
        r2    = (sub["referral_score"] == "2").sum() / total * 100
        rna   = (sub["referral_score"] == "NA").sum() / total * 100
        rmiss = (sub["referral_score"].isin(["0", "1"])).sum() / total * 100
        referral_2.append(r2)
        referral_na.append(rna)
        referral_miss.append(rmiss)

    x = np.arange(len(models))
    tier_colors = [PALETTE[0] if df[df["model_name"] == m]["model_tier"].iloc[0] == "dev"
                   else PALETTE[1] for m in models]

    fig, ax = plt.subplots(figsize=(11, 5))
    b1 = ax.bar(x, referral_2,   color="#4daf4a", label="Referred (score=2)", edgecolor="white")
    b2 = ax.bar(x, referral_na,  bottom=referral_2, color="#aec7e8",
                label="Not referred (NA)", edgecolor="white")
    b3 = ax.bar(x, referral_miss, bottom=np.array(referral_2) + np.array(referral_na),
                color="#e41a1c", label="Missed / partial (0 or 1)", edgecolor="white")

    # annotate referral rate
    for i, val in enumerate(referral_2):
        ax.text(i, val / 2, f"{val:.0f}%", ha="center", va="center",
                fontsize=9, color="white", fontweight="bold")

    ax.set_xticks(x)
    ax.set_xticklabels(models, rotation=18, ha="right", fontsize=9)
    ax.set_ylabel("% of turns", fontsize=10)
    ax.set_ylim(0, 105)
    ax.set_title("Professional Referral Behavior at Stress Test Level",
                 fontsize=12, fontweight="bold")
    ax.legend(fontsize=9)

    plt.tight_layout()
    _save(fig, out_dir, "E4_referral_at_stress.png")

    print("\n=== E4 Referral at stress_test ===")
    for m, r2, rna, rm in zip(models, referral_2, referral_na, referral_miss):
        print(f"{m:30s}  referred={r2:.1f}%  not-referred={rna:.1f}%  missed={rm:.1f}%")


# ── E5: Help sub-metrics by severity ──────────────────────────────────────────

def plot_e5_submetrics(df: pd.DataFrame, out_dir: Path):
    """How does the composition of help change across severity?"""
    valid    = df[df["severity_tier"].isin(SEVERITY_ORDER)].copy()
    x_labels = [SEVERITY_LABELS[t] for t in SEVERITY_ORDER]
    metrics  = [
        ("emotional_validation", "Emotional Validation", "#e41a1c"),
        ("actionable_support",   "Actionable Support",   "#377eb8"),
        ("cognitive_reframing",  "Cognitive Reframing",  "#4daf4a"),
    ]

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle("Help Sub-Metric Composition Across Severity Tiers",
                 fontsize=13, fontweight="bold")

    for ax, tier in [(axes[0], "dev"), (axes[1], "prod")]:
        sub = valid[valid["model_tier"] == tier]
        for col, label, color in metrics:
            grp  = sub.groupby("severity_tier", observed=True)[col].mean()
            vals = [grp.get(t, np.nan) for t in SEVERITY_ORDER]
            ax.plot(x_labels, vals, marker="o", label=label,
                    color=color, linewidth=2.2, markersize=8)
            ax.annotate(f"{vals[-1]:.2f}", (x_labels[-1], vals[-1]),
                        textcoords="offset points", xytext=(6, 0), fontsize=8, color=color)

        ax.set_title(f"{'Dev' if tier == 'dev' else 'Prod'} Models", fontsize=10)
        ax.set_ylim(2.5, 5.5)
        ax.set_ylabel("Mean Score (1–5)", fontsize=9)
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    _save(fig, out_dir, "E5_help_submetrics.png")


# ── E6: Error tags by severity ────────────────────────────────────────────────

def plot_e6_error_tags(df: pd.DataFrame, out_dir: Path):
    valid   = df[df["severity_tier"].isin(SEVERITY_ORDER)].copy()
    tags    = ["sycophancy", "over_solving", "over_medicalization"]
    labels  = ["Sycophancy", "Over-Solving", "Over-Medicalization"]
    colors  = ["#e41a1c", "#ff7f00", "#4daf4a"]

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle("Error Tag Rates Across Severity Tiers",
                 fontsize=13, fontweight="bold")

    for ax, tier in [(axes[0], "dev"), (axes[1], "prod")]:
        sub = valid[valid["model_tier"] == tier]
        x   = np.arange(len(SEVERITY_ORDER))
        w   = 0.25

        for i, (col, label, color) in enumerate(zip(tags, labels, colors)):
            vals = [sub[sub["severity_tier"] == t][col].mean() * 100
                    for t in SEVERITY_ORDER]
            ax.bar(x + (i - 1) * w, vals, w, label=label, color=color,
                   edgecolor="white", linewidth=0.8, alpha=0.85)

        ax.set_xticks(x)
        ax.set_xticklabels([SEVERITY_LABELS[t] for t in SEVERITY_ORDER], fontsize=9)
        ax.set_ylabel("% of turns", fontsize=9)
        ax.set_title(f"{'Dev' if tier == 'dev' else 'Prod'} Models", fontsize=10)
        ax.set_ylim(0, 100)
        ax.legend(fontsize=8)

    plt.tight_layout()
    _save(fig, out_dir, "E6_error_tags_by_severity.png")


# ── Summary stats ──────────────────────────────────────────────────────────────

def print_summary(df: pd.DataFrame):
    valid = df[df["severity_tier"].isin(SEVERITY_ORDER)]
    print("\n=== Overall escalation (all models) ===")
    print(valid.groupby("severity_tier", observed=True)[
        ["harm_score", "help_score", "sycophancy"]
    ].mean().round(3).to_string())

    print("\n=== Per-model harm at each severity tier ===")
    pivot = valid.groupby(["model_name", "severity_tier"], observed=True)[
        "harm_score"
    ].mean().unstack("severity_tier").round(3)
    pivot["delta"] = (pivot.get("stress_test", 0) - pivot.get("baseline", 0)).round(3)
    print(pivot.to_string())


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--eval-dir", default="data/eval_results")
    parser.add_argument("--output",   default="analysis/figures/severity")
    args = parser.parse_args()

    eval_dir = Path(args.eval_dir)
    out_dir  = Path(args.output)

    print(f"Loading data from {eval_dir} ...")
    df = load_all(eval_dir)
    print(f"Loaded {len(df)} turns\n")

    print_summary(df)
    print("\nGenerating figures...")
    plot_e1_escalation(df, out_dir)
    plot_e2_harm_delta(df, out_dir)
    plot_e3_sycophancy(df, out_dir)
    plot_e4_referral(df, out_dir)
    plot_e5_submetrics(df, out_dir)
    plot_e6_error_tags(df, out_dir)

    print(f"\nDone. Saved to: {out_dir}")


if __name__ == "__main__":
    main()
