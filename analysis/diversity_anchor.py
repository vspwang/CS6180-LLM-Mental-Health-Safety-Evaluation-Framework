"""
Diversity Anchor Analysis.

LLM-generated scenarios are constructed with a two-dimensional diversity anchor:
  - Opening type  (7 styles): how the speaker begins their message
  - Temporal context (3 types): how long the issue has been present

Investigates whether these narrative dimensions affect model safety behavior,
independently of theme or severity.

Produces:
  D1_opening_type_scores.png       — harm/help/sycophancy by opening type
  D2_temporal_context_scores.png   — harm/help/sycophancy by temporal context
  D3_anchor_heatmap.png            — opening × temporal heatmap for harm & sycophancy
  D4_opening_error_tags.png        — error tag rates by opening type
  D5_temporal_by_model.png         — temporal context effect per model (prod)
  D6_opening_by_theme.png          — does opening type interact with theme?

Usage:
  python analysis/diversity_anchor.py
  python analysis/diversity_anchor.py --output analysis/figures/diversity_anchor
"""

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats

sns.set_theme(style="whitegrid", font_scale=1.05)
PALETTE = sns.color_palette("Set2")

SEVERITY_ORDER   = ["baseline", "medium", "stress_test"]
MODEL_ORDER_PROD = ["Claude Haiku 4.5", "DeepSeek V3.2", "GPT-5.4 Mini", "Gemini 3 Flash Preview"]

# Short labels for opening types
OPENING_LABELS = {
    "by referencing a specific recent moment":       "Recent\nmoment",
    "by contrasting with how things used to be":     "Contrast\nwith past",
    "with something they noticed about themselves":  "Self-\nnoticed",
    "with something someone else said or did":       "Other\nperson",
    "with a failed attempt to feel better":          "Failed\nattempt",
    "by describing a physical feeling":              "Physical\nfeeling",
    "with a conclusion they've reached about themselves": "Self-\nconclusion",
}

TEMPORAL_LABELS = {
    "This has been going on for a while.":     "Chronic\n(ongoing)",
    "This just started recently.":             "Acute\n(recent onset)",
    "This used to not bother the speaker.":    "Shifted\n(new sensitivity)",
}


# ── Build anchor lookup ────────────────────────────────────────────────────────

def build_anchor_map(stimuli_dir: Path) -> dict:
    anchor_map = {}
    for path in stimuli_dir.rglob("*.json"):
        if "_metadata" in str(path):
            continue
        with open(path, encoding="utf-8") as f:
            d = json.load(f)
        meta = d.get("generation_metadata", {})
        anchor = meta.get("diversity_anchor", "")
        if not anchor:
            continue
        parts = anchor.split(". ")
        opening  = parts[0].replace("The speaker opens ", "").strip()
        temporal = parts[1].strip() if len(parts) > 1 else None
        anchor_map[d["stimulus_id"]] = {
            "anchor":           anchor,
            "opening_type":     opening,
            "opening_label":    OPENING_LABELS.get(opening, opening),
            "temporal_context": temporal,
            "temporal_label":   TEMPORAL_LABELS.get(temporal, temporal) if temporal else "No temporal",
        }
    return anchor_map


# ── Load eval results ──────────────────────────────────────────────────────────

def load_all(eval_dir: Path, anchor_map: dict) -> pd.DataFrame:
    records = []
    llm_dir = eval_dir / "llm_generated_scenarios"

    for path in sorted(llm_dir.rglob("eval_*.json")):
        if any(p.startswith("_") for p in path.parts):
            continue
        model_tier = "dev" if "dev_models" in str(path) else "prod"
        input_len  = "long" if "long_input" in str(path) else "short"

        with open(path, encoding="utf-8") as f:
            d = json.load(f)

        sid        = d.get("stimulus_id", "")
        model_name = d.get("model_name", "unknown")
        theme      = d.get("theme", "unknown")

        if sid not in anchor_map:
            continue
        anchor_info = anchor_map[sid]

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
                "model_name":      model_name,
                "model_tier":      model_tier,
                "theme":           theme,
                "input_len":       input_len,
                "severity_tier":   t.get("severity_tier"),
                "stimulus_id":     sid,
                "opening_type":    anchor_info["opening_type"],
                "opening_label":   anchor_info["opening_label"],
                "temporal_context":anchor_info["temporal_context"],
                "temporal_label":  anchor_info["temporal_label"],
                "harm_score":      harm,
                "help_score":      help_s,
                "sycophancy":      "sycophancy"          in tags,
                "over_solving":    "over_solving"         in tags,
                "over_medicalization": "over_medicalization" in tags,
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


# ── D1: Opening type → harm / help / sycophancy ───────────────────────────────

def plot_d1_opening(df: pd.DataFrame, out_dir: Path):
    order  = sorted(df["opening_label"].unique())
    colors = sns.color_palette("tab10", len(order))

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    fig.suptitle("Opening Type Effect on Model Responses",
                 fontsize=13, fontweight="bold")

    metrics = [
        ("harm_score",  "Harm Score (0–4, lower = better)", (0, 3)),
        ("help_score",  "Help Score (1–5, higher = better)", (2, 5)),
        ("sycophancy",  "Sycophancy Rate (%)", (0, 90)),
    ]

    agg = df.groupby("opening_label").agg(
        harm_mean   =("harm_score",  "mean"),
        harm_se     =("harm_score",  "sem"),
        help_mean   =("help_score",  "mean"),
        help_se     =("help_score",  "sem"),
        syco_mean   =("sycophancy",  "mean"),
        syco_se     =("sycophancy",  "sem"),
        n           =("harm_score",  "count"),
    ).reindex(order)

    for ax, (col, ylabel, ylim) in zip(axes, metrics):
        mean_col = col.replace("_score","_mean").replace("sycophancy","syco_mean")
        se_col   = col.replace("_score","_se").replace("sycophancy","syco_se")
        vals = agg[mean_col].values * (100 if col == "sycophancy" else 1)
        errs = agg[se_col].values  * (100 if col == "sycophancy" else 1)

        bars = ax.bar(order, vals, yerr=errs, capsize=4,
                      color=colors, edgecolor="white", linewidth=1.2, alpha=0.85)
        for bar, val in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + errs[list(vals).index(val)] + 0.02,
                    f"{val:.2f}" if col != "sycophancy" else f"{val:.1f}%",
                    ha="center", va="bottom", fontsize=7.5)

        # Overall mean line
        overall = df[col].mean() * (100 if col == "sycophancy" else 1)
        ax.axhline(overall, color="gray", linestyle="--", linewidth=1,
                   alpha=0.6, label=f"Overall mean ({overall:.2f})")
        ax.set_ylabel(ylabel, fontsize=9)
        ax.set_ylim(*ylim)
        ax.tick_params(axis="x", rotation=15, labelsize=7.5)
        ax.legend(fontsize=7)

    plt.tight_layout()
    _save(fig, out_dir, "D1_opening_type_scores.png")

    print("\n=== D1 Opening type stats ===")
    print(agg[["harm_mean","help_mean","syco_mean","n"]].round(3).to_string())


# ── D2: Temporal context → harm / help / sycophancy ──────────────────────────

def plot_d2_temporal(df: pd.DataFrame, out_dir: Path):
    temporal_order = ["Chronic\n(ongoing)", "Acute\n(recent onset)", "Shifted\n(new sensitivity)"]
    temporal_order = [t for t in temporal_order if t in df["temporal_label"].unique()]
    colors = [PALETTE[0], PALETTE[1], PALETTE[2]]

    fig, axes = plt.subplots(1, 3, figsize=(13, 5))
    fig.suptitle("Temporal Context Effect: Chronic vs Acute vs Shifted",
                 fontsize=13, fontweight="bold")

    metrics = [
        ("harm_score", "Harm Score (0–4)", (0, 2.5)),
        ("help_score", "Help Score (1–5)", (3, 5)),
        ("sycophancy", "Sycophancy Rate (%)", (0, 80)),
    ]

    for ax, (col, ylabel, ylim) in zip(axes, metrics):
        vals, errs, ns = [], [], []
        for tl in temporal_order:
            sub = df[df["temporal_label"] == tl][col].dropna()
            vals.append(sub.mean() * (100 if col == "sycophancy" else 1))
            errs.append(sub.sem()  * (100 if col == "sycophancy" else 1))
            ns.append(len(sub))

        bars = ax.bar(temporal_order, vals, yerr=errs, capsize=5,
                      color=colors[:len(temporal_order)],
                      edgecolor="white", linewidth=1.2, alpha=0.85)
        for bar, val, n in zip(bars, vals, ns):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + errs[list(vals).index(val)] + 0.02,
                    f"{val:.2f}\n(n={n})" if col != "sycophancy" else f"{val:.1f}%\n(n={n})",
                    ha="center", va="bottom", fontsize=8)

        overall = df[col].mean() * (100 if col == "sycophancy" else 1)
        ax.axhline(overall, color="gray", linestyle="--", linewidth=1, alpha=0.6)
        ax.set_ylabel(ylabel, fontsize=9)
        ax.set_ylim(*ylim)
        ax.tick_params(axis="x", labelsize=9)

    plt.tight_layout()
    _save(fig, out_dir, "D2_temporal_context_scores.png")

    # Stats
    print("\n=== D2 Temporal context stats ===")
    for col in ["harm_score", "help_score", "sycophancy"]:
        groups = [df[df["temporal_label"] == tl][col].dropna() for tl in temporal_order]
        f, p = stats.f_oneway(*[g for g in groups if len(g) > 0])
        means = [f"{g.mean():.3f}" for g in groups]
        print(f"  {col}: {' | '.join(means)}  ANOVA F={f:.2f} p={p:.3e}")


# ── D3: Opening × Temporal heatmap ────────────────────────────────────────────

def plot_d3_heatmap(df: pd.DataFrame, out_dir: Path):
    temporal_order = ["Chronic\n(ongoing)", "Acute\n(recent onset)", "Shifted\n(new sensitivity)"]
    temporal_order = [t for t in temporal_order if t in df["temporal_label"].unique()]
    opening_order  = sorted(df["opening_label"].unique())

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle("Opening Type × Temporal Context: Harm & Sycophancy Heatmap",
                 fontsize=13, fontweight="bold")

    for ax, col, title, cmap, vmin, vmax, scale in [
        (axes[0], "harm_score", "Harm Score",      "YlOrRd", 0,  2.5, 1),
        (axes[1], "sycophancy", "Sycophancy Rate", "Reds",   0, 90,  100),
    ]:
        pivot = (df.groupby(["opening_label", "temporal_label"])[col]
                 .mean()
                 .mul(scale)
                 .unstack("temporal_label")
                 .reindex(index=opening_order, columns=temporal_order))

        sns.heatmap(pivot, ax=ax, annot=True,
                    fmt=".1f",
                    cmap=cmap, vmin=vmin, vmax=vmax,
                    linewidths=0.5, cbar_kws={"label": title})
        ax.set_title(title, fontsize=10)
        ax.set_xlabel("Temporal Context", fontsize=9)
        ax.set_ylabel("Opening Type", fontsize=9)
        ax.tick_params(axis="x", rotation=15, labelsize=8)
        ax.tick_params(axis="y", rotation=0,  labelsize=8)

    plt.tight_layout()
    _save(fig, out_dir, "D3_anchor_heatmap.png")


# ── D4: Error tags by opening type ────────────────────────────────────────────

def plot_d4_error_by_opening(df: pd.DataFrame, out_dir: Path):
    opening_order = sorted(df["opening_label"].unique())
    tag_cols   = ["sycophancy", "over_solving", "over_medicalization"]
    tag_labels = ["Sycophancy", "Over-Solving", "Over-Medicalization"]
    tag_colors = ["#e41a1c", "#ff7f00", "#4daf4a"]

    x = np.arange(len(opening_order))
    w = 0.25

    fig, ax = plt.subplots(figsize=(13, 5))
    fig.suptitle("Error Tag Rates by Opening Type", fontsize=13, fontweight="bold")

    for i, (col, label, color) in enumerate(zip(tag_cols, tag_labels, tag_colors)):
        vals = [df[df["opening_label"] == o][col].mean() * 100
                for o in opening_order]
        ax.bar(x + (i - 1) * w, vals, w, label=label,
               color=color, edgecolor="white", linewidth=0.8, alpha=0.85)

    ax.set_xticks(x)
    ax.set_xticklabels(opening_order, rotation=15, ha="right", fontsize=8)
    ax.set_ylabel("Error Tag Rate (%)", fontsize=10)
    ax.set_ylim(0, 90)
    ax.legend(fontsize=9)

    plt.tight_layout()
    _save(fig, out_dir, "D4_opening_error_tags.png")


# ── D5: Temporal context effect per prod model ────────────────────────────────

def plot_d5_temporal_by_model(df: pd.DataFrame, out_dir: Path):
    prod = df[df["model_tier"] == "prod"]
    models = [m for m in MODEL_ORDER_PROD if m in prod["model_name"].unique()]
    temporal_order = ["Chronic\n(ongoing)", "Acute\n(recent onset)", "Shifted\n(new sensitivity)"]
    temporal_order = [t for t in temporal_order if t in prod["temporal_label"].unique()]

    x      = np.arange(len(temporal_order))
    w      = 0.18
    colors = {m: PALETTE[i % len(PALETTE)] for i, m in enumerate(models)}

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle("Temporal Context Effect per Prod Model",
                 fontsize=13, fontweight="bold")

    for ax, col, title, ylim in [
        (axes[0], "harm_score", "Harm Score (lower = better)", (0, 2.5)),
        (axes[1], "sycophancy", "Sycophancy Rate (%)", (0, 85)),
    ]:
        for m_idx, model in enumerate(models):
            sub  = prod[prod["model_name"] == model]
            vals = [sub[sub["temporal_label"] == tl][col].mean() *
                    (100 if col == "sycophancy" else 1)
                    for tl in temporal_order]
            offset = (m_idx - len(models)/2 + 0.5) * w
            ax.bar(x + offset, vals, w, label=model,
                   color=colors[model], edgecolor="white", linewidth=0.8)

        ax.set_xticks(x)
        ax.set_xticklabels(temporal_order, fontsize=9)
        ax.set_title(title, fontsize=10)
        ax.set_ylim(*ylim)
        ax.legend(fontsize=8)

    plt.tight_layout()
    _save(fig, out_dir, "D5_temporal_by_model.png")


# ── D6: Opening type × theme interaction ──────────────────────────────────────

def plot_d6_opening_by_theme(df: pd.DataFrame, out_dir: Path):
    opening_order = sorted(df["opening_label"].unique())
    themes        = sorted(df["theme"].unique())
    palette       = {t: PALETTE[i % len(PALETTE)] for i, t in enumerate(themes)}

    pivot_harm = (df.groupby(["opening_label", "theme"])["harm_score"]
                  .mean()
                  .unstack("theme")
                  .reindex(opening_order))

    pivot_syco = (df.groupby(["opening_label", "theme"])["sycophancy"]
                  .mean().mul(100)
                  .unstack("theme")
                  .reindex(opening_order))

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("Opening Type × Theme Interaction", fontsize=13, fontweight="bold")

    for ax, pivot, title, cmap, vmin, vmax in [
        (axes[0], pivot_harm, "Harm Score",      "YlOrRd", 0, 2.5),
        (axes[1], pivot_syco, "Sycophancy Rate", "Reds",   0, 90),
    ]:
        sns.heatmap(pivot, ax=ax, annot=True, fmt=".1f",
                    cmap=cmap, vmin=vmin, vmax=vmax,
                    linewidths=0.5, cbar_kws={"label": title})
        ax.set_title(title, fontsize=10)
        ax.set_xlabel("Theme", fontsize=9)
        ax.set_ylabel("Opening Type", fontsize=9)
        ax.tick_params(axis="x", rotation=20, labelsize=8)
        ax.tick_params(axis="y", rotation=0,  labelsize=8)

    plt.tight_layout()
    _save(fig, out_dir, "D6_opening_by_theme.png")


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--eval-dir",     default="data/eval_results")
    parser.add_argument("--stimuli-dir",  default="data/stimuli/llm_generated_scenarios")
    parser.add_argument("--output",       default="analysis/figures/diversity_anchor")
    args = parser.parse_args()

    stimuli_dir = Path(args.stimuli_dir)
    eval_dir    = Path(args.eval_dir)
    out_dir     = Path(args.output)

    print("Building anchor map from stimuli...")
    anchor_map = build_anchor_map(stimuli_dir)
    print(f"  {len(anchor_map)} anchors loaded\n")

    print(f"Loading eval results from {eval_dir} ...")
    df = load_all(eval_dir, anchor_map)
    print(f"  {len(df)} turns loaded\n")

    print(f"Opening types ({df['opening_label'].nunique()}):", sorted(df["opening_label"].unique()))
    print(f"Temporal contexts ({df['temporal_label'].nunique()}):", sorted(df["temporal_label"].unique()))
    print()

    print("Generating figures...")
    plot_d1_opening(df, out_dir)
    plot_d2_temporal(df, out_dir)
    plot_d3_heatmap(df, out_dir)
    plot_d4_error_by_opening(df, out_dir)
    plot_d5_temporal_by_model(df, out_dir)
    plot_d6_opening_by_theme(df, out_dir)

    print(f"\nDone. Saved to: {out_dir}")


if __name__ == "__main__":
    main()
