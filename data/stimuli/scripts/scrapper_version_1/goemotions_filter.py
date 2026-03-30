"""
GoEmotions Baseline Candidate Filter
Filters GoEmotions dataset by target emotion labels per theme.
Outputs candidates to CSV for manual baseline selection.
See DESIGN.md Section 8a for selection workflow.
"""

from datasets import load_dataset
import pandas as pd

LABEL_NAMES = [
    "admiration", "amusement", "anger", "annoyance", "approval",
    "caring", "confusion", "curiosity", "desire", "disappointment",
    "disapproval", "disgust", "embarrassment", "excitement", "fear",
    "gratitude", "grief", "joy", "love", "nervousness",
    "optimism", "pride", "realization", "relief", "remorse",
    "sadness", "surprise", "neutral"
]

# From DESIGN.md Section 3 — guilt_shame and anhedonia skipped (fully synthetic)
THEME_FILTERS = {
    "work_burnout": ["disappointment", "annoyance"],
    "relationship_distress": ["sadness"],  # GoEmotions has no 'loneliness' label
    "low_self_worth": ["embarrassment", "disappointment"],
    "anxiety_panic": ["nervousness", "fear"],
}

def main():
    print("Loading GoEmotions dataset...")
    ds = load_dataset("google-research-datasets/go_emotions", "simplified", split="train")

    rows = []
    for theme, target_labels in THEME_FILTERS.items():
        target_indices = [i for i, name in enumerate(LABEL_NAMES) if name in target_labels]
        for idx, example in enumerate(ds):
            if any(label in target_indices for label in example["labels"]):
                label_names = [LABEL_NAMES[l] for l in example["labels"]]
                rows.append({
                    "theme": theme,
                    "goemotions_id": example.get("id", f"row_{idx}"),
                    "text": example["text"],
                    "emotion_labels": ", ".join(label_names),
                })

    df = pd.DataFrame(rows)

    # Keep casual-length samples, remove reddit artifacts
    df["word_count"] = df["text"].str.split().str.len()
    df = df[(df["word_count"] >= 5) & (df["word_count"] <= 50)]
    df = df[~df["text"].str.contains(r"http|www\.|/r/|/u/|\[deleted\]|\[removed\]", case=False, regex=True)]

    df["selected"] = ""
    df = df.sort_values(["theme", "word_count"])

    output_path = "/Users/chenchenfeng/Desktop/Spring_2026/CS6180/03_Final_Project/CS6180-LLM-Mental-Health-Safety-Evaluation-Framework/scenarios/scripts/goemotions_candidates.csv"
    df.to_csv(output_path, index=False)
    print(f"\nExported {len(df)} candidates to {output_path}")
    print(f"\nBreakdown by theme:")
    print(df.groupby("theme").size().to_string())
    print(f"\nNext: open CSV, review candidates, mark 8 selections in 'selected' column.")

if __name__ == "__main__":
    main()
