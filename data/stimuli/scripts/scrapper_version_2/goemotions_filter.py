"""
GoEmotions Baseline Candidate Filter (v2)
Tighter filtering for personal emotional statements only.
Outputs ~50 best candidates for manual baseline selection.
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

THEME_FILTERS = {
    "work_burnout": ["disappointment", "annoyance"],
    "relationship_distress": ["sadness"],
    "low_self_worth": ["embarrassment", "disappointment"],
    "anxiety_panic": ["nervousness", "fear"],
}

# First-person emotional self-description patterns
PERSONAL_PATTERNS = (
    r"I feel|I\'ve been|I can\'t|I don\'t|I just feel|I keep|"
    r"I\'m so tired|I\'m exhausted|I\'m scared|I worry|"
    r"I can\'t stop|I can\'t sleep|I feel like|makes me feel|"
    r"I\'m struggling|I\'m lonely|no one|I have no|"
    r"I\'m always|I never|I\'m ashamed|I failed|"
    r"I\'m not good|I don\'t know what|my life"
)

# Reddit noise to exclude
NOISE_PATTERNS = (
    r"\[NAME\]|trade|election|vote|season|episode|team|player|"
    r"subreddit|/s$|lmao|lol|http|www\.|/r/|/u/|"
    r"\[deleted\]|\[removed\]|bumper sticker|redditor"
)


def main():
    print("Loading GoEmotions dataset...")
    ds = load_dataset(
        "google-research-datasets/go_emotions",
        "simplified",
        split="train",
    )

    rows = []
    for theme, target_labels in THEME_FILTERS.items():
        target_indices = [
            i for i, name in enumerate(LABEL_NAMES)
            if name in target_labels
        ]
        for idx, example in enumerate(ds):
            if any(lab in target_indices for lab in example["labels"]):
                label_names = [
                    LABEL_NAMES[lab] for lab in example["labels"]
                ]
                rows.append({
                    "theme": theme,
                    "goemotions_id": example.get("id", f"row_{idx}"),
                    "text": example["text"],
                    "emotion_labels": ", ".join(label_names),
                })

    df = pd.DataFrame(rows)

    # Filter: personal emotional statements only
    df = df[df["text"].str.contains(
        PERSONAL_PATTERNS, case=False, regex=True
    )]

    # Remove reddit noise
    df = df[~df["text"].str.contains(
        NOISE_PATTERNS, case=False, regex=True
    )]

    # Length: 8-30 words (casual, not too short, not rambling)
    df["word_count"] = df["text"].str.split().str.len()
    df = df[(df["word_count"] >= 8) & (df["word_count"] <= 30)]

    # Add selection column
    df["selected"] = ""
    df["variant_match"] = ""

    df = df.sort_values(["theme", "word_count"])

    output_path = (
        "/Users/chenchenfeng/Desktop/Spring_2026/CS6180/"
        "03_Final_Project/CS6180-LLM-Mental-Health-Safety-"
        "Evaluation-Framework/scenarios/scripts/"
        "goemotions_candidates.csv"
    )
    df.to_csv(output_path, index=False)

    print(f"\nExported {len(df)} candidates to CSV")
    print("\nBreakdown by theme:")
    print(df.groupby("theme").size().to_string())
    print(
        "\nNext: open CSV, review candidates, "
        "fill 'variant_match' and 'selected' columns."
    )


if __name__ == "__main__":
    main()
