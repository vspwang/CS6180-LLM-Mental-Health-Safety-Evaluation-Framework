"""
GoEmotions Baseline Candidate Filter (v3)
Targets only the 7 missing baselines from v2.
Broadened labels, all splits, variant-specific keyword matching.
See DESIGN.md Section 8a and scrapper_version_2/result.md.
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

# Broadened label sets to cast a wider net
VARIANT_SEARCHES = {
    "relationship_distress__enmeshment": {
        "labels": ["sadness", "anger", "annoyance", "fear"],
        "keywords": (
            r"partner|relationship|boyfriend|girlfriend|spouse|"
            r"husband|wife|clingy|space|guilt|control|"
            r"suffocate|jealous|possessive|depend|"
            r"upset when I|won't let me|always has to"
        ),
    },
    "anxiety_panic__insomnia": {
        "labels": ["nervousness", "fear", "sadness"],
        "keywords": (
            r"sleep|can't sleep|insomnia|awake|night|"
            r"bed|3 am|4 am|toss|rest|nightmare|"
            r"up all night|lying awake|racing"
        ),
    },
    "anxiety_panic__overthinking": {
        "labels": ["nervousness", "fear", "confusion"],
        "keywords": (
            r"overthink|can't stop thinking|won't stop|"
            r"my head|my mind|spiral|ruminate|replay|"
            r"what if|worry about everything|anxious|"
            r"panic|racing thoughts|brain won't"
        ),
    },
    "guilt_shame__burden": {
        "labels": ["remorse", "sadness", "embarrassment"],
        "keywords": (
            r"burden|drag|fault|my fault|blame myself|"
            r"hurting|dragging|weigh|everyone down|"
            r"better off without|I'm the problem|"
            r"I feel guilty|I feel bad for"
        ),
    },
    "guilt_shame__failure": {
        "labels": ["remorse", "sadness", "disappointment"],
        "keywords": (
            r"let .* down|failed|disappoint|"
            r"should have|shouldn't have|regret|"
            r"my mistake|I messed up|I screwed up|"
            r"I ruined|I broke|I hurt"
        ),
    },
    "anhedonia__flatness": {
        "labels": ["neutral", "sadness", "disappointment"],
        "keywords": (
            r"feel nothing|don't feel|numb|empty|"
            r"don't care anymore|doesn't matter|"
            r"no emotion|can't feel|flat|blank|"
            r"used to .* but now|used to enjoy"
        ),
    },
    "anhedonia__disconnection": {
        "labels": ["neutral", "sadness", "disappointment"],
        "keywords": (
            r"going through the motions|autopilot|"
            r"disconnected|detached|not really here|"
            r"just existing|don't feel present|"
            r"routine|zombie|mechanical|hollow"
        ),
    },
}

NOISE = (
    r"\[NAME\]|http|www\.|/r/|/u/|\[deleted\]|\[removed\]|"
    r"lmao|lol|subreddit|bumper sticker|redditor"
)


def main():
    print("Loading GoEmotions (all splits)...")
    all_rows = []
    for split in ["train", "validation", "test"]:
        ds = load_dataset(
            "google-research-datasets/go_emotions",
            "simplified",
            split=split,
        )
        for idx, example in enumerate(ds):
            all_rows.append({
                "split": split,
                "idx": idx,
                "goemotions_id": example.get("id", f"{split}_{idx}"),
                "text": example["text"],
                "labels": example["labels"],
            })

    print(f"Total samples across all splits: {len(all_rows)}")

    results = []
    for variant_key, config in VARIANT_SEARCHES.items():
        theme, variant = variant_key.split("__")
        target_indices = [
            i for i, name in enumerate(LABEL_NAMES)
            if name in config["labels"]
        ]

        for row in all_rows:
            if not any(lab in target_indices for lab in row["labels"]):
                continue

            text = row["text"]

            # Must contain first-person language
            if not any(
                p in text.lower()
                for p in ["i ", "i'", "my ", "me "]
            ):
                continue

            # Must match variant keywords
            if not pd.Series(
                [text]
            ).str.contains(
                config["keywords"], case=False, regex=True
            ).iloc[0]:
                continue

            word_count = len(text.split())
            if word_count < 6 or word_count > 35:
                continue

            label_names = [
                LABEL_NAMES[lab] for lab in row["labels"]
            ]

            results.append({
                "theme": theme,
                "variant": variant,
                "goemotions_id": row["goemotions_id"],
                "text": text,
                "emotion_labels": ", ".join(label_names),
                "word_count": word_count,
                "selected": "",
            })

    df = pd.DataFrame(results)

    # Remove noise
    if len(df) > 0:
        df = df[~df["text"].str.contains(
            NOISE, case=False, regex=True
        )]

    df = df.sort_values(["theme", "variant", "word_count"])

    output_path = (
        "/Users/chenchenfeng/Desktop/Spring_2026/CS6180/"
        "03_Final_Project/CS6180-LLM-Mental-Health-Safety-"
        "Evaluation-Framework/scenarios/scripts/"
        "scrapper_version_3/goemotions_candidates.csv"
    )
    df.to_csv(output_path, index=False)

    print(f"\nExported {len(df)} candidates")
    print("\nBreakdown by variant:")
    if len(df) > 0:
        print(
            df.groupby(["theme", "variant"]).size().to_string()
        )
    else:
        print("No candidates found.")

    print("\nNext: review candidates, mark selections.")


if __name__ == "__main__":
    main()
