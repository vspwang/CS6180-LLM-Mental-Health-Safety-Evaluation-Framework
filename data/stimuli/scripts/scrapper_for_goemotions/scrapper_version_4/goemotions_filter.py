"""
GoEmotions Baseline Candidate Filter (v4)
No label filtering — keyword-only search across all 54K samples.
Targets the 4 remaining baselines that v3 couldn't fill:
  - relationship_distress / enmeshment
  - guilt_shame / burden
  - anhedonia / flatness
  - anhedonia / disconnection
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

# Keyword-only searches — no label requirement
VARIANT_SEARCHES = {
    "relationship_distress__enmeshment": (
        r"partner .* upset|boyfriend .* upset|girlfriend .* upset|"
        r"husband .* upset|wife .* upset|"
        r"can't have my own|no space|need space|want space|"
        r"guilt.* when I|guilty.* when I|"
        r"clingy|suffocating|smothering|possessive|"
        r"controls|controlling|won't let me|"
        r"gets mad when I|gets upset when I|"
        r"jealous when|can't do anything without|"
        r"always has to know|checks my|"
        r"codependent|too attached|too close|"
        r"breathing room|own life|own friends|"
        r"everything I do|watches me|follows me"
    ),
    "guilt_shame__burden": (
        r"I'm a burden|I am a burden|"
        r"burden to|burden on|"
        r"dragging .* down|drag .* down|"
        r"better off without me|"
        r"I'm the problem|I am the problem|"
        r"everyone.*suffer.*because of me|"
        r"my fault.*they|their .* my fault|"
        r"I make .* worse|I ruin|I'm ruining|"
        r"weighing .* down|holding .* back|"
        r"they .* deal with me|put up with me|"
        r"don't deserve .* patience|"
        r"I'm too much|I am too much|"
        r"sick of me|tired of me|"
        r"I shouldn't be here"
    ),
    "anhedonia__flatness": (
        r"feel nothing|don't feel anything|"
        r"can't feel anything|stopped feeling|"
        r"emotionally numb|I'm numb|I feel numb|"
        r"empty inside|feel empty|I'm empty|"
        r"no emotions|lost .* feeling|"
        r"don't care about anything|"
        r"nothing makes me|nothing excites|"
        r"used to enjoy .* but|used to love .* but|"
        r"used to care .* but|"
        r"can't cry|want to cry but|"
        r"flat|hollow inside|void"
    ),
    "anhedonia__disconnection": (
        r"going through the motions|"
        r"just existing|merely existing|"
        r"on autopilot|auto.?pilot|"
        r"not really here|not really present|"
        r"disconnected from|detached from|"
        r"watching .* from outside|"
        r"don't feel real|nothing feels real|"
        r"just surviving|not living|"
        r"same thing every day|same routine|"
        r"robot|robotic|mechanical|"
        r"functioning but|alive but"
    ),
}

NOISE = (
    r"\[NAME\]|http|www\.|/r/|/u/|\[deleted\]|\[removed\]|"
    r"lmao|lol|subreddit|redditor|"
    r"movie|film|episode|season|team|player|"
    r"trump|obama|brexit|election|democrat|republican"
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
                "goemotions_id": example.get("id", f"{split}_{idx}"),
                "text": example["text"],
                "labels": example["labels"],
            })

    print(f"Total samples: {len(all_rows)}")

    results = []
    for variant_key, keywords in VARIANT_SEARCHES.items():
        theme, variant = variant_key.split("__")

        for row in all_rows:
            text = row["text"]

            # First-person filter
            text_lower = text.lower()
            if not any(
                p in text_lower
                for p in ["i ", "i'", "my ", "me ", "i\n"]
            ):
                continue

            # Keyword match only — no label filter
            if not pd.Series(
                [text]
            ).str.contains(
                keywords, case=False, regex=True
            ).iloc[0]:
                continue

            word_count = len(text.split())
            if word_count < 6 or word_count > 40:
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

    if len(df) > 0:
        df = df[~df["text"].str.contains(
            NOISE, case=False, regex=True
        )]

    df = df.sort_values(["theme", "variant", "word_count"])

    output_path = (
        "/Users/chenchenfeng/Desktop/Spring_2026/CS6180/"
        "03_Final_Project/CS6180-LLM-Mental-Health-Safety-"
        "Evaluation-Framework/scenarios/scripts/"
        "scrapper_version_4/goemotions_candidates.csv"
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


if __name__ == "__main__":
    main()
