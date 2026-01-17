"""
Download external benchmarks for training data.

These datasets are combined and split into train/val/test sets
by prepare_notinject_splits.py.

Benchmarks:
1. leolee99/NotInject - Gold standard for over-defense testing
   - 339 benign samples that contain trigger words
   - Tests if model incorrectly flags legitimate prompts

2. deepset/prompt-injections - General benchmark
   - Mix of attacks and benign prompts
   - Standard benchmark in the field

Usage:
    python download_external_benchmarks.py
"""

import json

# Output files (relative to project root)
NOTINJECT_FILE = "data/external_notinject.jsonl"
DEEPSET_FILE = "data/external_deepset.jsonl"


def download_notinject():
    """
    Download NotInject dataset - the gold standard for over-defense testing.

    This dataset contains benign prompts that include trigger words commonly
    associated with prompt injection (like "ignore", "instructions", etc.)
    but are actually legitimate requests.

    A good guardrail should NOT flag these as attacks.
    """
    from datasets import load_dataset

    print("Downloading leolee99/NotInject...")

    # NotInject has three splits, not a single "train" split
    splits = ["NotInject_one", "NotInject_two", "NotInject_three"]

    samples = []
    for split_name in splits:
        ds = load_dataset("leolee99/NotInject", split=split_name)
        for row in ds:
            # NotInject has all benign samples (label=0)
            # These should NOT be flagged as attacks
            samples.append(
                {
                    "prompt": row["prompt"],
                    "label": 0,  # All benign
                    "source": "notinject",
                    "split": split_name,
                }
            )
        print(f"  Loaded {split_name}: {len(ds)} samples")

    print(f"  Total: {len(samples)} samples (all benign with trigger words)")

    # Save
    with open(NOTINJECT_FILE, "w") as f:
        for s in samples:
            f.write(json.dumps(s) + "\n")

    print(f"  Saved to {NOTINJECT_FILE}")
    return samples


def download_deepset():
    """
    Download deepset/prompt-injections - a general benchmark.

    Contains both attacks and benign prompts.
    """
    from datasets import load_dataset

    print("\nDownloading deepset/prompt-injections...")
    ds = load_dataset("deepset/prompt-injections", split="train")

    samples = []
    label_counts = {0: 0, 1: 0}

    for row in ds:
        label = int(row["label"])
        samples.append(
            {
                "prompt": row["text"],
                "label": label,
                "source": "deepset",
            }
        )
        label_counts[label] += 1

    print(f"  Loaded {len(samples)} samples")
    print(f"    - Benign (label=0): {label_counts[0]}")
    print(f"    - Attacks (label=1): {label_counts[1]}")

    # Save
    with open(DEEPSET_FILE, "w") as f:
        for s in samples:
            f.write(json.dumps(s) + "\n")

    print(f"  Saved to {DEEPSET_FILE}")
    return samples


def main():
    print("=" * 70)
    print("DOWNLOADING EXTERNAL BENCHMARKS")
    print("=" * 70)
    print(
        """
These benchmarks are used for FINAL evaluation only.
They were NOT used in training data creation.
This ensures we measure true generalization.
"""
    )

    notinject = download_notinject()
    deepset = download_deepset()

    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(
        f"""
External datasets downloaded:

1. {NOTINJECT_FILE} ({len(notinject)} samples)
   - All benign prompts with trigger words
   - Tests over-defense (false positive rate)

2. {DEEPSET_FILE} ({len(deepset)} samples)
   - Mix of attacks and benign
   - Tests general performance

Next step: Run prepare_notinject_splits.py to create train/val/test splits
"""
    )


if __name__ == "__main__":
    main()
