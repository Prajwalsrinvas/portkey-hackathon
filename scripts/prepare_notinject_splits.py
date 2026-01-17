"""
Prepare training data from NotInject + deepset for fine-tuning.

Strategy:
- Use NotInject (over-defense benchmark) as primary training signal
- Add attack samples from deepset to balance classes
- Split 70/15/15 for train/val/test (proper methodology)
- Validation: used for early stopping during training
- Test: touched ONLY for final evaluation (never seen during training)
"""

import json
import random

SEED = 42
random.seed(SEED)


def load_jsonl(path):
    samples = []
    with open(path) as f:
        for line in f:
            samples.append(json.loads(line))
    return samples


def stratified_split(data, ratios, seed=42):
    """
    Split data into multiple sets while preserving class distribution.

    Args:
        data: list of samples with 'label' field
        ratios: list of ratios (e.g., [0.7, 0.15, 0.15] for train/val/test)
        seed: random seed

    Returns:
        list of splits corresponding to ratios
    """
    random.seed(seed)

    # Separate by label
    by_label = {}
    for sample in data:
        label = sample["label"]
        if label not in by_label:
            by_label[label] = []
        by_label[label].append(sample)

    # Shuffle each class
    for label in by_label:
        random.shuffle(by_label[label])

    # Create splits
    splits = [[] for _ in ratios]

    for label, samples in by_label.items():
        n = len(samples)
        idx = 0
        for i, ratio in enumerate(ratios):
            # Calculate count for this split
            if i == len(ratios) - 1:
                # Last split gets the rest
                count = n - idx
            else:
                count = int(n * ratio)

            splits[i].extend(samples[idx : idx + count])
            idx += count

    # Shuffle each split
    for split in splits:
        random.shuffle(split)

    return splits


def main():
    print("=" * 70)
    print("PREPARING NOTINJECT + DEEPSET TRAINING DATA")
    print("=" * 70)
    print("Using 70/15/15 train/val/test split (proper methodology)")

    # Load datasets
    notinject = load_jsonl("data/external_notinject.jsonl")
    deepset = load_jsonl("data/external_deepset.jsonl")

    print("\nLoaded:")
    print(f"  NotInject: {len(notinject)} samples (all benign with triggers)")
    print(f"  deepset: {len(deepset)} samples")

    # Separate deepset by label
    deepset_benign = [s for s in deepset if s["label"] == 0]
    deepset_attacks = [s for s in deepset if s["label"] == 1]
    print(f"    - Benign: {len(deepset_benign)}")
    print(f"    - Attacks: {len(deepset_attacks)}")

    # Combine all data
    all_benign = notinject + deepset_benign  # 339 + 343 = 682
    all_attacks = deepset_attacks  # 203

    # Ensure all samples have label field
    for s in all_benign:
        s["label"] = 0
    for s in all_attacks:
        s["label"] = 1

    all_data = all_benign + all_attacks

    print("\nCombined dataset:")
    print(f"  Benign (label=0): {len(all_benign)}")
    print(f"  Attacks (label=1): {len(all_attacks)}")
    print(f"  Total: {len(all_data)}")

    # Split 70/15/15
    train_data, val_data, test_data = stratified_split(
        all_data, ratios=[0.70, 0.15, 0.15], seed=SEED
    )

    # Clean up samples for training (only need prompt and label)
    train_clean = [{"prompt": s["prompt"], "label": s["label"]} for s in train_data]
    val_clean = [{"prompt": s["prompt"], "label": s["label"]} for s in val_data]
    test_clean = [{"prompt": s["prompt"], "label": s["label"]} for s in test_data]

    print("\n" + "=" * 70)
    print("FINAL SPLITS (70/15/15)")
    print("=" * 70)

    for name, data in [
        ("TRAIN", train_clean),
        ("VAL", val_clean),
        ("TEST", test_clean),
    ]:
        safe = sum(1 for s in data if s["label"] == 0)
        unsafe = sum(1 for s in data if s["label"] == 1)
        print(f"\n{name} SET: {len(data)} samples")
        print(f"  Safe (label=0): {safe} ({safe/len(data)*100:.1f}%)")
        print(f"  Unsafe (label=1): {unsafe} ({unsafe/len(data)*100:.1f}%)")

    # Verify no overlap
    train_prompts = set(s["prompt"] for s in train_clean)
    val_prompts = set(s["prompt"] for s in val_clean)
    test_prompts = set(s["prompt"] for s in test_clean)

    print("\n" + "=" * 70)
    print("OVERLAP CHECK")
    print("=" * 70)
    print(f"  Train ∩ Val: {len(train_prompts & val_prompts)} (should be 0)")
    print(f"  Train ∩ Test: {len(train_prompts & test_prompts)} (should be 0)")
    print(f"  Val ∩ Test: {len(val_prompts & test_prompts)} (should be 0)")

    # Save
    with open("data/train_notinject.jsonl", "w") as f:
        for s in train_clean:
            f.write(json.dumps(s) + "\n")

    with open("data/val_notinject.jsonl", "w") as f:
        for s in val_clean:
            f.write(json.dumps(s) + "\n")

    with open("data/test_notinject.jsonl", "w") as f:
        for s in test_clean:
            f.write(json.dumps(s) + "\n")

    print("\n" + "=" * 70)
    print("FILES CREATED")
    print("=" * 70)
    print(f"  data/train_notinject.jsonl ({len(train_clean)} samples) - for training")
    print(f"  data/val_notinject.jsonl ({len(val_clean)} samples) - for early stopping")
    print(
        f"  data/test_notinject.jsonl ({len(test_clean)} samples) - for final evaluation ONLY"
    )

    print("\nNote: Validation set is used for early stopping during training.")
    print(
        "      Test set is ONLY used for final evaluation after training is complete."
    )


if __name__ == "__main__":
    main()
