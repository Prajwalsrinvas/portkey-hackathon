"""
Benchmark PangolinGuard on geekyrakshit/prompt-injection-dataset.
Outputs CSV with query, ground_truth, prediction for error analysis.
"""

import modal

app = modal.App("benchmark-to-csv")

image = modal.Image.debian_slim(python_version="3.11").pip_install(
    "transformers>=4.48.0",
    "torch>=2.5.0",
    "accelerate",
    "datasets",
    "pandas",
)

# Config
SAMPLE_SIZE = 100000
MODEL_NAME = "dcarpintero/pangolin-guard-base"
DATASET_NAME = "geekyrakshit/prompt-injection-dataset"
OUTPUT_CSV = "/data/pangolin_benchmark_100k.csv"


@app.function(
    image=image,
    gpu="H100",  # Fastest GPU for 100K samples
    timeout=3600,  # 1 hour max
    volumes={"/data": modal.Volume.from_name("guardrail-data")},
)
def benchmark_to_csv(sample_size=SAMPLE_SIZE):
    import random

    import pandas as pd
    import torch
    from datasets import load_dataset
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    print(f"{'='*70}")
    print(f"BENCHMARKING: {MODEL_NAME}")
    print(f"DATASET: {DATASET_NAME}")
    print(f"Sample size: {sample_size}")
    print(f"{'='*70}\n")

    # Load model
    print("Loading model...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)
    model.eval()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)
    print(f"Model loaded on {device}")
    print(f"Label mapping: {model.config.id2label}\n")

    # Load dataset
    print(f"Loading {DATASET_NAME}...")
    dataset = load_dataset(DATASET_NAME, split="train")
    print(f"  Total samples in train: {len(dataset)}")

    # Random sample
    random.seed(42)  # Reproducibility
    indices = random.sample(range(len(dataset)), min(sample_size, len(dataset)))
    sampled = dataset.select(indices)
    print(f"  Sampled {len(sampled)} samples")

    # Run inference and collect results
    results = []
    correct = 0
    total = len(sampled)

    print(f"\nRunning inference on {total} samples...")
    for i, row in enumerate(sampled):
        text = row["prompt"]
        ground_truth = int(row["label"])

        # Tokenize and predict
        inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
        inputs = {k: v.to(device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = model(**inputs)
            probs = torch.softmax(outputs.logits, dim=-1)
            prediction = torch.argmax(probs, dim=-1).item()
            confidence = probs[0][prediction].item()

        is_correct = prediction == ground_truth
        if is_correct:
            correct += 1

        results.append(
            {
                "query": text,
                "ground_truth": ground_truth,
                "prediction": prediction,
                "confidence": round(confidence, 4),
                "correct": is_correct,
            }
        )

        # Progress
        if (i + 1) % 5000 == 0:
            acc = correct / (i + 1) * 100
            print(f"  Processed {i+1}/{total} - Running accuracy: {acc:.1f}%")

    # Create DataFrame and save
    df = pd.DataFrame(results)
    df.to_csv(OUTPUT_CSV, index=False)
    print(f"\nResults saved to {OUTPUT_CSV}")

    # Summary stats
    accuracy = correct / total * 100
    print(f"\n{'='*70}")
    print("SUMMARY")
    print(f"{'='*70}")
    print(f"Overall Accuracy: {accuracy:.1f}% ({correct}/{total})")

    # Breakdown by ground truth
    for label in [0, 1]:
        subset = df[df["ground_truth"] == label]
        label_correct = subset["correct"].sum()
        label_total = len(subset)
        label_acc = label_correct / label_total * 100 if label_total > 0 else 0
        label_name = "Benign" if label == 0 else "Attack"
        print(
            f"{label_name} (label={label}): {label_acc:.1f}% ({label_correct}/{label_total})"
        )

    # Error breakdown
    print("\nError Analysis:")
    errors = df[~df["correct"]]
    fn = len(
        errors[(errors["ground_truth"] == 1) & (errors["prediction"] == 0)]
    )  # Missed attacks
    fp = len(
        errors[(errors["ground_truth"] == 0) & (errors["prediction"] == 1)]
    )  # False positives
    print(f"  False Negatives (missed attacks): {fn}")
    print(f"  False Positives (over-defense): {fp}")

    # Commit volume to persist CSV
    modal.Volume.from_name("guardrail-data").commit()

    return {
        "accuracy": accuracy,
        "total": total,
        "correct": correct,
        "false_negatives": fn,
        "false_positives": fp,
        "csv_path": OUTPUT_CSV,
    }


@app.local_entrypoint()
def main():
    result = benchmark_to_csv.remote()
    print(f"\nBenchmark complete. CSV saved to Modal volume at {result['csv_path']}")
    print("CSV saved to Modal volume. See README for download instructions.")
