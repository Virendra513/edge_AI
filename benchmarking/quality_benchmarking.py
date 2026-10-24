# Benchmark Llama-3.2-1B-Instruct GGUF Variants on Sentiment Classification


'''
**Workflow:**
1. Load `benchmark.csv` (columns: `sentence`, `label`).
2. For every `.gguf` model in `modle321B/llama-3.2-1B-Instruct-gguf/`, run zero-shot sentiment classification (0 = negative, 1 = positive).
3. Parse the model's reply to extract a numeric label and persist predictions.
4. Compute precision, recall, F1, accuracy, ROC-AUC, classification report, confusion matrix.
5. Save metrics CSV, predictions CSV, AUROC plot, and confusion-matrix plot inside a per-model folder (named after the GGUF file).

**Required libraries:** `llama-cpp-python`, `pandas`, `numpy`, `scikit-learn`, `matplotlib`.



metrics:
    - Precision
    - Recall
    - Accuracy
    - F1 Score
    - n_valid
    - n_invalid
    - valid_ratio
    - coverage_adjusted_f1 = valid_f1 * valid_ratio
'''



import os
import sys
import glob
import re
import time
import subprocess



# Ensure dependencies

_PKG_MAP = {
    "llama-cpp-python": "llama_cpp",
    "pandas": "pandas",
    "numpy": "numpy",
    "scikit-learn": "sklearn",
    "matplotlib": "matplotlib",
}

for pkg, mod in _PKG_MAP.items():
    try:
        __import__(mod)
    except ImportError:
        subprocess.check_call([
            sys.executable,
            "-m",
            "pip",
            "install",
            pkg,
            "--quiet"
        ])


# Imports

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    roc_curve,
    confusion_matrix,
)

from llama_cpp import Llama



# Paths & configuration


WORKDIR = "/work"

BENCHMARK_CSV = os.path.join(
    WORKDIR,
    "benchmarking",
    "benchmark.csv"
)

BENCH_DIR = os.path.join(
    WORKDIR,
    "benchmarking"
)

MODEL_DIR = os.path.join(
    WORKDIR,
    "modle321B",
    "llama-3.2-1B-Instruct-gguf"
)


# Model configuration


SEED = 42

# Smoke test
# Set to None for full dataset
MAX_SAMPLES = 1000

MAX_TOKENS = 4

N_CTX = 512

N_THREADS = max(
    1,
    os.cpu_count() or 1
)


# System prompt

SYSTEM_PROMPT = (
    "You are a binary text classifier. Given a sentence, respond with only "
    "a single digit: 1 or 0. Do not output anything else."
)


# Validate paths

assert os.path.isdir(
    MODEL_DIR
), f"MODEL_DIR not found: {MODEL_DIR}"

assert os.path.isfile(
    BENCHMARK_CSV
), f"BENCHMARK_CSV not found: {BENCHMARK_CSV}"

os.makedirs(
    BENCH_DIR,
    exist_ok=True
)


print(f"WORKDIR = {WORKDIR}")
print(f"MODEL_DIR = {MODEL_DIR}")
print(f"BENCH_DIR = {BENCH_DIR}")
print(f"BENCHMARK = {BENCHMARK_CSV}")


# Load benchmark CSV

df = pd.read_csv(BENCHMARK_CSV)

df.columns = [
    c.strip().lower()
    for c in df.columns
]

assert {
    "sentence",
    "label"
}.issubset(df.columns), (
    f"Missing columns: {df.columns.tolist()}"
)

# Remove rows where sentence or label is missing
df = df.dropna(
    subset=["sentence", "label"]
).reset_index(drop=True)

# Convert labels to integer
df["label"] = df["label"].astype(int)

# Optional sample limit
if MAX_SAMPLES is not None:
    df = df.head(
        MAX_SAMPLES
    ).reset_index(drop=True)


print(
    df.shape,
    df["label"].value_counts().to_dict()
)

# Build prompt

def build_prompt(sentence: str) -> str:

    return (
        "<|start_header_id|>system<|end_header_id|>\n\n"
        f"{SYSTEM_PROMPT}"
        "<|eot_id|>"
        "<|start_header_id|>user<|end_header_id|>\n\n"
        f"Sentence: {sentence}\n"
        "Label:<|eot_id|>"
        "<|start_header_id|>assistant<|end_header_id|>\n\n"
    )


# Parse model output

def parse_label(text: str):

    """
    Extract the first 0 or 1 from model output.

    Returns:
        0 or 1 -> valid prediction
        None   -> invalid/unparsable prediction
    """

    match = re.search(
        r"[01]",
        text
    )

    if match:
        return int(
            match.group()
        )

    return None


# Evaluate one model

def evaluate_one(
    model_path: str,
    data: pd.DataFrame
) -> dict:

    model_name = os.path.splitext(
        os.path.basename(model_path)
    )[0]

    model_dir = os.path.join(
        BENCH_DIR,
        model_name
    )

    os.makedirs(
        model_dir,
        exist_ok=True
    )

    print(
        f"\n=== Evaluating {model_name} ==="
    )

    # Load Llama model

    llm = Llama(
        model_path=model_path,
        n_ctx=N_CTX,
        n_threads=N_THREADS,
        n_gpu_layers=0,
        logits_all=False,
        verbose=False,
        seed=SEED,
    )


    # Ground truth and predictions

    y_true = data["label"].tolist()

    y_pred = []



    # Inference

    start = time.time()

    for sentence in data["sentence"]:

        prompt = build_prompt(
            sentence
        )

        out = llm(
            prompt,
            max_tokens=MAX_TOKENS,
            temperature=0.0,
            stop=["<|eot_id|>"],
        )

        text = out["choices"][0]["text"].strip()

        prediction = parse_label(
            text
        )

        y_pred.append(
            prediction
        )

    elapsed = time.time() - start


    # Separate VALID and INVALID predictions

    valid_mask = [
        p is not None
        for p in y_pred
    ]

    n_samples = len(
        y_pred
    )

    n_valid = sum(
        valid_mask
    )

    n_invalid = sum(
        not v
        for v in valid_mask
    )

    valid_ratio = (
        n_valid / n_samples
        if n_samples > 0
        else 0.0
    )


    # ONLY valid predictions are used for metrics

    y_true_valid = [
        t
        for t, p in zip(
            y_true,
            y_pred
        )
        if p is not None
    ]

    y_pred_valid = [
        p
        for p in y_pred
        if p is not None
    ]


    print(
        f"  Total samples : {n_samples}"
    )

    print(
        f"  Valid : {n_valid}"
    )

    print(
        f"  Invalid : {n_invalid}"
    )

    print(
        f"  Valid ratio : {valid_ratio:.3f}"
    )

    # Handle case where all predictions are invalid

    if n_valid == 0:

        print(
            "  WARNING: All predictions are invalid."
        )

        accuracy = float("nan")
        precision = float("nan")
        recall = float("nan")
        f1 = float("nan")
        auc_val = float("nan")

        coverage_adjusted_f1 = float("nan")

        cm = np.zeros(
            (2, 2),
            dtype=int
        )

    else:

        # Classification metrics

        accuracy = accuracy_score(
            y_true_valid,
            y_pred_valid
        )

        precision = precision_score(
            y_true_valid,
            y_pred_valid,
            zero_division=0
        )

        recall = recall_score(
            y_true_valid,
            y_pred_valid,
            zero_division=0
        )

        f1 = f1_score(
            y_true_valid,
            y_pred_valid,
            zero_division=0
        )



        # Coverage-adjusted F1
        # Custom metric:
        # adjusted_F1 =
        #       valid_F1 * (valid / total)

        coverage_adjusted_f1 = (
            f1 * valid_ratio
        )

        # Confusion Matrix
 
        cm = confusion_matrix(
            y_true_valid,
            y_pred_valid,
            labels=[0, 1]
        )


        
        # ROC-AUC
        #
        # These are hard 0/1 predictions, not probabilities.
        # Therefore this is NOT a full probability ROC curve.

        try:

            fpr, tpr, _ = roc_curve(
                y_true_valid,
                y_pred_valid
            )

            auc_val = roc_auc_score(
                y_true_valid,
                y_pred_valid
            )

        except ValueError:

            fpr = np.array([
                0,
                1
            ])

            tpr = np.array([
                0,
                1
            ])

            auc_val = float("nan")


    # Save per-sample predictions


    pred_df = data.copy()

    # Raw parsed prediction:
    # 0 / 1 for valid
    # NaN for invalid
    pred_df["predicted_label"] = y_pred

    # Explicit validity flag
    pred_df["prediction_valid"] = [
        p is not None
        for p in y_pred
    ]

    pred_csv_path = os.path.join(
        model_dir,
        f"{model_name}.csv"
    )

    pred_df.to_csv(
        pred_csv_path,
        index=False
    )

    # Save confusion matrix
    
    cm_df = pd.DataFrame(
        cm,
        index=[
            "true_0",
            "true_1"
        ],
        columns=[
            "pred_0",
            "pred_1"
        ]
    )

    cm_df.to_csv(
        os.path.join(
            model_dir,
            "confusion_matrix.csv"
        )
    )


    # Plot confusion matrix

    fig, ax = plt.subplots(
        figsize=(4, 4)
    )

    im = ax.imshow(
        cm,
        cmap="Blues"
    )

    ax.set_xlabel(
        "Predicted"
    )

    ax.set_ylabel(
        "True"
    )

    ax.set_title(
        f"Confusion Matrix: {model_name}"
    )

    ax.set_xticks(
        [0, 1],
        ["0", "1"]
    )

    ax.set_yticks(
        [0, 1],
        ["0", "1"]
    )


    for i in range(
        cm.shape[0]
    ):

        for j in range(
            cm.shape[1]
        ):

            ax.text(
                j,
                i,
                cm[i, j],
                ha="center",
                va="center",
                color="black"
            )


    fig.colorbar(
        im,
        fraction=0.046,
        pad=0.04
    )

    fig.tight_layout()

    fig.savefig(
        os.path.join(
            model_dir,
            "confusion_matrix.png"
        ),
        dpi=150
    )

    plt.close(fig)

    # Plot ROC curve

    fig, ax = plt.subplots(
        figsize=(5, 5)
    )

    ax.plot(
        fpr,
        tpr,
        marker="o",
        label=f"AUC = {auc_val:.3f}"
    )

    ax.plot(
        [0, 1],
        [0, 1],
        "--",
        color="gray"
    )

    ax.set_xlabel(
        "False Positive Rate"
    )

    ax.set_ylabel(
        "True Positive Rate"
    )

    ax.set_title(
        f"ROC Curve: {model_name}"
    )

    ax.legend()

    fig.tight_layout()

    fig.savefig(
        os.path.join(
            model_dir,
            "roc_curve.png"
        ),
        dpi=150
    )

    plt.close(fig)

    # Metrics dictionary

    metrics = {

        "model": model_name,

        "n_samples": n_samples,

        "n_valid": n_valid,

        "n_invalid": n_invalid,

        "valid_ratio": valid_ratio,

        # F1 calculated ONLY on valid predictions
        "f1_score": f1,

        "accuracy": accuracy,

        "precision": precision,

        "recall": recall,

        "roc_auc": auc_val,

        # Custom metric:
        # F1_valid * valid_ratio
        "coverage_adjusted_f1": coverage_adjusted_f1,

        "seconds": elapsed,

        "output_dir": model_dir,
    }

    # Print results
    

    print(
        f"  Accuracy = {accuracy:.3f}"
    )

    print(
        f"  Precision = {precision:.3f}"
    )

    print(
        f"  Recall = {recall:.3f}"
    )

    print(
        f"  F1 (valid only) = {f1:.3f}"
    )

    print(
        f"  Coverage-adjusted F1 = {coverage_adjusted_f1:.3f}"
    )

    print(
        f"  ROC-AUC = {auc_val:.3f}"
    )

    print(
        f"  Invalid = {n_invalid}"
    )

    print(
        f"  Valid ratio = {valid_ratio:.3f}"
    )

    print(
        f"  Time = {elapsed:.1f}s"
    )

    print(
        f"  Output = {model_dir}"
    )


    
    # Free model

    del llm

    return metrics



# Discover GGUF models

gguf_paths = sorted(
    glob.glob(
        os.path.join(
            MODEL_DIR,
            "*.gguf"
        )
    )
)

print(
    f"\nFound {len(gguf_paths)} GGUF models"
)

for p in gguf_paths:

    print(
        " -",
        os.path.basename(p)
    )


# Run benchmark

all_metrics = []


for path in gguf_paths:

    try:

        result = evaluate_one(
            path,
            df
        )

        all_metrics.append(
            result
        )

    except Exception as e:

        print(
            f"!! Failed on {path}: {e}"
        )

# Summary

if all_metrics:

    summary = pd.DataFrame(
        all_metrics
    ).sort_values(
        "f1_score",
        ascending=False
    )

    summary.to_csv(
        os.path.join(
            BENCH_DIR,
            "summary_metrics.csv"
        ),
        index=False
    )

    print(
        "\n SUMMARY "
    )

    print(
        summary.to_string(
            index=False
        )
    )

else:

    summary = pd.DataFrame()

    print(
        "\n SUMMARY "
    )

    print(
        "No models completed successfully."
    )


# Plot F1 comparison

if not summary.empty:

    # Sort from low to high for horizontal bar plot
    s = summary.sort_values(
        "f1_score"
    )

    fig, ax = plt.subplots(
        figsize=(
            9,
            max(
                3,
                0.4 * len(s)
            )
        )
    )

    ax.barh(
        s["model"],
        s["f1_score"],
        color="steelblue"
    )

    ax.set_xlabel(
        "F1 Score (Valid Predictions Only)"
    )

    ax.set_title(
        "F1 across Llama-3.2-1B GGUF Quantizations"
    )


    for i, v in enumerate(
        s["f1_score"]
    ):

        ax.text(
            v,
            i,
            f"  {v:.3f}",
            va="center"
        )


    fig.tight_layout()

    out = os.path.join(
        BENCH_DIR,
        "f1_comparison.png"
    )

    fig.savefig(
        out,
        dpi=150,
        bbox_inches="tight"
    )

    plt.close(fig)

    print(
        "Saved:",
        out
    )

else:

    print(
        "Skipped F1 bar chart: "
        "no models completed successfully."
    )