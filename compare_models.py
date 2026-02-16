"""
compare_models.py
══════════════════
Evaluate base GoEmotions vs fine-tuned MELD model on the held-out test set.

Produces:
  • comparison_metrics.csv      — per-class P/R/F1 for both models
  • comparison_bar_chart.png    — grouped bar chart (F1 per emotion)
  • confusion_matrix_base.png   — confusion matrix for base model
  • confusion_matrix_finetuned.png — confusion matrix for fine-tuned model
  • comparison_summary.txt      — plain-text summary for copy-paste

Usage:
    python compare_models.py
"""

import csv
import time
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from transformers import pipeline

warnings.filterwarnings("ignore")

# ── Configuration ────────────────────────────────────────────────────────────

TEST_CSV = Path("test_subtitles.csv")
BASE_MODEL = "SamLowe/roberta-base-go_emotions"
FINETUNED_MODEL = "./finetuned_model"

MELD_LABELS = ["anger", "disgust", "fear", "joy", "neutral", "sadness", "surprise"]

# GoEmotions 28 → MELD 7 mapping
GO_TO_MELD = {
    "admiration":     "joy",
    "amusement":      "joy",
    "anger":          "anger",
    "annoyance":      "anger",
    "approval":       "joy",
    "caring":         "joy",
    "confusion":      "neutral",
    "curiosity":      "neutral",
    "desire":         "joy",
    "disappointment": "sadness",
    "disapproval":    "anger",
    "disgust":        "disgust",
    "embarrassment":  "sadness",
    "excitement":     "joy",
    "fear":           "fear",
    "gratitude":      "joy",
    "grief":          "sadness",
    "joy":            "joy",
    "love":           "joy",
    "nervousness":    "fear",
    "neutral":        "neutral",
    "optimism":       "joy",
    "pride":          "joy",
    "realization":    "neutral",
    "relief":         "joy",
    "remorse":        "sadness",
    "sadness":        "sadness",
    "surprise":       "surprise",
}


# ── Helper Functions ─────────────────────────────────────────────────────────

def load_test_data(csv_path: Path):
    """Load test CSV → list of (text, true_label)."""
    data = []
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            text = row["text"].strip()
            label = row["label"].strip().lower()
            if text and label in MELD_LABELS:
                data.append((text, label))
    print(f"  Loaded {len(data)} test samples")
    return data


def run_inference(model_path, texts, batch_size=16, map_labels=False):
    """Run inference with a HuggingFace model and return predicted labels."""
    print(f"  Loading model: {model_path}")
    clf = pipeline(
        "text-classification",
        model=model_path,
        top_k=None,
        device=-1,
        truncation=True,
    )

    print(f"  Running inference on {len(texts)} samples (batch_size={batch_size}) ...")
    start = time.time()
    results = clf(texts, batch_size=batch_size)
    elapsed = time.time() - start
    print(f"  Inference completed in {elapsed:.1f}s ({elapsed / len(texts) * 1000:.1f} ms/sample)")

    preds = []
    for result in results:
        top = max(result, key=lambda x: x["score"])
        label = top["label"].lower()
        if map_labels:
            label = GO_TO_MELD.get(label, "neutral")
        preds.append(label)

    return preds


def compute_metrics(y_true, y_pred, model_name):
    """Compute and return a dict of metrics."""
    acc = accuracy_score(y_true, y_pred)
    f1_w = f1_score(y_true, y_pred, average="weighted", zero_division=0)
    f1_macro = f1_score(y_true, y_pred, average="macro", zero_division=0)
    prec_w = precision_score(y_true, y_pred, average="weighted", zero_division=0)
    rec_w = recall_score(y_true, y_pred, average="weighted", zero_division=0)

    print(f"\n  ── {model_name} ──")
    print(f"  Accuracy:         {acc:.4f}")
    print(f"  Weighted F1:      {f1_w:.4f}")
    print(f"  Macro F1:         {f1_macro:.4f}")
    print(f"  Weighted Prec:    {prec_w:.4f}")
    print(f"  Weighted Recall:  {rec_w:.4f}")

    return {
        "accuracy": acc,
        "f1_weighted": f1_w,
        "f1_macro": f1_macro,
        "precision_weighted": prec_w,
        "recall_weighted": rec_w,
    }


def plot_confusion_matrix(y_true, y_pred, labels, title, filename):
    """Save a confusion matrix heatmap."""
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    fig, ax = plt.subplots(figsize=(8, 6))

    im = ax.imshow(cm, interpolation="nearest", cmap="Blues")
    ax.set_title(title, fontsize=14, fontweight="bold", pad=15)
    fig.colorbar(im, ax=ax, shrink=0.8)

    ax.set_xticks(range(len(labels)))
    ax.set_yticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=9)
    ax.set_yticklabels(labels, fontsize=9)
    ax.set_xlabel("Predicted", fontsize=11)
    ax.set_ylabel("Actual", fontsize=11)

    # Add text annotations
    thresh = cm.max() / 2.0
    for i in range(len(labels)):
        for j in range(len(labels)):
            ax.text(j, i, str(cm[i, j]),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black",
                    fontsize=9)

    plt.tight_layout()
    plt.savefig(filename, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {filename}")


def plot_comparison_bar_chart(report_base, report_ft, filename):
    """Save a grouped bar chart comparing F1 scores per emotion."""
    emotions = MELD_LABELS
    f1_base = [report_base.get(e, {}).get("f1-score", 0) for e in emotions]
    f1_ft = [report_ft.get(e, {}).get("f1-score", 0) for e in emotions]

    x = np.arange(len(emotions))
    width = 0.35

    fig, ax = plt.subplots(figsize=(10, 5.5))
    bars1 = ax.bar(x - width / 2, f1_base, width, label="Base GoEmotions",
                   color="#64748b", edgecolor="white", linewidth=0.5)
    bars2 = ax.bar(x + width / 2, f1_ft, width, label="Fine-Tuned (MELD)",
                   color="#10b981", edgecolor="white", linewidth=0.5)

    ax.set_xlabel("Emotion", fontsize=11)
    ax.set_ylabel("F1 Score", fontsize=11)
    ax.set_title("Per-Emotion F1 Score: Base vs Fine-Tuned Model", fontsize=13, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels([e.capitalize() for e in emotions], fontsize=10)
    ax.set_ylim(0, 1.0)
    ax.legend(fontsize=10)
    ax.grid(axis="y", alpha=0.3)
    ax.set_facecolor("#fafafa")

    # Add value labels on bars
    for bar in bars1:
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2, h + 0.01,
                f"{h:.2f}", ha="center", va="bottom", fontsize=7.5, color="#475569")
    for bar in bars2:
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2, h + 0.01,
                f"{h:.2f}", ha="center", va="bottom", fontsize=7.5, color="#065f46")

    plt.tight_layout()
    plt.savefig(filename, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {filename}")


def save_metrics_csv(report_base, report_ft, filename):
    """Save per-class comparison metrics to CSV."""
    with open(filename, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["Emotion",
                     "Base_Precision", "Base_Recall", "Base_F1",
                     "FineTuned_Precision", "FineTuned_Recall", "FineTuned_F1",
                     "F1_Improvement"])

        for emo in MELD_LABELS:
            rb = report_base.get(emo, {"precision": 0, "recall": 0, "f1-score": 0})
            rf = report_ft.get(emo, {"precision": 0, "recall": 0, "f1-score": 0})
            improvement = rf["f1-score"] - rb["f1-score"]
            w.writerow([
                emo,
                f"{rb['precision']:.4f}", f"{rb['recall']:.4f}", f"{rb['f1-score']:.4f}",
                f"{rf['precision']:.4f}", f"{rf['recall']:.4f}", f"{rf['f1-score']:.4f}",
                f"{improvement:+.4f}",
            ])

    print(f"  Saved: {filename}")


def save_summary_txt(metrics_base, metrics_ft, report_base, report_ft, filename):
    """Save a formatted plain-text summary."""
    lines = []
    lines.append("=" * 70)
    lines.append("  MODEL COMPARISON SUMMARY")
    lines.append("  Base: SamLowe/roberta-base-go_emotions (28 labels → mapped to 7)")
    lines.append("  Fine-Tuned: ./finetuned_model (MELD, 7 labels)")
    lines.append("=" * 70)
    lines.append("")

    # Overall metrics table
    lines.append("┌─────────────────────┬──────────────┬──────────────┬────────────┐")
    lines.append("│ Metric              │ Base Model   │ Fine-Tuned   │ Δ Change   │")
    lines.append("├─────────────────────┼──────────────┼──────────────┼────────────┤")

    for metric, label in [
        ("accuracy", "Accuracy"),
        ("f1_weighted", "Weighted F1"),
        ("f1_macro", "Macro F1"),
        ("precision_weighted", "Weighted Precision"),
        ("recall_weighted", "Weighted Recall"),
    ]:
        b = metrics_base[metric]
        f = metrics_ft[metric]
        d = f - b
        sign = "+" if d >= 0 else ""
        lines.append(f"│ {label:<19s} │ {b:>10.4f}   │ {f:>10.4f}   │ {sign}{d:>8.4f}  │")

    lines.append("└─────────────────────┴──────────────┴──────────────┴────────────┘")
    lines.append("")

    # Per-class F1 table
    lines.append("┌─────────────┬──────────┬──────────┬────────────┐")
    lines.append("│ Emotion     │ Base F1  │ FT F1    │ Δ Change   │")
    lines.append("├─────────────┼──────────┼──────────┼────────────┤")

    for emo in MELD_LABELS:
        bf1 = report_base.get(emo, {}).get("f1-score", 0)
        ff1 = report_ft.get(emo, {}).get("f1-score", 0)
        d = ff1 - bf1
        sign = "+" if d >= 0 else ""
        lines.append(f"│ {emo:<11s} │ {bf1:>6.4f}   │ {ff1:>6.4f}   │ {sign}{d:>8.4f}  │")

    lines.append("└─────────────┴──────────┴──────────┴────────────┘")

    text = "\n".join(lines)
    with open(filename, "w", encoding="utf-8") as f:
        f.write(text)

    print(f"  Saved: {filename}")
    print()
    print(text)


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    print("\n" + "=" * 60)
    print("  MODEL COMPARISON: Base GoEmotions vs Fine-Tuned MELD")
    print("=" * 60)

    # Step 1: Load test data
    print("\n[1/6] Loading test data...")
    data = load_test_data(TEST_CSV)
    texts = [t for t, _ in data]
    y_true = [l for _, l in data]

    # Step 2: Run base model
    print("\n[2/6] Running BASE model (GoEmotions)...")
    y_pred_base = run_inference(BASE_MODEL, texts, batch_size=16, map_labels=True)

    # Step 3: Run fine-tuned model
    print("\n[3/6] Running FINE-TUNED model (MELD)...")
    y_pred_ft = run_inference(FINETUNED_MODEL, texts, batch_size=16, map_labels=False)

    # Step 4: Compute metrics
    print("\n[4/6] Computing metrics...")
    metrics_base = compute_metrics(y_true, y_pred_base, "Base GoEmotions")
    metrics_ft = compute_metrics(y_true, y_pred_ft, "Fine-Tuned MELD")

    report_base = classification_report(y_true, y_pred_base,
                                        labels=MELD_LABELS, output_dict=True, zero_division=0)
    report_ft = classification_report(y_true, y_pred_ft,
                                      labels=MELD_LABELS, output_dict=True, zero_division=0)

    # Step 5: Generate charts
    print("\n[5/6] Generating visualizations...")
    plot_comparison_bar_chart(report_base, report_ft, "comparison_bar_chart.png")
    plot_confusion_matrix(y_true, y_pred_base, MELD_LABELS,
                          "Confusion Matrix — Base GoEmotions (mapped to 7)",
                          "confusion_matrix_base.png")
    plot_confusion_matrix(y_true, y_pred_ft, MELD_LABELS,
                          "Confusion Matrix — Fine-Tuned (MELD)",
                          "confusion_matrix_finetuned.png")

    # Step 6: Save reports
    print("\n[6/6] Saving reports...")
    save_metrics_csv(report_base, report_ft, "comparison_metrics.csv")
    save_summary_txt(metrics_base, metrics_ft, report_base, report_ft, "comparison_summary.txt")

    print("\n✅ Comparison complete! Output files:")
    print("   • comparison_metrics.csv")
    print("   • comparison_bar_chart.png")
    print("   • confusion_matrix_base.png")
    print("   • confusion_matrix_finetuned.png")
    print("   • comparison_summary.txt")
    print()


if __name__ == "__main__":
    main()
