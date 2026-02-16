"""
finetune.py
────────────
Fine-tune SamLowe/roberta-base-go_emotions on MELD subtitle data.

Prerequisites:
    pip install datasets scikit-learn accelerate matplotlib

Usage:
    1. First run:   python prepare_meld.py    (creates labeled_subtitles.csv)
    2. Then run:    python finetune.py         (trains the model)
    3. Update emotion_model.py:  model="./finetuned_model"

Training time: ~30-60 min on CPU for the full MELD dataset.
"""

import os
import pandas as pd
import numpy as np
from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
)
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, classification_report

# ═══════════════════════════════════════════════════════════════════════
# Configuration — adjust these as needed
# ═══════════════════════════════════════════════════════════════════════

BASE_MODEL = "SamLowe/roberta-base-go_emotions"
DATASET_PATH = "labeled_subtitles.csv"
OUTPUT_DIR = "./finetuned_model"
EPOCHS = 4
BATCH_SIZE = 16
LEARNING_RATE = 2e-5
MAX_LENGTH = 128        # max tokens per subtitle line
VAL_SPLIT = 0.15        # 15% for validation


def main():
    # ── Load Dataset ──────────────────────────────────────────────────
    if not os.path.exists(DATASET_PATH):
        print(f"❌ {DATASET_PATH} not found!")
        print("   Run 'python prepare_meld.py' first.")
        return

    df = pd.read_csv(DATASET_PATH)
    print(f"📂 Loaded {len(df):,} samples from {DATASET_PATH}")

    # ── Build Label Maps ──────────────────────────────────────────────
    labels = sorted(df["label"].unique().tolist())
    label2id = {label: i for i, label in enumerate(labels)}
    id2label = {i: label for label, i in label2id.items()}
    num_labels = len(labels)

    print(f"🏷️  {num_labels} emotion classes: {', '.join(labels)}")

    df["label_id"] = df["label"].map(label2id)

    # ── Train / Validation Split ──────────────────────────────────────
    train_df, val_df = train_test_split(
        df, test_size=VAL_SPLIT, stratify=df["label_id"], random_state=42
    )
    print(f"📊 Train: {len(train_df):,}  |  Val: {len(val_df):,}")

    train_ds = Dataset.from_pandas(train_df[["text", "label_id"]].reset_index(drop=True))
    val_ds = Dataset.from_pandas(val_df[["text", "label_id"]].reset_index(drop=True))

    # ── Tokenizer ─────────────────────────────────────────────────────
    print(f"\n📦 Loading tokenizer from {BASE_MODEL}...")
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)

    def tokenize_fn(batch):
        tokens = tokenizer(
            batch["text"],
            padding="max_length",
            truncation=True,
            max_length=MAX_LENGTH,
        )
        tokens["labels"] = batch["label_id"]
        return tokens

    print("🔤 Tokenizing datasets...")
    train_ds = train_ds.map(tokenize_fn, batched=True, remove_columns=["text", "label_id"])
    val_ds = val_ds.map(tokenize_fn, batched=True, remove_columns=["text", "label_id"])

    train_ds.set_format("torch")
    val_ds.set_format("torch")

    # ── Model ─────────────────────────────────────────────────────────
    print(f"🧠 Loading model from {BASE_MODEL}...")
    model = AutoModelForSequenceClassification.from_pretrained(
        BASE_MODEL,
        num_labels=num_labels,
        id2label=id2label,
        label2id=label2id,
        ignore_mismatched_sizes=True,  # replaces the classification head
        problem_type="single_label_classification",  # use CrossEntropyLoss, not BCE
    )

    # ── Metrics ───────────────────────────────────────────────────────
    def compute_metrics(eval_pred):
        logits, label_ids = eval_pred
        preds = np.argmax(logits, axis=-1)
        return {
            "accuracy": accuracy_score(label_ids, preds),
            "f1_weighted": f1_score(label_ids, preds, average="weighted"),
        }

    # ── Training Arguments ────────────────────────────────────────────
    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        num_train_epochs=EPOCHS,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE,
        learning_rate=LEARNING_RATE,
        warmup_ratio=0.1,
        weight_decay=0.01,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="f1_weighted",
        greater_is_better=True,
        logging_steps=10,
        report_to="none",       # no wandb/tensorboard
        fp16=False,             # CPU-only, no mixed precision
        dataloader_num_workers=0,
    )

    # ── Trainer ───────────────────────────────────────────────────────
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        compute_metrics=compute_metrics,
    )

    # ── Train ─────────────────────────────────────────────────────────
    print("\n" + "═" * 60)
    print("🚀 STARTING FINE-TUNING")
    print(f"   Model:      {BASE_MODEL}")
    print(f"   Epochs:     {EPOCHS}")
    print(f"   Batch size: {BATCH_SIZE}")
    print(f"   LR:         {LEARNING_RATE}")
    print("═" * 60 + "\n")

    train_result = trainer.train()

    # ── Plot Training Curves ──────────────────────────────────────────
    plot_training_curves(trainer)

    # ── Evaluate on Validation Set ────────────────────────────────────
    print("\n📈 Final Evaluation on Validation Set:")
    results = trainer.evaluate()
    print(f"   Accuracy:     {results['eval_accuracy']:.4f}")
    print(f"   F1 (weighted): {results['eval_f1_weighted']:.4f}")

    # ── Detailed Classification Report ────────────────────────────────
    predictions = trainer.predict(val_ds)
    preds = np.argmax(predictions.predictions, axis=-1)
    true_labels = predictions.label_ids

    target_names = [id2label[i] for i in range(num_labels)]
    report = classification_report(true_labels, preds, target_names=target_names)
    print(f"\n📋 Classification Report:\n{report}")

    # ── Save Classification Report to File ─────────────────────────
    with open("classification_report.txt", "w") as f:
        f.write(report)
    print("✅ Classification report saved to classification_report.txt")

    # ── Save Model ────────────────────────────────────────────────────
    trainer.save_model(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)

    print("═" * 60)
    print(f"✅ Fine-tuned model saved to: {os.path.abspath(OUTPUT_DIR)}")
    print()
    print("📝 Next steps:")
    print("   1. Open emotion_model.py")
    print('   2. Change:  model="SamLowe/roberta-base-go_emotions"')
    print('      To:      model="./finetuned_model"')
    print("   3. Restart: python app.py")
    print("═" * 60)


def plot_training_curves(trainer):
    """
    Extract training logs and plot loss / accuracy / F1 curves.
    Saves the plot as 'training_curves.png'.
    """
    import matplotlib
    matplotlib.use("Agg")  # non-interactive backend (no GUI needed)
    import matplotlib.pyplot as plt

    log_history = trainer.state.log_history

    # ── Extract training loss (logged every N steps) ──────────────
    train_steps = []
    train_losses = []
    for entry in log_history:
        if "loss" in entry and "eval_loss" not in entry:
            train_steps.append(entry.get("step", 0))
            train_losses.append(entry["loss"])

    # ── Extract eval metrics (logged every epoch) ─────────────────
    eval_epochs = []
    eval_losses = []
    eval_accs = []
    eval_f1s = []
    for entry in log_history:
        if "eval_loss" in entry:
            eval_epochs.append(entry.get("epoch", 0))
            eval_losses.append(entry["eval_loss"])
            eval_accs.append(entry.get("eval_accuracy", 0))
            eval_f1s.append(entry.get("eval_f1_weighted", 0))

    # ── Plot ──────────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle("Fine-Tuning Training Curves", fontsize=14, fontweight="bold")

    # 1. Training Loss vs Steps
    axes[0].plot(train_steps, train_losses, color="#FF6347", linewidth=1.5, alpha=0.8)
    axes[0].set_xlabel("Training Steps")
    axes[0].set_ylabel("Loss")
    axes[0].set_title("Training Loss")
    axes[0].grid(True, alpha=0.3)

    # 2. Eval Loss + Accuracy vs Epoch
    ax2 = axes[1]
    ax2.plot(eval_epochs, eval_losses, "o-", color="#4A90D9", linewidth=2, label="Eval Loss")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Loss", color="#4A90D9")
    ax2.tick_params(axis="y", labelcolor="#4A90D9")
    ax2.set_title("Eval Loss & Accuracy")
    ax2.grid(True, alpha=0.3)

    ax2b = ax2.twinx()
    ax2b.plot(eval_epochs, eval_accs, "s-", color="#2ECC71", linewidth=2, label="Accuracy")
    ax2b.set_ylabel("Accuracy", color="#2ECC71")
    ax2b.tick_params(axis="y", labelcolor="#2ECC71")

    # 3. F1 Score vs Epoch
    axes[2].plot(eval_epochs, eval_f1s, "D-", color="#9B59B6", linewidth=2, markersize=8)
    axes[2].set_xlabel("Epoch")
    axes[2].set_ylabel("F1 (Weighted)")
    axes[2].set_title("Weighted F1 Score")
    axes[2].grid(True, alpha=0.3)

    # Add value annotations on F1 plot
    for x, y in zip(eval_epochs, eval_f1s):
        axes[2].annotate(f"{y:.3f}", (x, y), textcoords="offset points",
                        xytext=(0, 10), ha="center", fontsize=9)

    plt.tight_layout()
    plt.savefig("training_curves.png", dpi=150, bbox_inches="tight")
    print("\n📊 Training curves saved to training_curves.png")
    plt.close()


if __name__ == "__main__":
    main()
