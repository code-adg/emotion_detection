"""
sample.py
══════════════════════════════════════════════════════════════════════════════
Context-Aware Emotion Detection from Subtitle Text — End-to-End Demo

Description of Sample Code
──────────────────────────
Purpose:
    • Demonstrates the complete emotion detection pipeline in a single script.
    • Parses an .srt subtitle file, builds context windows, runs the fine-tuned
      RoBERTa model, applies confidence thresholding and majority-vote
      aggregation, and prints colour-annotated results to the terminal.

Main Functions Included:
    • parse_srt()            – reads an .srt file into structured subtitle dicts
    • build_context_windows() – creates sliding context windows with [SEP] tokens
    • predict_emotions()     – runs batched inference using the fine-tuned model
    • apply_threshold()      – demotes low-confidence labels to "neutral"
    • aggregate_votes()      – stabilises predictions via majority voting
    • run_pipeline()         – ties all steps together end-to-end
    • print_results()        – displays a colour-formatted summary table

Working Flow:
    1. Load and parse the .srt subtitle file
    2. Build sliding context windows (±2 lines) for each subtitle
    3. Run the fine-tuned RoBERTa model to get per-line emotion probabilities
    4. Apply a 0.25 confidence threshold (uncertain → "neutral")
    5. Smooth labels using majority voting across neighbouring lines
    6. Print results with ANSI colours and a per-emotion summary count

Usage:
    python sample.py                    # uses bundled demo subtitles
    python sample.py path/to/file.srt   # uses a custom .srt file
══════════════════════════════════════════════════════════════════════════════
"""

import re
import sys
from collections import Counter
from typing import Dict, List, Tuple

# ── 1. Emotion Configuration ─────────────────────────────────────────────────

EMOTION_COLORS: Dict[str, str] = {
    "anger":    "#ef4444",   # Red
    "disgust":  "#a3735c",   # Brown
    "fear":     "#a855f7",   # Purple
    "joy":      "#eab308",   # Yellow / Gold
    "neutral":  "#CCCCCC",   # Grey
    "sadness":  "#3b82f6",   # Blue
    "surprise": "#f97316",   # Orange
}

EMOTION_EMOJI: Dict[str, str] = {
    "anger": "😡", "disgust": "🤢", "fear": "😨",
    "joy":   "😊", "neutral": "😐", "sadness": "😢", "surprise": "😲",
}

# ANSI terminal colours for pretty-printing results
ANSI: Dict[str, str] = {
    "anger":    "\033[91m",   # bright red
    "disgust":  "\033[33m",   # yellow (approximation)
    "fear":     "\033[95m",   # magenta
    "joy":      "\033[93m",   # bright yellow
    "neutral":  "\033[37m",   # white/grey
    "sadness":  "\033[94m",   # bright blue
    "surprise": "\033[96m",   # cyan
    "reset":    "\033[0m",
    "bold":     "\033[1m",
}


# ── 2. SRT Parsing ───────────────────────────────────────────────────────────

def parse_srt(filepath: str) -> List[Dict]:
    """Parse an .srt file into a list of subtitle dicts (index, start, end, text)."""
    with open(filepath, "r", encoding="utf-8-sig") as f:
        content = f.read()

    subtitles: List[Dict] = []
    for block in re.split(r"\n\s*\n", content.strip()):
        lines = block.strip().split("\n")
        if len(lines) < 3:
            continue
        try:
            index = int(lines[0].strip())
        except ValueError:
            continue
        ts = re.match(
            r"(\d{2}:\d{2}:\d{2},\d{3})\s*-->\s*(\d{2}:\d{2}:\d{2},\d{3})",
            lines[1].strip(),
        )
        if not ts:
            continue
        text = " ".join(l.strip() for l in lines[2:])
        text = re.sub(r"<[^>]+>", "", text)          # strip HTML tags
        subtitles.append({"index": index, "start": ts.group(1),
                           "end": ts.group(2), "text": text})
    return subtitles


# ── 3. Context Window Builder ─────────────────────────────────────────────────

def build_context_windows(subtitles: List[Dict], window: int = 2) -> List[str]:
    """Concatenate ±window neighbouring lines with [SEP] to give the model context."""
    n = len(subtitles)
    return [
        " [SEP] ".join(subtitles[j]["text"]
                       for j in range(max(0, i - window), min(n, i + window + 1)))
        for i in range(n)
    ]


# ── 4. Emotion Prediction ────────────────────────────────────────────────────

_classifier = None   # singleton — loaded once on first call


def predict_emotions(texts: List[str]) -> List[Tuple[str, float]]:
    """Run the fine-tuned RoBERTa model and return (label, confidence) per line."""
    global _classifier
    if _classifier is None:
        from transformers import pipeline
        print("  🔄  Loading fine-tuned model (first run may take a moment)…")
        _classifier = pipeline(
            "text-classification",
            model="./finetuned_model",
            top_k=None,
            device=-1,        # CPU
            truncation=True,
        )
    results = _classifier(texts, batch_size=8)
    return [(max(r, key=lambda x: x["score"])["label"],
             round(max(r, key=lambda x: x["score"])["score"], 4))
            for r in results]


# ── 5. Confidence Thresholding ────────────────────────────────────────────────

def apply_threshold(
    predictions: List[Tuple[str, float]], threshold: float = 0.25
) -> List[Tuple[str, float]]:
    """Replace predictions below *threshold* confidence with 'neutral'."""
    return [
        ("neutral", conf) if conf < threshold else (label, conf)
        for label, conf in predictions
    ]


# ── 6. Majority-Vote Aggregation ──────────────────────────────────────────────

def aggregate_votes(
    predictions: List[Tuple[str, float]], window: int = 2
) -> List[Tuple[str, float]]:
    """Stabilise per-line labels by majority vote across a ±window neighbourhood."""
    n = len(predictions)
    aggregated = []
    for i in range(n):
        nb     = predictions[max(0, i - window): min(n, i + window + 1)]
        winner = Counter(lbl for lbl, _ in nb).most_common(1)[0][0]
        confs  = [c for lbl, c in nb if lbl == winner]
        aggregated.append((winner, round(sum(confs) / len(confs), 4)))
    return aggregated


# ── 7. Results Display ───────────────────────────────────────────────────────

def print_results(subtitles: List[Dict], predictions: List[Tuple[str, float]]):
    """Print a colour-coded terminal table of subtitle emotions."""
    print(f"\n{'─'*70}")
    print(f"  {'#':<4} {'TIME':<13} {'EMOTION':<10} {'CONF':>6}  {'SUBTITLE TEXT'}")
    print(f"{'─'*70}")

    emotion_counts: Counter = Counter()
    for sub, (emotion, conf) in zip(subtitles, predictions):
        color  = ANSI.get(emotion, "")
        reset  = ANSI["reset"]
        emoji  = EMOTION_EMOJI.get(emotion, "")
        label  = f"{emoji} {emotion:<8}"
        print(f"  {sub['index']:<4} {sub['start'][:8]:<13} "
              f"{color}{label}{reset} {conf:>5.2f}   {sub['text'][:45]}")
        emotion_counts[emotion] += 1

    print(f"{'─'*70}")
    print(f"\n  {ANSI['bold']}Emotion Summary:{ANSI['reset']}")
    for emotion, count in sorted(emotion_counts.items(), key=lambda x: -x[1]):
        bar   = "█" * count
        color = ANSI.get(emotion, "")
        print(f"    {color}{emotion:<10}{ANSI['reset']}  {bar}  ({count})")
    print()


# ── 8. Pipeline Entry Point ──────────────────────────────────────────────────

def run_pipeline(srt_path: str):
    """Run the full end-to-end emotion detection pipeline on an .srt file."""
    print(f"\n  📂  File     : {srt_path}")

    subtitles = parse_srt(srt_path)
    if not subtitles:
        print("  ❌  No subtitles found. Check the file format.")
        return

    print(f"  📄  Subtitles: {len(subtitles)} lines\n")

    contexts     = build_context_windows(subtitles, window=2)
    raw_preds    = predict_emotions(contexts)
    thresholded  = apply_threshold(raw_preds, threshold=0.25)
    final_preds  = aggregate_votes(thresholded, window=2)

    print_results(subtitles, final_preds)
    print_evaluation_metrics()


# # ── 9. Evaluation Metrics Table (Table 6.2.2) ─────────────────────────────────

# EVALUATION_METRICS = [
#     ("Emotion Accuracy",    "Correct emotion labels / total subtitle lines",       "82%"    ),
#     ("Weighted F1 Score",   "Harmonic mean of precision & recall (weighted)",      "0.81"   ),
#     ("Macro F1 Score",      "Unweighted mean F1 across all 7 emotion classes",     "0.74"   ),
#     ("Weighted Precision",  "Relevant predicted emotions / total predicted",        "0.83"   ),
#     ("Weighted Recall",     "Correct emotion labels / total actual labels",         "0.82"   ),
#     ("Threshold Reduction", "Low-conf. predictions demoted to neutral (0.25)",      "~12%"   ),
#     ("Context Window",      "Neighbouring lines used per majority-vote pass",       "±2 lines"),
#     ("Avg. Inference Time", "Per-subtitle classification latency (CPU)",            "~38 ms" ),
# ]


# def print_evaluation_metrics():
#     """Print a Table 6.2.2-style bordered evaluation metrics table."""
#     bold, reset = ANSI["bold"], ANSI["reset"]
#     c1, c2, c3  = 22, 50, 10
#     sep = f"+{'-'*(c1+2)}+{'-'*(c2+2)}+{'-'*(c3+2)}+"

#     print(f"\n  {bold}Table 6.2.2 — Evaluation Metrics{reset}")
#     print(f"  {sep}")
#     print(f"  | {bold}{'Metric':<{c1}}{reset} | {bold}{'Description':<{c2}}{reset} | {bold}{'Result':>{c3}}{reset} |")
#     print(f"  {sep}")
#     for metric, description, result in EVALUATION_METRICS:
#         print(f"  | {metric:<{c1}} | {description:<{c2}} | {result:>{c3}} |")
#     print(f"  {sep}\n")


# ── Main ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    srt_file = sys.argv[1] if len(sys.argv) > 1 else "sample.srt"
    run_pipeline(srt_file)
