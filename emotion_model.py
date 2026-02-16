"""
emotion_model.py
─────────────────
Emotion classification using a fine-tuned HuggingFace transformer.

Base Model: SamLowe/roberta-base-go_emotions
Fine-tuned on: MELD subtitle data
Labels: 7 emotions (anger, disgust, fear, joy, neutral, sadness, surprise)
"""

from collections import Counter
from typing import List, Tuple

from transformers import pipeline

# ── Singleton Model Loader ────────────────────────────────────────────────────

_classifier = None


def _get_classifier():
    """
    Lazy-load the emotion classification pipeline (CPU-only).
    The model (~300 MB) is downloaded automatically on first use and
    cached locally by HuggingFace.
    """
    global _classifier
    if _classifier is None:
        _classifier = pipeline(
            "text-classification",
            model="./finetuned_model",
            top_k=None,          # return scores for all 7 labels
            device=-1,           # force CPU
            truncation=True,
        )
    return _classifier


# ── Prediction ────────────────────────────────────────────────────────────────

def predict_emotions(texts: List[str]) -> List[Tuple[str, float]]:
    """
    Classify a list of context-windowed subtitle strings.

    Parameters
    ----------
    texts : list of str
        One context string per subtitle.

    Returns
    -------
    list of (label, confidence)
        The top-scoring emotion label and its confidence for each text.
    """
    classifier = _get_classifier()
    results = classifier(texts, batch_size=8)

    predictions: List[Tuple[str, float]] = []
    for result in results:
        # result is a list of dicts sorted by score desc
        top = max(result, key=lambda x: x["score"])
        predictions.append((top["label"], round(top["score"], 4)))

    return predictions


# ── Confidence Thresholding ───────────────────────────────────────────────────

def apply_threshold(
    predictions: List[Tuple[str, float]],
    threshold: float = 0.3,
) -> List[Tuple[str, float]]:
    """
    Replace predictions whose confidence falls below *threshold* with
    ("neutral", confidence) to avoid unreliable labels.

    Parameters
    ----------
    predictions : list of (label, confidence)
    threshold : float
        Minimum confidence to keep the predicted label.

    Returns
    -------
    list of (label, confidence)
        Filtered predictions.
    """
    filtered: List[Tuple[str, float]] = []
    for label, conf in predictions:
        if conf < threshold:
            filtered.append(("neutral", conf))
        else:
            filtered.append((label, conf))
    return filtered


# ── Majority-Vote Aggregation ─────────────────────────────────────────────────

def aggregate_votes(
    predictions: List[Tuple[str, float]],
    window: int = 2,
) -> List[Tuple[str, float]]:
    """
    Stabilize emotion labels by majority-voting across overlapping windows.

    For each subtitle at position *i*, collect the labels of subtitles in
    the range [i - window, i + window] and pick the most common label.
    The confidence returned is the average confidence of all entries that
    voted for the winning label.

    Parameters
    ----------
    predictions : list of (label, confidence)
    window : int
        Number of neighbours on each side to include in the vote.

    Returns
    -------
    list of (label, confidence)
        Aggregated (stabilized) predictions.
    """
    n = len(predictions)
    aggregated: List[Tuple[str, float]] = []

    for i in range(n):
        start = max(0, i - window)
        end = min(n, i + window + 1)

        # Collect labels in the neighbourhood
        neighbourhood = predictions[start:end]
        labels = [lbl for lbl, _ in neighbourhood]

        # Majority vote
        counter = Counter(labels)
        winner, _ = counter.most_common(1)[0]

        # Average confidence of the winning label
        winning_confs = [c for lbl, c in neighbourhood if lbl == winner]
        avg_conf = round(sum(winning_confs) / len(winning_confs), 4)

        aggregated.append((winner, avg_conf))

    return aggregated
