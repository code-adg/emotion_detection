"""
srt_utils.py
─────────────
SRT subtitle parsing, context-window generation, and color-coded SRT output.
"""

import re
from typing import List, Dict, Tuple

# ── Emotion → VLC <font color> mapping (MELD — 7 labels) ──────────────────────
EMOTION_COLORS: Dict[str, str] = {
    "anger":    "#ef4444",  # Red
    "disgust":  "#a3735c",  # Brown
    "fear":     "#a855f7",  # Purple
    "joy":      "#eab308",  # Yellow/Gold
    "neutral":  "#CCCCCC",  # Grey (lightened for subtitle readability)
    "sadness":  "#3b82f6",  # Blue
    "surprise": "#f97316",  # Orange
}


# ── SRT Parsing ───────────────────────────────────────────────────────────────

def parse_srt(filepath: str) -> List[Dict]:
    """
    Parse an .srt file into a list of subtitle entries.

    Returns
    -------
    list of dict
        Each dict has keys: index (int), start (str), end (str), text (str).
    """
    with open(filepath, "r", encoding="utf-8-sig") as f:
        content = f.read()

    # Split on blank lines (one or more) to get blocks
    blocks = re.split(r"\n\s*\n", content.strip())
    subtitles: List[Dict] = []

    for block in blocks:
        lines = block.strip().split("\n")
        if len(lines) < 3:
            continue

        try:
            index = int(lines[0].strip())
        except ValueError:
            continue

        # Timestamp line: 00:01:23,456 --> 00:01:25,789
        timestamp_match = re.match(
            r"(\d{2}:\d{2}:\d{2},\d{3})\s*-->\s*(\d{2}:\d{2}:\d{2},\d{3})",
            lines[1].strip(),
        )
        if not timestamp_match:
            continue

        start_time = timestamp_match.group(1)
        end_time = timestamp_match.group(2)
        text = " ".join(line.strip() for line in lines[2:])

        # Strip any existing HTML-style tags from the raw text
        text = re.sub(r"<[^>]+>", "", text)

        subtitles.append({
            "index": index,
            "start": start_time,
            "end": end_time,
            "text": text,
        })

    return subtitles


# ── Context-Window Builder ────────────────────────────────────────────────────

def build_context_windows(
    subtitles: List[Dict], window: int = 2
) -> List[str]:
    """
    Create a sliding context window for each subtitle line.

    For each subtitle at position *i*, concatenate the text of subtitles
    from (i - window) to (i + window) inclusive, separated by ' [SEP] '.

    Parameters
    ----------
    subtitles : list of dict
        Output of ``parse_srt``.
    window : int
        Number of lines before and after the current line to include.

    Returns
    -------
    list of str
        One context string per subtitle, same length as *subtitles*.
    """
    contexts: List[str] = []
    n = len(subtitles)

    for i in range(n):
        start = max(0, i - window)
        end = min(n, i + window + 1)
        parts = [subtitles[j]["text"] for j in range(start, end)]
        contexts.append(" [SEP] ".join(parts))

    return contexts


# ── Color-Coded SRT Generation ────────────────────────────────────────────────

def generate_colored_srt(
    subtitles: List[Dict],
    emotions: List[Tuple[str, float]],
    output_path: str,
) -> str:
    """
    Write a new .srt file with VLC-compatible <font color> tags.

    Parameters
    ----------
    subtitles : list of dict
        Original parsed subtitles.
    emotions : list of (label, confidence)
        One emotion prediction per subtitle.
    output_path : str
        Destination path for the colored .srt file.

    Returns
    -------
    str
        The *output_path* written to.
    """
    lines: List[str] = []

    for sub, (emotion, confidence) in zip(subtitles, emotions):
        color = EMOTION_COLORS.get(emotion, "#CCCCCC")
        lines.append(str(sub["index"]))
        lines.append(f"{sub['start']} --> {sub['end']}")
        lines.append(f'<font color="{color}">{sub["text"]}</font>')
        lines.append("")  # blank line separator

    with open(output_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    return output_path
