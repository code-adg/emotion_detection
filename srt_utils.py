import re
from typing import List, Dict, Tuple

EMOTION_COLORS: Dict[str, str] = {
    "anger":    "#ef4444",
    "disgust":  "#a3735c",
    "fear":     "#a855f7",
    "joy":      "#eab308",
    "neutral":  "#CCCCCC",
    "sadness":  "#3b82f6",
    "surprise": "#f97316",
}

EMOTION_EMOJI: Dict[str, str] = {
    "anger":    "😡",
    "disgust":  "🤢",
    "fear":     "😨",
    "joy":      "😊",
    "neutral":  "😐",
    "sadness":  "😢",
    "surprise": "😲",
}


def parse_srt(filepath: str) -> List[Dict]:
    with open(filepath, "r", encoding="utf-8-sig") as f:
        content = f.read()

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

        timestamp_match = re.match(
            r"(\d{2}:\d{2}:\d{2},\d{3})\s*-->\s*(\d{2}:\d{2}:\d{2},\d{3})",
            lines[1].strip(),
        )
        if not timestamp_match:
            continue

        start_time = timestamp_match.group(1)
        end_time = timestamp_match.group(2)
        text = " ".join(line.strip() for line in lines[2:])

        text = re.sub(r"<[^>]+>", "", text)

        subtitles.append({
            "index": index,
            "start": start_time,
            "end": end_time,
            "text": text,
        })

    return subtitles


def build_context_windows(
    subtitles: List[Dict], window: int = 2
) -> List[str]:
    contexts: List[str] = []
    n = len(subtitles)

    for i in range(n):
        start = max(0, i - window)
        end = min(n, i + window + 1)
        parts = [subtitles[j]["text"] for j in range(start, end)]
        contexts.append(" [SEP] ".join(parts))

    return contexts


def generate_colored_srt(
    subtitles: List[Dict],
    emotions: List[Tuple[str, float]],
    output_path: str,
    with_emoji: bool = False,
) -> str:
    lines: List[str] = []

    for sub, (emotion, confidence) in zip(subtitles, emotions):
        color = EMOTION_COLORS.get(emotion, "#CCCCCC")
        emoji = EMOTION_EMOJI.get(emotion, "") if with_emoji else ""
        text = f"{emoji} {sub['text']}" if emoji else sub["text"]
        lines.append(str(sub["index"]))
        lines.append(f"{sub['start']} --> {sub['end']}")
        lines.append(f'<font color="{color}">{text}</font>')
        lines.append("")

    with open(output_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    return output_path
