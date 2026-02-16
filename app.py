"""
app.py
───────
Flask application for Context-Aware Emotion Detection from Subtitle Text.
"""

import os
import uuid

from flask import Flask, render_template, request, jsonify, send_from_directory

from srt_utils import parse_srt, build_context_windows, generate_colored_srt
from emotion_model import predict_emotions, apply_threshold, aggregate_votes

# ── App Setup ─────────────────────────────────────────────────────────────────

app = Flask(__name__)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
UPLOAD_FOLDER = os.path.join(BASE_DIR, "uploads")
OUTPUT_FOLDER = os.path.join(BASE_DIR, "outputs")

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
app.config["OUTPUT_FOLDER"] = OUTPUT_FOLDER
app.config["MAX_CONTENT_LENGTH"] = 5 * 1024 * 1024  # 5 MB max upload


# ── Routes ────────────────────────────────────────────────────────────────────

@app.route("/")
def index():
    """Serve the main UI."""
    return render_template("index.html")


@app.route("/upload", methods=["POST"])
def upload():
    """
    Accept an .srt file, run the full emotion-detection pipeline,
    and return results as JSON.
    """
    # ── Validate upload ───────────────────────────────────────────────────
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded."}), 400

    file = request.files["file"]
    if file.filename == "":
        return jsonify({"error": "Empty filename."}), 400

    if not file.filename.lower().endswith(".srt"):
        return jsonify({"error": "Only .srt files are accepted."}), 400

    # ── Save uploaded file ────────────────────────────────────────────────
    unique_id = uuid.uuid4().hex[:8]
    safe_name = f"{unique_id}_{file.filename}"
    upload_path = os.path.join(UPLOAD_FOLDER, safe_name)
    file.save(upload_path)

    try:
        # ── Pipeline ─────────────────────────────────────────────────────
        # Step 1: Parse SRT
        subtitles = parse_srt(upload_path)
        if not subtitles:
            return jsonify({"error": "Could not parse any subtitles from the file."}), 400

        # Step 2: Extract individual subtitle texts for classification
        individual_texts = [sub["text"] for sub in subtitles]

        # Step 3: Predict emotions on individual lines
        raw_predictions = predict_emotions(individual_texts)

        # Step 4: Apply confidence threshold (lower for 28-class model)
        thresholded = apply_threshold(raw_predictions, threshold=0.25)

        # Step 5: Context-aware aggregation via majority voting
        #         This is where context awareness happens — each subtitle's
        #         final label is stabilized by its neighbours' predictions
        final_predictions = aggregate_votes(thresholded, window=2)

        # Step 6: Generate color-coded SRT
        output_name = f"colored_{safe_name}"
        output_path = os.path.join(OUTPUT_FOLDER, output_name)
        generate_colored_srt(subtitles, final_predictions, output_path)

        # ── Build response ───────────────────────────────────────────────
        results = []
        for sub, (emotion, confidence) in zip(subtitles, final_predictions):
            results.append({
                "index": sub["index"],
                "start": sub["start"],
                "end": sub["end"],
                "text": sub["text"],
                "emotion": emotion,
                "confidence": confidence,
            })

        return jsonify({
            "success": True,
            "count": len(results),
            "results": results,
            "download": f"/download/{output_name}",
        })

    except Exception as e:
        return jsonify({"error": f"Processing failed: {str(e)}"}), 500


@app.route("/download/<filename>")
def download(filename):
    """Serve a generated color-coded .srt file for download."""
    return send_from_directory(
        app.config["OUTPUT_FOLDER"],
        filename,
        as_attachment=True,
    )


# ── Entry Point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("\n  🚀  Starting Emotion Detection Server …")
    print("  📍  Open http://localhost:5000 in your browser\n")
    app.run(debug=True, host="0.0.0.0", port=5000, use_reloader=False)
