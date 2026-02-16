# 🎬 Context-Aware Emotion Detection from Subtitle Text

A web application that detects emotions in subtitle (`.srt`) files using a fine-tuned NLP transformer model and generates **color-coded subtitles** for VLC Media Player.

> **Final Year Engineering Project**

---

## ✨ Features

- **Upload & Analyze** — Drag-and-drop `.srt` files for instant emotion analysis
- **Fine-Tuned Model** — RoBERTa model fine-tuned on the MELD (Friends TV show) dataset for 7 emotions
- **Context-Aware** — Majority-vote aggregation across neighbouring subtitles stabilizes predictions
- **Color-Coded Output** — Download emotion-colored `.srt` files playable in VLC
- **CPU-Only** — Runs on any machine without a GPU

## 🎭 Supported Emotions

| Emotion | Color |
|---------|-------|
| Joy | 🟢 Green |
| Sadness | 🔵 Blue |
| Anger | 🔴 Red |
| Fear | 🟣 Purple |
| Surprise | 🟡 Yellow |
| Disgust | 🟤 Brown |
| Neutral | ⚪ White |

---

## 🚀 Getting Started

### Prerequisites

- Python 3.10+
- pip

### 1. Clone the Repository

```bash
git clone https://github.com/YOUR_USERNAME/subtitle-emotion-detection.git
cd subtitle-emotion-detection
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Download the Fine-Tuned Model

The fine-tuned model (~500 MB) is too large for GitHub. Download it from:

> 📥 **[Download Link — Google Drive / Hugging Face]** *(add your link here)*

After downloading, place the model files in a `finetuned_model/` folder in the project root:

```
project-root/
├── finetuned_model/
│   ├── config.json
│   ├── model.safetensors
│   ├── tokenizer.json
│   ├── tokenizer_config.json
│   └── training_args.bin
├── app.py
└── ...
```

### 4. Run the Application

```bash
python app.py
```

Open **http://localhost:5000** in your browser.

---

## 📁 Project Structure

```
├── app.py                  # Flask web server & API routes
├── emotion_model.py        # Emotion classification (HuggingFace pipeline)
├── srt_utils.py            # SRT parsing, context windows & colored output
├── finetune.py             # Fine-tuning script (RoBERTa on MELD)
├── prepare_meld.py         # MELD dataset download & preparation
├── compare_models.py       # Base vs fine-tuned model comparison
├── requirements.txt        # Python dependencies
├── sample.srt              # Sample subtitle file for testing
├── demo.srt                # Demo subtitle file
├── PROJECT_PROGRESS.md     # Detailed development log
├── templates/
│   └── index.html          # Main UI page
├── static/
│   ├── style.css           # Dark theme with glassmorphism
│   └── script.js           # Upload logic, animations & results
├── uploads/                # (auto-created) temporary uploads
└── outputs/                # (auto-created) color-coded SRT outputs
```

---

## ⚙️ How It Works

```
Upload .srt → Parse Subtitles → Classify Each Line → Apply Threshold → Majority-Vote Aggregation → Color-Coded .srt
```

1. **Parse** — Extract subtitle entries from the `.srt` file
2. **Classify** — Run each subtitle line through the fine-tuned RoBERTa model
3. **Threshold** — Replace low-confidence predictions (<0.25) with "neutral"
4. **Aggregate** — Majority-vote across a sliding window (±2 neighbours) for context-aware stabilization
5. **Output** — Generate a color-coded `.srt` file with `<font color>` tags compatible with VLC

---

## 🛠️ Technology Stack

| Component | Technology |
|-----------|------------|
| Backend | Python 3 + Flask |
| NLP Model | RoBERTa (GoEmotions → fine-tuned on MELD) |
| ML Framework | PyTorch (CPU-only) |
| Transformers | HuggingFace Transformers |
| Frontend | HTML5 + CSS3 + JavaScript |
| UI Design | Dark theme, glassmorphism, Bootstrap 5 |
| Subtitle Format | SRT with VLC `<font color>` tags |

---

## 🔬 Fine-Tuning Details

- **Base Model:** `SamLowe/roberta-base-go_emotions` (28-class Reddit emotions)
- **Dataset:** MELD (Multimodal EmotionLines Dataset — Friends TV show, ~9,800 dialogues)
- **Training:** 4 epochs, lr=2e-5, batch size 16
- **Result:** 60.2% accuracy / 0.59 F1 (vs ~14% random chance for 7 classes)

To re-run fine-tuning:

```bash
python prepare_meld.py      # Download & prepare MELD dataset
python finetune.py           # Fine-tune the model (GPU recommended)
```

---

## 📝 API Endpoints

| Route | Method | Description |
|-------|--------|-------------|
| `/` | GET | Serve the main web UI |
| `/upload` | POST | Upload `.srt` file, returns emotion analysis as JSON |
| `/download/<filename>` | GET | Download the color-coded `.srt` file |

---

## ⚠️ Disclaimer

This tool is intended for **academic and research purposes only**. It is not a substitute for professional psychological or clinical assessment. Emotion detection from text has inherent limitations and should not be used for diagnostic purposes.

---

## 📄 License

This project was developed as a Final Year Engineering Project.
