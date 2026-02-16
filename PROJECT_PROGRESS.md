# 📋 Project Progress Report
## Context-Aware Emotion Detection from Subtitle Text Using NLP
### Final Year Engineering Project

---

## 🗓️ Development Timeline

**Start Date:** February 10, 2026  
**Status:** In Progress (UI Overhaul & Integration)

---

## Phase 1: Project Setup & Core Implementation
**Date:** February 10, 2026

### Step 1 — Requirements Analysis & Planning
- Analyzed the full project requirements for a local, CPU-only web application
- Defined three features:
  - **Feature 1 (Active):** SRT-only emotion processing with NLP pipeline
  - **Feature 2 (UI Only):** Video + SRT upload — marked "In Progress"
  - **Feature 3 (UI Only):** Video-only ASR + Emotion — marked "Future Work"
- Created an implementation plan covering project structure, backend modules, frontend design, and verification strategy

### Step 2 — Project Structure Created
```
d:\FYP\Projects\New folder\
├── app.py                  ← Flask server & routes
├── emotion_model.py        ← HuggingFace emotion classifier
├── srt_utils.py            ← SRT parsing & colored output
├── requirements.txt        ← Python dependencies
├── sample.srt              ← Test file (10 subtitles)
├── templates/
│   └── index.html          ← Main UI page
├── static/
│   ├── style.css           ← Glassmorphism dark theme
│   └── script.js           ← Upload & results logic
├── uploads/                ← (auto-created) temp uploads
└── outputs/                ← (auto-created) colored SRTs
```

### Step 3 — Backend Implementation

#### `srt_utils.py` — Subtitle Processing Module
| Function | Purpose |
|----------|---------|
| `parse_srt(filepath)` | Parses `.srt` file → list of `{index, start, end, text}` dicts |
| `build_context_windows(subtitles, window=2)` | Creates sliding context window (2 before + current + 2 after) |
| `generate_colored_srt(subtitles, emotions, output_path)` | Creates VLC-compatible color-coded `.srt` with `<font color>` tags |

#### `emotion_model.py` — Emotion Classification Module
| Function | Purpose |
|----------|---------|
| `predict_emotions(texts)` | Batch-classifies text using HuggingFace transformer |
| `apply_threshold(predictions, threshold)` | Replaces low-confidence predictions with "neutral" |
| `aggregate_votes(predictions, window=2)` | Majority-vote stabilization across overlapping windows |

**Initial Model:** `j-hartmann/emotion-english-distilroberta-base` (7 emotions)

#### `app.py` — Flask Application
| Route | Method | Purpose |
|-------|--------|---------|
| `/` | GET | Serve the main UI |
| `/upload` | POST | Accept `.srt`, run full pipeline, return JSON |
| `/download/<filename>` | GET | Serve color-coded `.srt` for download |

**Processing Pipeline:**
1. Save uploaded `.srt` file
2. Parse subtitle entries
3. Extract individual subtitle texts
4. Predict emotions using transformer model
5. Apply confidence thresholding
6. Run majority-vote aggregation (context-aware stabilization)
7. Generate color-coded `.srt` output
8. Return results as JSON

### Step 4 — Frontend Implementation

#### `templates/index.html`
- Three feature cards in a responsive layout
- Feature 1: Active with file upload (drag & drop), analyze button, results table, download button
- Feature 2: Disabled with "Feature under development" overlay
- Feature 3: Disabled with "Planned for next phase" overlay
- Ethics disclaimer at the bottom

#### `static/style.css`
- Dark gradient background (`#0f0c29` → `#302b63` → `#24243e`)
- Glassmorphism card effects with `backdrop-filter: blur()`
- Emotion-colored pill badges for each emotion label
- Confidence bar visualizations
- Smooth hover animations and transitions
- Fully responsive design

#### `static/script.js`
- Drag-and-drop file upload with visual feedback
- `FormData` POST to `/upload` endpoint
- Loading spinner animation during processing
- Dynamic results table rendering
- Error toast notifications
- Download button show/hide logic

### Step 5 — Dependencies Installed
```
pip install flask transformers torch
```
- Flask (web framework) — already installed
- Transformers (HuggingFace) — installed v5.1.0
- PyTorch (inference engine) — already installed v2.10.0

---

## Phase 2: Bug Fixes & Improvements
**Date:** February 10–11, 2026

### Step 6 — Fixed Flask Reloader Issue
**Problem:** Flask's debug mode `watchdog` was detecting file changes inside the `torch` package directory and continuously restarting the server.

**Fix:** Added `use_reloader=False` to `app.run()`:
```python
app.run(debug=True, host="0.0.0.0", port=5000, use_reloader=False)
```

### Step 7 — Model Change: Switched to GoEmotions
**Request:** User asked to switch to `SamLowe/roberta-base-go_emotions`

**Changes Made:**
- Updated `emotion_model.py` → new model name
- Updated `srt_utils.py` → expanded color mapping from 7 to **28 emotions**
- Updated `static/style.css` → added CSS classes for all 28 emotion pills

**GoEmotions labels (28):** admiration, amusement, anger, annoyance, approval, caring, confusion, curiosity, desire, disappointment, disapproval, disgust, embarrassment, excitement, fear, gratitude, grief, joy, love, nervousness, optimism, pride, realization, relief, remorse, sadness, surprise, neutral

### Step 8 — Fixed "All Admiration" Prediction Bug
**Problem:** All 10 test subtitles were being classified as "admiration" regardless of content.

**Root Cause:** The context window was concatenating 5 subtitle lines together before feeding to the model. The mixed emotional content confused the model into defaulting to the most generic positive label.

**Fix in `app.py`:**
```diff
- contexts = build_context_windows(subtitles, window=2)
- raw_predictions = predict_emotions(contexts)
+ individual_texts = [sub["text"] for sub in subtitles]
+ raw_predictions = predict_emotions(individual_texts)
```

- Now classifies **each subtitle individually** for accurate per-line emotion
- Lowered confidence threshold from `0.4` → `0.25` (28 classes produce lower per-class scores)
- **Context-awareness** is preserved through the **majority-vote aggregation** step

---

## Phase 3: Fine-Tuning Pipeline
**Date:** February 11, 2026

### Step 9 — User Asked About Fine-Tuning
User asked how to fine-tune the model for better subtitle-specific predictions.

**Guidance Provided:**
- Fine-tuning = transfer learning on domain-specific data
- Recommended the MELD dataset (Friends TV show dialogues, ~13K utterances)
- Explained benefits: GoEmotions was trained on Reddit comments, subtitles have different patterns

### Step 10 — Created `prepare_meld.py`
**Purpose:** Download and prepare the MELD dataset for fine-tuning.

**Initial version:** Used HuggingFace `datasets` library → problem: downloads **10GB** of video/audio/text.

**Updated version (v2):** Downloads only text CSV files (~3 MB) directly from MELD's GitHub repository.

**Final version (v3):** Added robust CSV parsing with 3 fallback strategies + built-in curated 290-sample subtitle dataset as fallback if download fails.

**Output:** `labeled_subtitles.csv` with ~9,853 dialogue samples across 7 emotions.

### Step 11 — Created `finetune.py`
**Purpose:** Fine-tune `SamLowe/roberta-base-go_emotions` on MELD subtitle data.

**Configuration:**
| Parameter | Value | Rationale |
|-----------|-------|-----------|
| Base Model | `SamLowe/roberta-base-go_emotions` | Strong pretrained emotion model |
| Learning Rate | `2e-5` | Standard for RoBERTa fine-tuning |
| Epochs | 4 | Enough for convergence without overfitting |
| Batch Size | 16 | Balanced speed vs memory |
| Warmup Ratio | 10% | Prevents early training instability |
| Weight Decay | 0.01 | Light regularization |
| Max Token Length | 128 | Sufficient for subtitle lines |

**Features:**
- Train/validation split (85/15) with stratification
- Accuracy and weighted F1 metrics per epoch
- Detailed classification report at the end
- Training loss curve visualization (`training_curves.png`)
- Saves classification report to `classification_report.txt`

### Step 12 — Fixed `problem_type` Error
**Problem:** Training crashed with `ValueError: Target size (torch.Size([16])) must be the same as input size (torch.Size([16, 7]))`.

**Root Cause:** GoEmotions model defaults to **multi-label classification** (BCEWithLogitsLoss), expecting one-hot encoded labels. We needed **single-label** (CrossEntropyLoss) with integer labels.

**Fix:**
```python
model = AutoModelForSequenceClassification.from_pretrained(
    BASE_MODEL,
    ...
    problem_type="single_label_classification",  # ← added this line
)
```

### Step 13 — Training Successful
**Results:**
- Training completed on Google Colab T4 GPU in ~6 minutes
- **Final Accuracy:** 60.22% (vs random chance ~14%)
- **Best F1 Score:** 0.5918
- **Per-Class Performance:**
  - Neutral: 0.76 F1 (excellent)
  - Surprise: 0.57 F1 (good)
  - Joy: 0.54 F1 (decent)
  - Anger: 0.38 F1 (moderate)
  - Fear/Disgust: Lower due to small sample size

**Outcome:**
- Downloaded `finetuned_model` folder (~500 MB)
- Updated `emotion_model.py` to prompt user to use `./finetuned_model`
- Updated docstrings to reflect 7-class MELD schema (anger, disgust, fear, joy, neutral, sadness, surprise)

---

## Phase 4: UI Overhaul & Polish
**Date:** February 11, 2026

### Step 14 — Complete UI Redesign
**Objective:** Transform the "AI-generated" look into a premium, professional interface.

**Changes Made:**
1. **Tech Stack Update:** Added **Bootstrap 5** for robust grid & components.
2. **Visual Style:**
   - Replaced purple gradients with deep emerald/black theme (`#0a0a0f`)
   - Added **custom cursor** with magnetic hover effects
   - Implemented **Inter/Outfit** typography
3. **New Features:**
   - **Splash Screen:** Cinematic intro animation with logo reveal
   - **Pipeline Visualization:** Real-time animated steps showing the NLP process (Parse → Tokenize → Predict → Aggregate)
   - **Lazy Loading:** IntersectionObserver for smooth scroll animations
   - **Emotion Summary Cards:** Statistical breakdown of detected emotions
4. **Code Operations:**
   - Refactored `index.html` (Bootstrap integration)
   - Rewrote `style.css` (700+ lines of custom CSS variables & animations)
   - Rewrote `script.js` (Pipeline logic, confetti effects, cursor tracking)

### Step 15 — System Verification
- Verified `app.py` serves the new UI correctly
- Confirmed file upload → pipeline animation → results table flow works
- Validated VLC color key compatibility for all 7 emotions

---

## Technology Stack Summary

| Component | Technology |
|-----------|-----------|
| Backend | Python 3.11 + Flask |
| NLP Model | RoBERTa (GoEmotions → fine-tuned on MELD) |
| ML Framework | PyTorch (CPU-only) |
| Transformers | HuggingFace Transformers v5.1.0 |
| Frontend | HTML5 + CSS3 + JavaScript (Vanilla) |
| Design | Glassmorphism, dark gradient theme |
| Subtitle Format | SRT with VLC `<font color>` tags |
| Dataset | MELD (Friends TV show dialogues) |

---

## Key Design Decisions

1. **CPU-only inference** — no GPU required, runs on any laptop
2. **Individual line classification** — more accurate than context-concatenated text
3. **Context-awareness via aggregation** — majority voting across neighbours stabilizes predictions
4. **Confidence thresholding** — prevents weak predictions from affecting output
5. **Transfer learning** — fine-tuning a pretrained model rather than training from scratch
6. **VLC-compatible coloring** — uses `<font color>` tags supported by VLC Media Player
7. **Ethical disclaimers** — UI clearly states this is not a medical/clinical tool
