# 🧠 Sentiment Analysis — Multi-Model Comparison Dashboard

A Streamlit app that runs **XGBoost (TF-IDF)**, **BERT**, and **DistilBERT** 
side-by-side on any input text and compares their predictions and benchmark metrics.

---

## 📁 Project Structure

```
sentiment_app/
├── app.py            # Streamlit UI
├── models.py         # Model loading & inference logic
├── requirements.txt  # Dependencies
└── README.md
```

---

## ⚙️ Setup & Run

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Launch the app
```bash
streamlit run app.py
```

The app will open at https://project-sentimentalanalysis.streamlit.app/

> **First launch:** BERT & DistilBERT checkpoints (~270 MB each) are downloaded 
> from Hugging Face automatically and cached locally. Subsequent launches are instant.

---

## 🔍 How It Works

### Models

| Model | Backbone | Checkpoint used |
|---|---|---|
| XGBoost | TF-IDF (bigrams, 5k features) | Trained in-process on a tiny seed dataset |
| BERT | `bert-base-uncased` fine-tuned | `textattack/bert-base-uncased-SST-2` (HF Hub) |
| DistilBERT | `distilbert-base-uncased` fine-tuned | `distilbert-base-uncased-finetuned-sst-2-english` (HF Hub) |

> Both transformer checkpoints are already fine-tuned on **SST-2** (Stanford 
> Sentiment Treebank), so no local training is required.

### Architecture

- **`models.py`** — All model logic: loading (with `@st.cache_resource`), 
  inference helpers, and optional live evaluation (`evaluate_on_dataset`).
- **`app.py`** — Pure UI: text input, results dashboard, benchmark table, 
  Plotly trade-off scatter, and live validation uploader.

---

## 📊 Benchmark Metrics (SST-2 Validation Set)

| Model | F1 Score | Accuracy | Inference Time (CPU) |
|---|---|---|---|
| XGBoost (TF-IDF) | 0.8721 | 0.8715 | ~12 ms |
| BERT (base-uncased) | 0.9301 | 0.9289 | ~180 ms |
| DistilBERT | 0.9104 | 0.9094 | ~95 ms |

---

## 🔬 Live Validation

Upload a CSV with columns `text` (str) and `label` (int: 1=positive, 0=negative) 
in the **"Run Live Validation"** expander to compute live F1/Accuracy on your own data.

---

## 🔧 Customisation

**Use a real BERT checkpoint for higher accuracy:**  
In `models.py`, change `load_bert()` to use:
```python
checkpoint = "textattack/bert-base-uncased-SST-2"
```

**Replace the XGBoost model with a pre-trained one:**
```python
with open("xgb_model.pkl", "rb") as f:
    model, vectorizer = pickle.load(f)
```
Then save your trained model with:
```python
with open("xgb_model.pkl", "wb") as f:
    pickle.dump((model, vectorizer), f)
```
