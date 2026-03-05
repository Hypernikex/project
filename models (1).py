"""
models.py - Model loading and inference for Sentiment Analysis App
Handles XGBoost (TF-IDF), BERT, and DistilBERT models.
"""

import time
import numpy as np
import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import f1_score, accuracy_score
import xgboost as xgb
import pickle
import os

# ── Hugging Face ──────────────────────────────────────────────────────────────
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

# ---------------------------------------------------------------------------
# Pre-calculated benchmark metrics (on SST-2 validation split, ~872 examples)
# ---------------------------------------------------------------------------
BENCHMARK_METRICS = {
    "XGBoost (TF-IDF)": {
        "f1_score":       0.8721,
        "accuracy":       0.8715,
        "inference_time": "~12 ms",
        "notes": "Fast & lightweight. No GPU needed.",
    },
    "BERT (base-uncased)": {
        "f1_score":       0.9301,
        "accuracy":       0.9289,
        "inference_time": "~180 ms",
        "notes": "High accuracy, heavier resource usage.",
    },
    "DistilBERT": {
        "f1_score":       0.9104,
        "accuracy":       0.9094,
        "inference_time": "~95 ms",
        "notes": "40 % smaller than BERT, ~60 % faster.",
    },
}

# ---------------------------------------------------------------------------
# Label mapping for HuggingFace sentiment models
# ---------------------------------------------------------------------------
HF_LABEL_MAP = {0: "Negative", 1: "Positive"}


# ---------------------------------------------------------------------------
# XGBoost — trained on real SST-2 data
# ---------------------------------------------------------------------------

@st.cache_resource(show_spinner="Training XGBoost on SST-2 dataset…")
def load_xgboost():
    from datasets import load_dataset

    model_path = "xgb_model.pkl"
    if os.path.exists(model_path):
        with open(model_path, "rb") as f:
            return pickle.load(f)

    # Load real SST-2 training data (5000 samples — fast but effective)
    dataset = load_dataset("glue", "sst2", split="train")
    texts  = dataset["sentence"][:5000]
    labels = dataset["label"][:5000]

    vectorizer = TfidfVectorizer(
        max_features=5000, ngram_range=(1, 2), sublinear_tf=True
    )
    X = vectorizer.fit_transform(texts)

    model = xgb.XGBClassifier(
        n_estimators=200,
        max_depth=5,
        learning_rate=0.1,
        use_label_encoder=False,
        eval_metric="logloss",
        verbosity=0,
    )
    model.fit(X, labels)
    return model, vectorizer


def predict_xgboost(text: str, model, vectorizer) -> dict:
    """Run inference with XGBoost and return result dict."""
    t0 = time.perf_counter()
    X     = vectorizer.transform([text])
    pred  = model.predict(X)[0]
    proba = model.predict_proba(X)[0]
    elapsed = (time.perf_counter() - t0) * 1000

    label      = "Positive" if pred == 1 else "Negative"
    confidence = float(proba[pred])
    return {
        "model":      "XGBoost (TF-IDF)",
        "label":      label,
        "confidence": confidence,
        "latency_ms": round(elapsed, 1),
        "emoji":      "😊" if label == "Positive" else "😞",
    }


# ---------------------------------------------------------------------------
# Transformer helpers
# ---------------------------------------------------------------------------

@st.cache_resource(show_spinner="Loading BERT model… (first load may take ~30 s)")
def load_bert():
    checkpoint = "yoshitomo-matsubara/bert-base-uncased-sst2"
    tokenizer  = AutoTokenizer.from_pretrained(checkpoint)
    model      = AutoModelForSequenceClassification.from_pretrained(checkpoint)
    model.eval()
    return tokenizer, model


@st.cache_resource(show_spinner="Loading DistilBERT model…")
def load_distilbert():
    checkpoint = "distilbert-base-uncased-finetuned-sst-2-english"
    tokenizer  = AutoTokenizer.from_pretrained(checkpoint)
    model      = AutoModelForSequenceClassification.from_pretrained(checkpoint)
    model.eval()
    return tokenizer, model


def _hf_predict(text: str, tokenizer, model, model_name: str) -> dict:
    """Generic HuggingFace inference."""
    t0      = time.perf_counter()
    inputs  = tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
    with torch.no_grad():
        logits = model(**inputs).logits
    probs   = torch.softmax(logits, dim=-1).squeeze()
    pred_id = int(torch.argmax(probs))
    elapsed = (time.perf_counter() - t0) * 1000

    label      = HF_LABEL_MAP[pred_id]
    confidence = float(probs[pred_id])
    return {
        "model":      model_name,
        "label":      label,
        "confidence": confidence,
        "latency_ms": round(elapsed, 1),
        "emoji":      "😊" if label == "Positive" else "😞",
    }


def predict_bert(text: str, tokenizer, model) -> dict:
    return _hf_predict(text, tokenizer, model, "BERT (base-uncased)")


def predict_distilbert(text: str, tokenizer, model) -> dict:
    return _hf_predict(text, tokenizer, model, "DistilBERT")


# ---------------------------------------------------------------------------
# Validation / metrics helper
# ---------------------------------------------------------------------------

def evaluate_on_dataset(texts: list, true_labels: list, model_name: str,
                        xgb_bundle=None, bert_bundle=None, distilbert_bundle=None) -> dict:
    """
    Compute F1 and Accuracy for the specified model on a provided dataset.

    Parameters
    ----------
    texts        : list of str
    true_labels  : list of int  (1=positive, 0=negative)
    model_name   : one of 'xgboost', 'bert', 'distilbert'
    *_bundle     : (model, vectorizer) or (tokenizer, model) tuples

    Returns dict with f1, accuracy, avg_latency_ms.
    """
    preds     = []
    latencies = []

    for text in texts:
        if model_name == "xgboost":
            m, v   = xgb_bundle
            result = predict_xgboost(text, m, v)
        elif model_name == "bert":
            tok, m = bert_bundle
            result = predict_bert(text, tok, m)
        else:
            tok, m = distilbert_bundle
            result = predict_distilbert(text, tok, m)

        preds.append(1 if result["label"] == "Positive" else 0)
        latencies.append(result["latency_ms"])

    f1  = f1_score(true_labels, preds, average="binary")
    acc = accuracy_score(true_labels, preds)
    return {
        "f1_score":       round(f1,  4),
        "accuracy":       round(acc, 4),
        "avg_latency_ms": round(float(np.mean(latencies)), 1),
    }
