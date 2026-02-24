"""
app.py - Streamlit UI for Multi-Model Sentiment Analysis Dashboard
Run with:  streamlit run app.py
"""

import streamlit as st
import pandas as pd
import time

from models import (
    load_xgboost,
    load_bert,
    load_distilbert,
    predict_xgboost,
    predict_bert,
    predict_distilbert,
    BENCHMARK_METRICS,
)

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Sentiment Analysis Comparison",
    page_icon="🧠",
    layout="wide",
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    /* Card container */
    .result-card {
        background: #1e1e2e;
        border-radius: 12px;
        padding: 1.4rem 1.6rem;
        border: 1px solid #313244;
        text-align: center;
    }
    .result-card h3 { color: #cdd6f4; font-size: 1rem; margin-bottom: 0.3rem; }
    .result-positive { color: #a6e3a1; font-size: 2rem; font-weight: 700; margin: 0.2rem 0; }
    .result-negative { color: #f38ba8; font-size: 2rem; font-weight: 700; margin: 0.2rem 0; }
    .result-conf { color: #a6adc8; font-size: 0.85rem; }
    .result-latency { color: #89b4fa; font-size: 0.8rem; margin-top: 0.4rem; }

    /* Section headers */
    .section-title {
        font-size: 1.1rem;
        font-weight: 600;
        color: #89dceb;
        border-left: 4px solid #89b4fa;
        padding-left: 0.6rem;
        margin: 1.5rem 0 0.8rem;
    }

    /* Metric table tweaks */
    thead tr th { background: #181825 !important; color: #cdd6f4 !important; }
</style>
""", unsafe_allow_html=True)


# ── Header ────────────────────────────────────────────────────────────────────
st.title("🧠 Sentiment Analysis — Model Comparison Dashboard")
st.caption(
    "Compare **XGBoost (TF-IDF)**, **BERT**, and **DistilBERT** side-by-side "
    "on any text you provide."
)
st.divider()


# ── Load models (cached) ──────────────────────────────────────────────────────
with st.spinner("Initialising models…"):
    xgb_model, xgb_vectorizer = load_xgboost()
    bert_tokenizer, bert_model = load_bert()
    distilbert_tokenizer, distilbert_model = load_distilbert()


# ── Input section ─────────────────────────────────────────────────────────────
st.markdown('<div class="section-title">📝 Input Text</div>', unsafe_allow_html=True)

user_text = st.text_area(
    label="Paste or type your text here:",
    placeholder="e.g. "The movie was an absolute masterpiece — I loved every second of it!"",
    height=140,
    label_visibility="collapsed",
)

col_btn, col_hint = st.columns([1, 5])
with col_btn:
    run_btn = st.button("🚀 Run Analysis", type="primary", use_container_width=True)
with col_hint:
    st.caption("Click **Run Analysis** to predict sentiment with all three models simultaneously.")


# ── Analysis ──────────────────────────────────────────────────────────────────
if run_btn:
    # ── Validation ──
    if not user_text.strip():
        st.error("⚠️  Please enter some text before running the analysis.")
        st.stop()

    if len(user_text.strip()) < 5:
        st.warning("⚠️  The text is very short. Results may be unreliable.")

    # ── Run all three models ──
    with st.spinner("Running inference across all models…"):
        try:
            results = {
                "XGBoost (TF-IDF)":  predict_xgboost(user_text, xgb_model, xgb_vectorizer),
                "BERT (base-uncased)": predict_bert(user_text, bert_tokenizer, bert_model),
                "DistilBERT":         predict_distilbert(user_text, distilbert_tokenizer, distilbert_model),
            }
        except Exception as exc:
            st.error(f"❌ An error occurred during inference: {exc}")
            st.stop()

    # ── Results dashboard ──
    st.markdown('<div class="section-title">📊 Prediction Results</div>', unsafe_allow_html=True)

    col1, col2, col3 = st.columns(3, gap="medium")

    def _sentiment_class(label: str) -> str:
        return "result-positive" if label == "Positive" else "result-negative"

    for col, (model_name, res) in zip([col1, col2, col3], results.items()):
        with col:
            css_class = _sentiment_class(res["label"])
            st.markdown(f"""
            <div class="result-card">
                <h3>{model_name}</h3>
                <div class="{css_class}">{res['emoji']} {res['label']}</div>
                <div class="result-conf">Confidence: {res['confidence']:.1%}</div>
                <div class="result-latency">⏱ Inference: {res['latency_ms']} ms</div>
            </div>
            """, unsafe_allow_html=True)

    # ── Confidence bar chart ──
    st.markdown('<div class="section-title">📈 Confidence Scores</div>', unsafe_allow_html=True)

    chart_data = pd.DataFrame({
        "Model":      list(results.keys()),
        "Confidence": [r["confidence"] for r in results.values()],
        "Sentiment":  [r["label"]      for r in results.values()],
    }).set_index("Model")

    st.bar_chart(chart_data["Confidence"], height=220)


# ── Comparison table (always visible) ─────────────────────────────────────────
st.divider()
st.markdown('<div class="section-title">📋 Model Performance Comparison (Benchmark on SST-2)</div>',
            unsafe_allow_html=True)
st.caption(
    "Metrics are pre-computed on the SST-2 validation set (~872 samples). "
    "Inference time is measured on a single CPU sample."
)

rows = []
for model_name, m in BENCHMARK_METRICS.items():
    rows.append({
        "Model":            model_name,
        "F1 Score":         f"{m['f1_score']:.4f}",
        "Accuracy":         f"{m['accuracy']:.4f}",
        "Inference Time":   m["inference_time"],
        "Notes":            m["notes"],
    })

df_metrics = pd.DataFrame(rows).set_index("Model")

st.dataframe(
    df_metrics,
    use_container_width=True,
    column_config={
        "F1 Score":       st.column_config.TextColumn("F1 Score ↑"),
        "Accuracy":       st.column_config.TextColumn("Accuracy ↑"),
        "Inference Time": st.column_config.TextColumn("Inference Time ↓"),
        "Notes":          st.column_config.TextColumn("Notes"),
    },
)

# ── Trade-off visual ──────────────────────────────────────────────────────────
with st.expander("📐 Accuracy vs Speed Trade-off", expanded=False):
    import plotly.graph_objects as go

    models      = list(BENCHMARK_METRICS.keys())
    accuracies  = [BENCHMARK_METRICS[m]["accuracy"]  for m in models]
    # Approximate numeric latency for the scatter (ms)
    latencies_num = [12, 180, 95]
    colors      = ["#89b4fa", "#a6e3a1", "#fab387"]

    fig = go.Figure()
    for i, m in enumerate(models):
        fig.add_trace(go.Scatter(
            x=[latencies_num[i]],
            y=[accuracies[i]],
            mode="markers+text",
            marker=dict(size=18, color=colors[i]),
            text=[m],
            textposition="top center",
            name=m,
        ))

    fig.update_layout(
        xaxis_title="Inference Time (ms, log scale)",
        yaxis_title="Accuracy",
        xaxis_type="log",
        yaxis=dict(range=[0.85, 0.95]),
        height=380,
        margin=dict(l=20, r=20, t=30, b=20),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#cdd6f4"),
    )
    st.plotly_chart(fig, use_container_width=True)


# ── Live validation section ───────────────────────────────────────────────────
st.divider()
with st.expander("🔬 Run Live Validation on Custom Dataset (optional)", expanded=False):
    st.info(
        "Upload a CSV with columns **text** and **label** (1=positive, 0=negative) "
        "to re-compute F1 / Accuracy live."
    )
    uploaded = st.file_uploader("Upload CSV", type=["csv"])
    selected_model = st.selectbox(
        "Select model to evaluate:",
        ["XGBoost (TF-IDF)", "BERT (base-uncased)", "DistilBERT"]
    )
    eval_btn = st.button("▶ Evaluate", key="eval_btn")

    if eval_btn and uploaded is not None:
        try:
            val_df = pd.read_csv(uploaded)
            if "text" not in val_df.columns or "label" not in val_df.columns:
                st.error("CSV must have columns 'text' and 'label'.")
            else:
                val_df = val_df.dropna(subset=["text", "label"])
                val_texts  = val_df["text"].astype(str).tolist()
                val_labels = val_df["label"].astype(int).tolist()

                from models import evaluate_on_dataset

                model_key_map = {
                    "XGBoost (TF-IDF)":  "xgboost",
                    "BERT (base-uncased)": "bert",
                    "DistilBERT":         "distilbert",
                }
                key = model_key_map[selected_model]

                progress_bar = st.progress(0, text="Evaluating…")
                results_eval = evaluate_on_dataset(
                    texts       = val_texts[:200],   # cap at 200 for speed
                    true_labels = val_labels[:200],
                    model_name  = key,
                    xgb_bundle          = (xgb_model, xgb_vectorizer),
                    bert_bundle         = (bert_tokenizer, bert_model),
                    distilbert_bundle   = (distilbert_tokenizer, distilbert_model),
                )
                progress_bar.progress(100, text="Done!")

                c1, c2, c3 = st.columns(3)
                c1.metric("F1 Score",        f"{results_eval['f1_score']:.4f}")
                c2.metric("Accuracy",         f"{results_eval['accuracy']:.4f}")
                c3.metric("Avg Latency (ms)", f"{results_eval['avg_latency_ms']}")

        except Exception as exc:
            st.error(f"Evaluation failed: {exc}")
    elif eval_btn and uploaded is None:
        st.warning("Please upload a CSV file first.")


# ── Footer ────────────────────────────────────────────────────────────────────
st.divider()
st.caption(
    "Built with [Streamlit](https://streamlit.io) · "
    "Models: XGBoost · [BERT](https://huggingface.co/textattack/bert-base-uncased-SST-2) · "
    "[DistilBERT](https://huggingface.co/distilbert-base-uncased-finetuned-sst-2-english)"
)
