import streamlit as st
import numpy as np
import pickle
import os
import gdown
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
from transformers import pipeline

# =========================
# PAGE CONFIG
# =========================
st.set_page_config(page_title="Sentiment Analyzer", layout="centered")
st.title("💬 Sentiment Analysis System")
st.write("Compare Bi-LSTM vs Transformer")

# =========================
# LOAD LSTM MODEL (Google Drive)
# =========================
@st.cache_resource
def load_lstm_model():
    output = "lstm_model.keras"
    if not os.path.exists(output):
        file_id = "1fuc8GRgjm95J-bqBi_F6jq6aKmzJeqlF"
        gdown.download(
            id=file_id,
            output=output,
            quiet=False
        )
    model = load_model(output)
    return model

# =========================
# LOAD TOKENIZER
# =========================
@st.cache_resource
def load_tokenizer():
    with open("tokenizer.pkl", "rb") as f:
        tokenizer = pickle.load(f)
    return tokenizer

model = load_lstm_model()
tokenizer = load_tokenizer()
max_len = 100

# =========================
# LSTM PREDICTION
# =========================
def predict_sentiment(text):
    seq = tokenizer.texts_to_sequences([text])
    padded = pad_sequences(seq, maxlen=max_len)
    pred = model.predict(padded, verbose=0)
    if pred.shape[-1] == 1:
        label = "Positive" if pred[0][0] > 0.5 else "Negative"
        confidence = float(pred[0][0])
    else:
        idx = np.argmax(pred)
        confidence = float(np.max(pred))
        mapping = {0: "Negative", 1: "Neutral", 2: "Positive"}
        label = mapping[idx]
    return label, confidence

# =========================
# TRANSFORMER MODEL
# =========================
@st.cache_resource
def load_transformer():
    return pipeline(
        "sentiment-analysis",
        model="distilbert-base-uncased-finetuned-sst-2-english",
        framework="pt"
    )

sentiment_model = load_transformer()

def predict_general(text):
    result = sentiment_model(text, truncation=True, max_length=512)[0]
    label = result["label"]
    confidence = result["score"]
    return label, confidence

# =========================
# UI INPUT
# =========================
text = st.text_area("Enter your text:")

if st.button("Analyze Sentiment"):
    if text.strip() == "":
        st.warning("Please enter some text.")
    else:
        lstm_label, lstm_conf = predict_sentiment(text)
        bert_label, bert_conf = predict_general(text)

        st.subheader("Results")
        col1, col2 = st.columns(2)

        with col1:
            st.markdown("### 🧠 Bi-LSTM")
            st.write(f"**Prediction:** {lstm_label}")
            st.write(f"**Confidence:** {lstm_conf:.2f}")

        with col2:
            st.markdown("### 🤖 Transformer")
            st.write(f"**Prediction:** {bert_label}")
            st.write(f"**Confidence:** {bert_conf:.2f}")

        st.subheader("Comparison")
        if lstm_label == bert_label:
            st.success("✅ Both models agree")
        else:
            st.error("⚠️ Models disagree")
            st.info("Transformer usually performs better on complex sentences.")
