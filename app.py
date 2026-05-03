import streamlit as st
import numpy as np
import pickle
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
from transformers import pipeline

# =========================
# 🔹 PAGE CONFIG
# =========================
st.set_page_config(page_title="Sentiment Analyzer", layout="centered")

st.title("💬 Sentiment Analysis System")
st.write("Compare Bi-LSTM vs Transformer (RoBERTa)")

# =========================
# 🔹 LOAD BI-LSTM MODEL
# =========================
@st.cache_resource
def load_lstm_model():
    model = load_model("lstm_model.keras")
    return model

@st.cache_resource
def load_tokenizer():
    with open("tokenizer.pkl", "rb") as f:
        tokenizer = pickle.load(f)
    return tokenizer

model = load_lstm_model()
tokenizer = load_tokenizer()

max_len = 100  # ⚠️ must match training

# =========================
# 🔹 LSTM PREDICTION
# =========================
def predict_sentiment(text):
    seq = tokenizer.texts_to_sequences([text])
    padded = pad_sequences(seq, maxlen=max_len)

    pred = model.predict(padded, verbose=0)[0]
    label = np.argmax(pred)
    confidence = np.max(pred)

    mapping = {0: "Negative", 1: "Neutral", 2: "Positive"}

    return mapping[label], confidence

# =========================
# 🔹 TRANSFORMER MODEL
# =========================
@st.cache_resource
def load_transformer():
    return pipeline(
        "text-classification",
        model="cardiffnlp/twitter-roberta-base-sentiment-latest"
    )

sentiment_model = load_transformer()

def predict_general(text):
    result = sentiment_model(text)[0]

    label = result["label"].lower()

    label_map = {
        "positive": "Positive",
        "negative": "Negative",
        "neutral": "Neutral",
        "label_0": "Negative",
        "label_1": "Neutral",
        "label_2": "Positive"
    }

    return label_map[label], result["score"]

# =========================
# 🔹 UI INPUT
# =========================
text = st.text_area("Enter your text:")

if st.button("Analyze Sentiment"):

    if text.strip() == "":
        st.warning("Please enter some text.")
    else:

        # LSTM
        lstm_label, lstm_conf = predict_sentiment(text)

        # Transformer
        bert_label, bert_conf = predict_general(text)

        # =========================
        # 🔹 OUTPUT
        # =========================
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

        # =========================
        # 🔹 COMPARISON
        # =========================
        st.subheader("Comparison")

        if lstm_label == bert_label:
            st.success("✅ Both models agree")
        else:
            st.error("⚠️ Models disagree")

        # Optional insight
        if lstm_label != bert_label:
            st.info("Transformer is generally more accurate for complex sentences due to attention mechanism.")