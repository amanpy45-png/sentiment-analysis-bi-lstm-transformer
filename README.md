#  Sentiment Analysis — Twitter Brand Feedback

> Analyze text data to discern sentiment, aiding businesses in customer feedback assessment and decision making.

---

##  Overview

This project builds a deep learning model that classifies text into **Positive**, **Negative**, or **Neutral** sentiment. It is designed to help businesses monitor brand reputation, track customer satisfaction, and make data-driven decisions based on social media feedback.

---

##  Dataset

| Property | Details |
|---|---|
| Source | Twitter Entity Sentiment Dataset |
| Training Samples | ~61,692 |
| Validation Samples | 828 |
| Classes | Positive, Negative, Neutral |
| Format | CSV (id, entity, sentiment, text) |

**Class Distribution:**

| Class | Count |
|---|---|
| Negative | 22,542 |
| Positive | 20,832 |
| Neutral | 18,318 |

The dataset is relatively balanced, which helps improve model training stability and reduces class bias.

---

##  Model Architecture

```
Input Text
↓
Embedding Layer (vocab=20000, embedding_dim=128)
↓
Bidirectional LSTM (192 units, return_sequences=True)
↓
Global Max Pooling
↓
Dense Layer (64 units, ReLU)
↓
Dropout (0.5)
↓
Dense Layer (32 units, ReLU)
↓
Output Layer (3 units, Softmax)
```

| Parameter | Value |
|---|---|
| Vocabulary Size | 20,000 |
| Embedding Dim | 128 |
| Max Sequence Length | 50 |
| Batch Size | 64 |
| Optimizer | Adam |
| Loss | Sparse Categorical Crossentropy |

---

##  Setup & Installation

### Prerequisites
```
Python 3.8+
TensorFlow 2.x
scikit-learn
pandas
numpy
```

### Install Dependencies
```bash
pip install tensorflow scikit-learn pandas numpy
```

---

##  Usage

### Training the Model
Run all cells in the notebook sequentially:
```
1. Load and preprocess data
2. Tokenize and pad sequences
3. Build and compile model
4. Train with callbacks
5. Evaluate and save
```

### Loading Saved Model
```python
import tensorflow as tf
import pickle
import numpy as np
import re
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Load model and tokenizer
model = tf.keras.models.load_model("sentiment_model.keras")
with open("tokenizer.pkl", "rb") as f:
    tokenizer = pickle.load(f)

print("Model and tokenizer loaded successfully!")
```

### Predicting Sentiment
```python
def predict_sentiment(text):
    text = clean_text(text)
    seq = tokenizer.texts_to_sequences([text])
    padded = pad_sequences(seq, maxlen=50)
    pred = model.predict(padded)[0]
    label = np.argmax(pred)
    mapping = {0: "Negative", 1: "Neutral", 2: "Positive"}

    print(f"Text      : {text}")
    print(f"Negative  : {pred[0]*100:.2f}%")
    print(f"Neutral   : {pred[1]*100:.2f}%")
    print(f"Positive  : {pred[2]*100:.2f}%")
    print(f"Prediction: {mapping[label]}")
    print("-" * 40)

    return mapping[label]

predict_sentiment("Netflix keeps buffering so annoying")
```

---

##  Results

### Validation Performance (Bi-LSTM Model)

| Metric | Score |
|---|---|
| Accuracy | **98%** |
| Macro F1 Score | **0.98** |
| Macro Precision | **0.98** |
| Macro Recall | **0.98** |

### Classification Report

```
              precision    recall  f1-score   support

    Negative       0.98      0.98      0.98       266
     Neutral       0.98      0.98      0.98       285
    Positive       0.97      0.97      0.97       277

    accuracy                           0.98       828
   macro avg       0.98      0.98      0.98       828
weighted avg       0.98      0.98      0.98       828
```

### Sample Predictions

| Text | Expected | Predicted |
|---|---|---|
| Apple's new iPhone is absolutely amazing | Positive |  Positive |
| This game keeps crashing so frustrating | Negative |  Negative |
| The service was okay nothing special | Neutral |  Neutral |
| Amazon delivery was super fast loved it | Positive |  Positive |
| Worst update they have ever released | Negative |  Negative |

---

##  Data Preprocessing

- Lowercasing all text
- Removing URLs, mentions (@user), and hashtag symbols
- Keeping emotion markers (`!`, `?`)
- Removing special characters and digits
- Tokenization with vocabulary size of 20,000
- Padding/truncating to max length of 50 tokens

---

##  Transformer Model (Benchmark / Comparison)

> A pretrained Transformer-based sentiment analysis model (RoBERTa) is used in this project for benchmarking purposes only. It is not trained on the Twitter dataset used for the Bi-LSTM model. Instead, it is used to compare how a large-scale pretrained language model performs against a domain-trained deep learning model.

###  Purpose of Using Transformer

- To evaluate **general language understanding** vs **domain-specific learning**
- To compare prediction quality on unseen/out-of-domain text
- To analyze differences in handling:
  - Negation (e.g., *"not bad"*, *"not a good boy"*)
  - Contextual sentiment
  - Complex sentence structures

###  Key Insight

The Transformer model (RoBERTa) is pretrained on a large corpus of general text data and therefore has strong natural language understanding capabilities. It is used **only for inference** — not training — making it a zero-shot benchmark against the domain-trained Bi-LSTM.

###  Model Comparison

| Model | Role | Training Data | Strength |
|---|---|---|---|
| **Bi-LSTM** | Primary trained model | Twitter brand dataset | High accuracy on brand/product feedback |
| **RoBERTa** | Pretrained benchmark | Large general corpus | Strong on general/casual English text |

###  Transformer Setup & Usage

```bash
pip install transformers torch
```

```python
from transformers import pipeline

sentiment = pipeline(
    "text-classification",
    model="cardiffnlp/twitter-roberta-base-sentiment-latest"
)

def predict_general(text):
    result = sentiment(text)[0]
    label_map = {
        'positive': 'Positive',
        'negative': 'Negative',
        'neutral':  'Neutral'
    }
    label = label_map[result['label'].lower()]
    score = result['score'] * 100

    print(f"Text      : {text}")
    print(f"Prediction: {label} ({score:.2f}%)")
    print("-" * 40)

    return label

predict_general("The staff was so kind and helpful")
```

---

##  Saved Files

| File | Description |
|---|---|
| `sentiment_model.keras` | Trained BiLSTM model |
| `tokenizer.pkl` | Fitted tokenizer with vocabulary |

---

##  Limitations

- The model is trained on Twitter-based sentiment data and is optimized for short, opinionated text
- Performance may vary on long, complex, or formal English sentences
- Best suited for customer feedback and social media text containing product or brand-related opinions
- For general English sentiment, the RoBERTa benchmark model is recommended

---

##  Business Use Cases

- **Brand Monitoring** — Track how customers feel about your brand on Twitter
- **Product Feedback** — Identify negative feedback on product launches
- **Customer Support** — Prioritize negative sentiment tickets automatically
- **Competitor Analysis** — Compare sentiment across competing brands
- **Decision Making** — Use sentiment trends to guide marketing strategy

---

##  Project Structure

```
sentiment-analysis/
│
├── twitter_training.csv          # Training dataset
├── twitter_validation.csv        # Validation dataset
├── sentiment_analysis.ipynb      # Main notebook
├── sentiment_model.keras         # Saved Bi-LSTM model
├── tokenizer.pkl                 # Saved tokenizer
└── README.md                     # Project documentation
```
##  Future Improvements

This project can be further enhanced in the following ways:

-  Improve Bi-LSTM model using an **attention layer** for better context understanding
-  **Fine-tune the Transformer model** on the same Twitter dataset for a fairer comparison
-  Extend the system to **multi-language sentiment analysis** to support global customer feedback
-  Improve generalization using **larger and more diverse sentiment datasets**
-  Improve text preprocessing for better handling of **emojis and slang words**
---

##  Author

> NLP-based sentiment analysis project using Bi-LSTM deep learning model with RoBERTa Transformer as a benchmark comparison.

---

##  License
MIT License
