"""
train.py — Model Training Script for Spam Detector AI
=======================================================
Loads the SMS Spam Collection dataset (spam.csv), applies text preprocessing,
trains a Scikit-learn Pipeline (TF-IDF + Multinomial Naive Bayes), evaluates
accuracy on a held-out test set, and serialises the model to ../model.pkl.

Usage:
    cd ml_training
    python train.py

Output:
    ../model.pkl  — Serialised pipeline ready for use by app.py

Sample test messages:
    Spam : "WINNER!! You have been selected to receive a £1000 prize! Call 09061701461."
    Ham  : "Hey, are we still meeting for lunch at noon today?"
"""

import pandas as pd
import numpy as np
import os
import string
import joblib

# ── NLTK for stopword removal ──
import nltk
from nltk.corpus import stopwords

# ── Scikit-learn components ──
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, accuracy_score

# ── Download NLTK stopwords corpus (first run only) ──
nltk.download('stopwords')
nltk.download('punkt')


def preprocess_text(text: str) -> str:
    """
    Clean and normalise a raw text message.

    Steps applied (in order):
      1. Convert to lowercase          — removes case sensitivity
      2. Strip punctuation             — reduces noise from symbols
      3. Tokenise and remove stopwords — keeps only meaningful content words

    Parameters
    ----------
    text : str  Raw input message.

    Returns
    -------
    str  Cleaned, space-joined token string.
    """
    # Step 1: Lowercase the entire message
    text = text.lower()

    # Step 2: Remove all punctuation characters
    text = "".join([ch for ch in text if ch not in string.punctuation])

    # Step 3: Remove NLTK English stopwords (e.g. "the", "is", "and")
    stop_words = set(stopwords.words('english'))
    tokens     = text.split()
    tokens     = [word for word in tokens if word not in stop_words]

    return " ".join(tokens)


def train_model():
    """
    Full training pipeline:
      1. Load dataset from spam.csv (latin-1 encoding, no header row).
      2. Drop rows with missing values.
      3. Preprocess every message using preprocess_text().
      4. Split into 80% train / 20% test sets.
      5. Build a Scikit-learn Pipeline:
             TfidfVectorizer — converts cleaned text to TF-IDF feature matrix
             MultinomialNB  — Naive Bayes classifier on word-count features
      6. Evaluate on the test set and print a classification report.
      7. Serialise the trained pipeline to ../model.pkl via joblib.
    """

    # ── File paths ──
    DATA_FILE  = "spam.csv"
    MODEL_FILE = "../model.pkl"   # Saved one level up (project root)

    # ── Step 1: Verify dataset exists ──
    if not os.path.exists(DATA_FILE):
        print(f"[ERROR] Dataset not found: {DATA_FILE}")
        return

    # ── Step 2: Load CSV ──
    # The file uses latin-1 encoding and has no header row.
    # Column 0 = label (ham/spam), Column 1 = message text.
    try:
        df = pd.read_csv(DATA_FILE, encoding='latin-1', header=None)
        df = df[[0, 1]]
        df.columns = ['label', 'message']
    except Exception as e:
        print(f"[ERROR] Failed to read CSV: {e}")
        return

    # ── Step 3: Drop rows with missing values ──
    df.dropna(inplace=True)

    # ── Normalise labels: keep only 'ham' and 'spam' rows ──
    df = df[df['label'].isin(['ham', 'spam'])]

    print(f"Dataset loaded.  Total records : {len(df)}")
    print(df['label'].value_counts(), "\n")

    # ── Step 4: Preprocess all messages ──
    print("Preprocessing messages…")
    df['clean'] = df['message'].apply(preprocess_text)

    # ── Step 5: Define features and labels ──
    X = df['clean']   # Preprocessed message text
    y = df['label']   # Target label: 'ham' or 'spam'

    # Split into training and test sets (80 / 20 ratio, fixed random seed)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # ── Step 6: Build the Scikit-learn Pipeline ──
    # TfidfVectorizer : Converts text to a TF-IDF weighted sparse matrix.
    #   - Term Frequency (TF)  : frequency of a word in the document.
    #   - Inverse Doc Freq (IDF): down-weights words common across all documents.
    # MultinomialNB   : Naive Bayes for discrete/count features; ideal for TF-IDF.
    pipeline = Pipeline([
        ('tfidf',      TfidfVectorizer()),
        ('classifier', MultinomialNB())
    ])

    # ── Step 7: Train the pipeline ──
    print("Training the model…")
    pipeline.fit(X_train, y_train)

    # ── Step 8: Evaluate on the test set ──
    preds = pipeline.predict(X_test)
    print("\nModel Performance")
    print("─" * 40)
    print(f"Accuracy : {accuracy_score(y_test, preds):.4f}")
    print(classification_report(y_test, preds))

    # ── Step 9: Save the trained pipeline to disk ──
    print(f"Saving model to {MODEL_FILE}…")
    joblib.dump(pipeline, MODEL_FILE)
    print("Model saved successfully!")


# ── Entry point ──
if __name__ == "__main__":
    train_model()
