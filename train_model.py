"""
Train the deployed model and save it to spam_model.joblib.

Run this once (locally) to produce the model file the Streamlit app loads:
    python train_model.py

We deploy Logistic Regression because it won the comparison in
spam_classifier.py. The whole thing — TF-IDF + classifier — is saved as a
single sklearn Pipeline, so the app can feed in raw (cleaned) text.
"""

import os
import zipfile
import urllib.request

import joblib
import pandas as pd

from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

from preprocessing import clean_text

DATA_CSV = "enron_spam_data.csv"
DATA_ZIP = "enron_spam_data.zip"
DATA_URL = (
    "https://raw.githubusercontent.com/"
    "MWiechmann/enron_spam_data/master/enron_spam_data.zip"
)
MODEL_PATH = "spam_model.joblib"


def load_data() -> pd.DataFrame:
    if not os.path.exists(DATA_CSV):
        if not os.path.exists(DATA_ZIP):
            print(f"Downloading dataset ...")
            urllib.request.urlretrieve(DATA_URL, DATA_ZIP)
        with zipfile.ZipFile(DATA_ZIP) as z:
            z.extractall(".")
    return pd.read_csv(DATA_CSV)


def main():
    df = load_data()
    df["Subject"] = df["Subject"].fillna("")
    df["Message"] = df["Message"].fillna("")
    X = (df["Subject"] + " " + df["Message"]).apply(clean_text)
    y = (df["Spam/Ham"] == "spam").astype(int)
    keep = X.str.len() > 0
    X, y = X[keep], y[keep]
    print(f"Training on {len(X):,} emails ...")

    pipeline = Pipeline([
        ("tfidf", TfidfVectorizer(
            stop_words="english", max_features=20000,
            ngram_range=(1, 2), min_df=2,
        )),
        ("clf", LogisticRegression(max_iter=1000, C=1.0)),
    ])
    pipeline.fit(X, y)

    joblib.dump(pipeline, MODEL_PATH)
    size_mb = os.path.getsize(MODEL_PATH) / 1e6
    print(f"Saved {MODEL_PATH} ({size_mb:.1f} MB)")


if __name__ == "__main__":
    main()
