"""
Spam Email Classifier — Enron dataset
=====================================

Pipeline:
    1. Load the Enron spam/ham dataset (auto-downloads if missing).
    2. Preprocess the text (combine subject + body, clean, vectorize with TF-IDF).
    3. Train two models: Logistic Regression and Random Forest.
    4. Compare them on held-out test data.
    5. Inspect what the model learned (most "spammy" words).

Run:  python spam_classifier.py
"""

import os
import re
import sys
import zipfile
import urllib.request

import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
)

# --------------------------------------------------------------------------
# 1. LOAD DATA
# --------------------------------------------------------------------------
DATA_CSV = "enron_spam_data.csv"
DATA_ZIP = "enron_spam_data.zip"
DATA_URL = (
    "https://raw.githubusercontent.com/"
    "MWiechmann/enron_spam_data/master/enron_spam_data.zip"
)


def load_data() -> pd.DataFrame:
    """Return the Enron dataset as a DataFrame, downloading it if needed."""
    if not os.path.exists(DATA_CSV):
        if not os.path.exists(DATA_ZIP):
            print(f"Downloading dataset from {DATA_URL} ...")
            urllib.request.urlretrieve(DATA_URL, DATA_ZIP)
        print("Extracting ...")
        with zipfile.ZipFile(DATA_ZIP) as z:
            z.extractall(".")

    df = pd.read_csv(DATA_CSV)
    print(f"Loaded {len(df):,} emails.")
    return df


# --------------------------------------------------------------------------
# 2. PREPROCESS
# --------------------------------------------------------------------------
def clean_text(text: str) -> str:
    """Lowercase, strip URLs/numbers/punctuation, collapse whitespace.

    We keep this deliberately simple. TF-IDF + the models do the heavy
    lifting; aggressive cleaning can actually *hurt* by deleting useful
    signal (e.g. lots of '$$$' is itself a spam giveaway).
    """
    text = text.lower()
    text = re.sub(r"http\S+|www\.\S+", " url ", text)   # normalise links
    text = re.sub(r"\d+", " num ", text)                 # normalise numbers
    text = re.sub(r"[^a-z\s]", " ", text)                # drop punctuation
    text = re.sub(r"\s+", " ", text).strip()             # collapse spaces
    return text


def preprocess(df: pd.DataFrame):
    """Combine subject + body into one cleaned text column and make labels."""
    df = df.copy()
    # Missing subjects/bodies become empty strings so we don't lose the row.
    df["Subject"] = df["Subject"].fillna("")
    df["Message"] = df["Message"].fillna("")

    # Subject lines carry a lot of spam signal, so include them.
    df["text"] = (df["Subject"] + " " + df["Message"]).apply(clean_text)

    # Map labels to integers: spam = 1 (the "positive" class we care about).
    df["label"] = (df["Spam/Ham"] == "spam").astype(int)

    # Drop rows that are empty after cleaning.
    df = df[df["text"].str.len() > 0]
    return df["text"], df["label"]


# --------------------------------------------------------------------------
# 3. + 4. TRAIN AND EVALUATE
# --------------------------------------------------------------------------
def evaluate(name, model, X_test, y_test):
    """Print a metrics block for one trained model and return its scores."""
    preds = model.predict(X_test)
    acc = accuracy_score(y_test, preds)
    prec = precision_score(y_test, preds)
    rec = recall_score(y_test, preds)
    f1 = f1_score(y_test, preds)
    tn, fp, fn, tp = confusion_matrix(y_test, preds).ravel()

    print(f"\n{'=' * 55}\n{name}\n{'=' * 55}")
    print(f"Accuracy : {acc:.4f}")
    print(f"Precision: {prec:.4f}   (of emails flagged spam, % truly spam)")
    print(f"Recall   : {rec:.4f}   (of real spam, % we caught)")
    print(f"F1 score : {f1:.4f}")
    print(f"\nConfusion matrix:")
    print(f"             pred ham   pred spam")
    print(f"  true ham   {tn:>8}   {fp:>9}   <- {fp} false alarms")
    print(f"  true spam  {fn:>8}   {tp:>9}   <- {fn} spam slipped through")
    return {"name": name, "accuracy": acc, "precision": prec,
            "recall": rec, "f1": f1}


def show_top_spam_words(model, vectorizer, n=15):
    """For Logistic Regression: print the words that push hardest toward spam.

    Each TF-IDF word becomes a feature with a learned weight (coefficient).
    Large positive weight => seeing this word increases the spam probability.
    This is why logistic regression is so interpretable.
    """
    feature_names = np.array(vectorizer.get_feature_names_out())
    coefs = model.coef_[0]
    top_spam = np.argsort(coefs)[-n:][::-1]
    top_ham = np.argsort(coefs)[:n]

    print(f"\n{'=' * 55}\nWhat the model learned (Logistic Regression)\n{'=' * 55}")
    print("Most spam-indicative words:")
    print("  " + ", ".join(feature_names[top_spam]))
    print("Most ham-indicative words:")
    print("  " + ", ".join(feature_names[top_ham]))


def main():
    df = load_data()
    X_text, y = preprocess(df)
    print(f"After cleaning: {len(X_text):,} usable emails "
          f"({y.mean() * 100:.1f}% spam)")

    # --- Train/test split BEFORE vectorizing, to avoid data leakage ---
    # The vectorizer must learn its vocabulary only from training data.
    X_train_text, X_test_text, y_train, y_test = train_test_split(
        X_text, y, test_size=0.2, random_state=42, stratify=y
    )

    # --- TF-IDF vectorization ---
    # TF-IDF = Term Frequency x Inverse Document Frequency.
    #   - Term Frequency: how often a word appears in THIS email.
    #   - Inverse Doc Frequency: down-weights words common across ALL emails
    #     ("the", "and") and up-weights distinctive ones ("viagra", "invoice").
    # The result is a sparse matrix: ~26k emails x up to 20k word-features.
    vectorizer = TfidfVectorizer(
        stop_words="english",   # drop common filler words
        max_features=20000,     # keep the 20k most informative words
        ngram_range=(1, 2),     # use single words AND adjacent word pairs
        min_df=2,               # ignore words appearing in <2 emails (noise)
    )
    X_train = vectorizer.fit_transform(X_train_text)   # learn vocab + transform
    X_test = vectorizer.transform(X_test_text)         # transform only
    print(f"TF-IDF matrix: {X_train.shape[0]:,} emails x "
          f"{X_train.shape[1]:,} features")

    results = []

    # --- Model A: Logistic Regression ---
    # A linear model: it learns one weight per word and adds them up.
    # Fast, and a strong baseline for high-dimensional sparse text.
    print("\nTraining Logistic Regression ...")
    logreg = LogisticRegression(max_iter=1000, C=1.0)
    logreg.fit(X_train, y_train)
    results.append(evaluate("LOGISTIC REGRESSION", logreg, X_test, y_test))

    # --- Model B: Random Forest ---
    # An ensemble of decision trees. Each tree splits on a handful of words;
    # the forest averages their votes. Great for tabular data with feature
    # interactions, but works with fewer features per split on sparse text.
    print("\nTraining Random Forest (this is the slow one) ...")
    rf = RandomForestClassifier(
        n_estimators=200, max_depth=None, n_jobs=-1, random_state=42
    )
    rf.fit(X_train, y_train)
    results.append(evaluate("RANDOM FOREST", rf, X_test, y_test))

    # --- Interpretability bonus ---
    show_top_spam_words(logreg, vectorizer)

    # --- 5. SIDE-BY-SIDE COMPARISON ---
    print(f"\n{'=' * 55}\nSUMMARY\n{'=' * 55}")
    summary = pd.DataFrame(results).set_index("name").round(4)
    print(summary.to_string())

    winner = summary["f1"].idxmax()
    print(f"\nBest model by F1 score: {winner}")


if __name__ == "__main__":
    main()
