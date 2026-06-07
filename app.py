"""
Spam Email Classifier — Streamlit app
=====================================

Loads the trained pipeline (spam_model.joblib) and lets a user paste an email
to see whether it's spam, with a confidence score and the words that pushed the
decision.

Run locally:  streamlit run app.py
"""

import joblib
import numpy as np
import streamlit as st

from preprocessing import clean_text

MODEL_PATH = "spam_model.joblib"

st.set_page_config(page_title="Spam Email Classifier", page_icon="📧")


@st.cache_resource
def load_model():
    """Load the trained pipeline once and cache it across reruns."""
    return joblib.load(MODEL_PATH)


def top_contributing_words(pipeline, cleaned, n=8):
    """Return words in this email that pushed hardest toward the prediction."""
    tfidf = pipeline.named_steps["tfidf"]
    clf = pipeline.named_steps["clf"]
    vec = tfidf.transform([cleaned])
    coefs = clf.coef_[0]
    # contribution of each present word = its tfidf value * its weight
    contributions = vec.multiply(coefs).toarray()[0]
    names = np.array(tfidf.get_feature_names_out())
    spam_idx = np.argsort(contributions)[-n:][::-1]
    ham_idx = np.argsort(contributions)[:n]
    spam_words = [(names[i], contributions[i]) for i in spam_idx if contributions[i] > 0]
    ham_words = [(names[i], contributions[i]) for i in ham_idx if contributions[i] < 0]
    return spam_words, ham_words


model = load_model()

st.title("📧 Spam Email Classifier")
st.write(
    "Paste an email below and the model will judge whether it's **spam** or "
    "**ham** (legitimate). Trained on ~33,000 real emails from the Enron "
    "dataset using TF-IDF + logistic regression."
)

SAMPLE = (
    "Subject: CONGRATULATIONS!!! You have WON $1,000,000!!!\n\n"
    "Dear winner, click here now to claim your free prize money. "
    "Limited time offer, act fast! Send your bank account details to "
    "verify your identity. 100% guaranteed, no risk!"
)

email_text = st.text_area("Email text", value=SAMPLE, height=220)

if st.button("Classify", type="primary"):
    if not email_text.strip():
        st.warning("Please paste some email text first.")
    else:
        cleaned = clean_text(email_text)
        proba = model.predict_proba([cleaned])[0]
        spam_prob = proba[1]
        is_spam = spam_prob >= 0.5

        if is_spam:
            st.error(f"🚫 **SPAM** — {spam_prob * 100:.1f}% confidence")
        else:
            st.success(f"✅ **HAM (legitimate)** — {(1 - spam_prob) * 100:.1f}% confidence")

        st.progress(float(spam_prob))
        st.caption(f"Spam probability: {spam_prob * 100:.1f}%")

        spam_words, ham_words = top_contributing_words(model, cleaned)
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("**Pushed toward spam**")
            if spam_words:
                for w, _ in spam_words:
                    st.markdown(f"- {w}")
            else:
                st.caption("—")
        with col2:
            st.markdown("**Pushed toward ham**")
            if ham_words:
                for w, _ in ham_words:
                    st.markdown(f"- {w}")
            else:
                st.caption("—")

st.divider()
st.caption(
    "Model: TF-IDF (20k features, unigrams + bigrams) → logistic regression. "
    "~99% test accuracy. See README for the full model comparison."
)
