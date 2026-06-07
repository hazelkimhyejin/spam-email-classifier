# Spam Email Classifier (Enron dataset)

A from-scratch spam filter that mirrors how real email providers work: turn raw
email text into numbers, train a model to recognise spam, and measure how well
it does on emails it has never seen.

## Quick start

```bash
pip install -r requirements.txt
python spam_classifier.py
```

The script auto-downloads the Enron spam/ham dataset (~33,000 real emails) the
first time you run it.

## Results

| Model               | Accuracy | Precision | Recall | F1     |
|---------------------|----------|-----------|--------|--------|
| Logistic Regression | 0.9905   | 0.9858    | 0.9956 | 0.9907 |
| Random Forest       | 0.9862   | 0.9837    | 0.9892 | 0.9865 |

Logistic Regression wins. The sections below explain why that is the *expected*
result, not luck.

## How the pipeline works

**1. Load the data.** 33,716 emails, each labelled `spam` or `ham` (not spam).
The set is nearly balanced (~51% spam), so plain accuracy is meaningful here.

**2. Preprocess the text.** We merge each email's subject and body, lowercase
it, normalise URLs and numbers, and strip punctuation. The cleaning is kept
light on purpose — over-cleaning deletes signal (a flood of `$$$` is itself a
spam tell).

**3. TF-IDF vectorization.** Models need numbers, not words. TF-IDF =
*Term Frequency × Inverse Document Frequency*:
- **Term Frequency**: how often a word appears in *this* email.
- **Inverse Document Frequency**: down-weights words common to *every* email
  ("the", "and") and up-weights distinctive ones ("invoice", "viagra").

Each email becomes a long, mostly-zero (sparse) vector across ~20,000
word/word-pair features. We also use bigrams (`ngram_range=(1,2)`) so phrases
like "click here" count as their own feature.

**4. Train/test split done right.** We split *before* fitting TF-IDF, so the
vectorizer learns its vocabulary only from training data. Letting it peek at
the test set is **data leakage** and inflates your scores dishonestly.

**5. Train two models and compare** on the 20% held-out test set, looking at
accuracy plus precision, recall, and F1 (defined below).

## Why Logistic Regression beats Random Forest here

This is the core lesson of the project. It comes down to the *shape* of the
data, not one model being "smarter."

- **The data is high-dimensional and sparse.** 20,000 features, almost all zero
  for any given email. Logistic regression is essentially a weighted vote: it
  assigns one weight per word and sums them. That linear structure is a near
  perfect match for "spam is roughly: presence of these words good, those words
  bad." Spam detection is *mostly linearly separable* in word space.

- **Random forests split one feature at a time.** Each tree asks yes/no
  questions like "does this email contain 'viagra'?" To use evidence from many
  weak signals at once, it needs deep trees and many of them. On sparse text
  where the signal is spread thinly across thousands of rare words, that's an
  awkward fit — the trees can't combine many faint clues as cleanly as a linear
  sum does.

- **Random forests shine where logistic regression struggles**: dense tabular
  data with non-linear interactions and thresholds (e.g. "approve loan if
  income > X *and* age < Y"). Text-with-TF-IDF is the opposite regime.

- **Bonus — interpretability.** Logistic regression hands you its reasoning.
  Its largest positive weights are the most spam-like words; the script prints
  them. From this run:
  - **Spammy:** free, click, viagra, money, online, meds, remove, account
  - **Hammy:** enron, vince, meeting, houston, thanks, attached, gas

  Those ham words are exactly what you'd expect from emails between Enron
  employees — a good sanity check that the model learned something real.

The general principle: **match the model to the data's structure.** Linear
models for wide, sparse, roughly-linear problems; tree ensembles for dense,
interaction-heavy ones. Always try the simple model first — here it both wins
*and* runs in a fraction of the time.

## Metric cheat-sheet

For spam, the "positive" class is spam.

- **Accuracy** — fraction of all emails classified correctly. Fine here because
  classes are balanced; misleading when they aren't.
- **Precision** — of emails we *flagged* as spam, how many really were. Low
  precision = real mail landing in the spam folder (false alarms). Costly.
- **Recall** — of emails that *were* spam, how many we caught. Low recall =
  spam reaching the inbox.
- **F1** — the harmonic mean of precision and recall; a single balanced score.

In a real product you'd often tune the decision threshold to favour
**precision** — wrongly trashing a real email annoys users far more than
letting one spam through.

## Ideas to extend it

- Tune the spam threshold and plot the precision/recall trade-off.
- Add a third model (Naive Bayes — the classic spam-filter algorithm) and see
  how the famous baseline compares.
- Use `GridSearchCV` to tune `C` (logistic) and `max_features` (TF-IDF).
- Save the trained model with `joblib` and write a `predict(email_text)`
  function so you can score new emails on demand.
- Look at the misclassified emails — they're usually the most interesting ones.
