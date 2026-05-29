# 📰 Fake News Detection

A text-classification model that flags news articles as **real** or **fake** using TF-IDF features and logistic regression.

## Pipeline
1. **Preprocessing** — clean and normalize article text (lowercasing, punctuation/stop-word removal).
2. **Vectorization** — TF-IDF to weight terms by how informative they are.
3. **Model** — logistic regression for an interpretable, strong text-classification baseline.
4. **Evaluation** — accuracy, precision/recall, and confusion matrix on a held-out test set.

## What it demonstrates
A clean, interpretable **NLP classification baseline** — and why logistic regression + TF-IDF is still a smart first move before reaching for heavier models.

## Stack
Python · scikit-learn · pandas · NumPy · Jupyter

## Run
Open the notebook and execute top-to-bottom: load → preprocess → vectorize → train → evaluate.
