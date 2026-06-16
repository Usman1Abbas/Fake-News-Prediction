<p align="center">
  <img src="assets/banner.svg" alt="Fake News Detection — Real or fake, classified with TF-IDF + logistic regression" width="100%">
</p>

<h1 align="center">Fake News Detection</h1>

<p align="center"><em>An interpretable NLP baseline that flags news articles as real or fake.</em></p>

<p align="center">
  <img alt="status" src="https://img.shields.io/badge/status-complete-1b9c85">
  <img alt="python" src="https://img.shields.io/badge/Python-3.x-0d1b2a">
  <img alt="scikit-learn" src="https://img.shields.io/badge/scikit--learn-LogReg-125a4d">
  <img alt="nlp" src="https://img.shields.io/badge/NLP-TF--IDF%20%C2%B7%20NLTK-e9c46a">
  <img alt="test accuracy" src="https://img.shields.io/badge/test%20accuracy-97.9%25-1b9c85">
</p>

**Fake News Detection** is a text-classification project that labels news articles as **real** or **fake**. It builds a clean, interpretable baseline with **TF-IDF** features and **logistic regression** in scikit-learn, trained on a **20,800-article** labeled dataset. Text is normalized with **NLTK** (stop-word removal + Porter stemming) before vectorization. The whole workflow lives in a single, runnable Jupyter notebook.

> A reminder that a well-tuned classical baseline reaches ~98% accuracy here — worth trying before reaching for heavier models.

---

## ✨ Features
- **Real/fake classification** of news articles from their author + headline text.
- **NLTK preprocessing** — lowercasing, non-letter stripping, English stop-word removal, and Porter stemming.
- **TF-IDF vectorization** to weight terms by how informative they are.
- **Logistic regression** for a fast, interpretable, and strong text-classification baseline.
- **Stratified 80/20 split** with a fixed random seed for reproducible evaluation.
- **Single self-contained notebook** — load, preprocess, vectorize, train, evaluate, predict.

## 🏗️ Pipeline
```text
train.csv (20,800 rows: id, title, author, text, label)
        │
        ▼
  fill missing values  →  content = author + " " + title
        │
        ▼
  NLTK cleaning  →  regex strip · lowercase · stop-word removal · Porter stemming
        │
        ▼
  TfidfVectorizer  →  sparse TF-IDF feature matrix
        │
        ▼
  train_test_split (test_size=0.2, stratify=label, random_state=2)
        │
        ▼
  LogisticRegression.fit  →  predict  →  accuracy_score
```
> Note: `label = 1` means **fake**, `label = 0` means **real**.

## 🚀 Run it
```bash
# 1. Clone
git clone https://github.com/Usman1Abbas/Fake-News-Prediction.git
cd Fake-News-Prediction

# 2. Install dependencies
pip install numpy pandas scikit-learn nltk jupyter

# 3. Launch the notebook
jupyter notebook Fake_News_Prediction.ipynb
```
The notebook downloads the NLTK `stopwords` corpus on first run. Point the `pd.read_csv(...)` path to your local `train.csv` (the Kaggle "Fake News" dataset with `id, title, author, text, label` columns), then run the cells top-to-bottom.

## 📊 Results
Logistic regression on the TF-IDF features:

| Split    | Accuracy |
|----------|----------|
| Training | **98.66%** |
| Test     | **97.91%** |

A worked example at the end of the notebook takes a held-out vector, predicts its class, and prints **"The news is Real"** / **"The news is Fake"**.

## 🔧 Stack
**Python** · **scikit-learn** (TfidfVectorizer, LogisticRegression) · **NLTK** (stopwords, PorterStemmer) · **pandas** · **NumPy** · **Jupyter**
