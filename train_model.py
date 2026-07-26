"""
Train and persist a sentiment analysis model.

Improvements over the original notebook:
- Larger, balanced dataset (90 rows instead of 6)
- Uses cross-validation (not just a single train/test split) since the
  dataset is still small enough that one split can be misleading
- Unigrams + bigrams in TF-IDF (helps catch phrases like "not worth")
- Saves the fitted vectorizer + best model to disk with joblib so they
  can be reused by an API instead of retraining every time
"""

import re
import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

DATA_PATH = "data/sample_reviews.csv"
MODEL_DIR = "model"


def clean_text(text: str) -> str:
    text = text.lower()
    text = re.sub(r"[^a-z\s]", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def main():
    df = pd.read_csv(DATA_PATH)
    df["clean_review"] = df["review"].apply(clean_text)

    X = df["clean_review"]
    y = df["sentiment"]

    vectorizer = TfidfVectorizer(stop_words="english", ngram_range=(1, 2), min_df=1)
    X_vec = vectorizer.fit_transform(X)

    candidates = {
        "Logistic Regression": LogisticRegression(max_iter=1000),
        "Multinomial Naive Bayes": MultinomialNB(),
    }

    print("=== Cross-validated accuracy (5-fold) ===")
    cv_scores = {}
    for name, model in candidates.items():
        scores = cross_val_score(model, X_vec, y, cv=5)
        cv_scores[name] = scores.mean()
        print(f"{name}: {scores.mean():.3f} (+/- {scores.std():.3f})")

    # Also report a held-out split for a human-readable report
    X_train, X_test, y_train, y_test = train_test_split(
        X_vec, y, test_size=0.2, random_state=42, stratify=y
    )

    print("\n=== Held-out test set report ===")
    for name, model in candidates.items():
        model.fit(X_train, y_train)
        preds = model.predict(X_test)
        print(f"\n--- {name} ---")
        print("Accuracy:", accuracy_score(y_test, preds))
        print(confusion_matrix(y_test, preds))
        print(classification_report(y_test, preds, zero_division=0))

    # Pick the best model by cross-validated score, then refit on ALL data
    # before saving -- more training data = a better final model, and we've
    # already used CV (not this final fit) to judge which model is stronger.
    best_name = max(cv_scores, key=cv_scores.get)
    best_model = candidates[best_name]
    best_model.fit(X_vec, y)
    print(f"\nSelected best model: {best_name} (cv accuracy {cv_scores[best_name]:.3f})")

    joblib.dump(best_model, f"{MODEL_DIR}/model.pkl")
    joblib.dump(vectorizer, f"{MODEL_DIR}/vectorizer.pkl")
    with open(f"{MODEL_DIR}/model_info.txt", "w") as f:
        f.write(f"{best_name}\ncv_accuracy={cv_scores[best_name]:.3f}\n")

    print(f"\nSaved model + vectorizer to {MODEL_DIR}/")


if __name__ == "__main__":
    main()
