
import pickle
from pathlib import Path

import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    precision_score,
    recall_score,
    f1_score,
)

from utils import clean_text

BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = Path("/app/Data/cleaned_news.csv")

print("Loading dataset...")
df = pd.read_csv(DATA_PATH)

print("Loading trained model...")
with (BASE_DIR / "model.pkl").open("rb") as f:
    model = pickle.load(f)

with (BASE_DIR / "vectorizer.pkl").open("rb") as f:
    vectorizer = pickle.load(f)

print("Preprocessing articles...")
texts = df["text"].fillna("").astype(str).apply(clean_text)

print("Vectorizing articles...")
X = vectorizer.transform(texts)

y_true = df["label"].astype(int)

print("Running predictions...")
y_pred = model.predict(X)

print("\n===== MODEL EVALUATION =====")
print(f"Total Articles: {len(df)}")
print(f"Accuracy:  {accuracy_score(y_true, y_pred):.4f}")
print(f"Precision: {precision_score(y_true, y_pred):.4f}")
print(f"Recall:    {recall_score(y_true, y_pred):.4f}")
print(f"F1 Score:  {f1_score(y_true, y_pred):.4f}")

print("\nClassification Report:")
print(
    classification_report(
        y_true,
        y_pred,
        labels=[0, 1],
        target_names=["FAKE", "REAL"],
        zero_division=0,
    )
)

print("\nConfusion Matrix:")
print(confusion_matrix(y_true, y_pred, labels=[0, 1]))

print("\nNOTE: Diagnostic evaluation only.")
print("Training-data overlap has not been ruled out.")
