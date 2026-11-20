
import pickle
import time
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
)

from utils import clean_text

start = time.time()
data_path = Path("/app/Data/cleaned_news.csv")

print("Step 1: Loading dataset...", flush=True)
df = pd.read_csv(data_path)

# Split BEFORE fitting vectorizer
train_df, test_df = train_test_split(
    df,
    test_size=0.20,
    random_state=42,
    stratify=df["label"],
)

print("Step 2: Preprocessing text...", flush=True)
X_train_text = train_df["text"].fillna("").apply(clean_text)
X_test_text = test_df["text"].fillna("").apply(clean_text)

y_train = train_df["label"]
y_test = test_df["label"]

print("Step 3: TF-IDF vectorization...", flush=True)
vectorizer = TfidfVectorizer(
    max_features=25000,
    ngram_range=(1, 3),
)

X_train = vectorizer.fit_transform(X_train_text)
X_test = vectorizer.transform(X_test_text)

print("Step 4: Training Logistic Regression...", flush=True)
model = LogisticRegression(
    max_iter=1000,
    random_state=42,
)
model.fit(X_train, y_train)

print("Step 5: Testing model...", flush=True)
predictions = model.predict(X_test)

print("\n===== HOLDOUT TEST RESULTS =====")
print("Accuracy:", round(accuracy_score(y_test, predictions) * 100, 2), "%")
print("\nClassification Report:")
print(classification_report(
    y_test,
    predictions,
    target_names=["FAKE", "REAL"],
))
print("\nConfusion Matrix:")
print(confusion_matrix(y_test, predictions))

# Save separately; do not overwrite current model
with open("/app/model_new.pkl", "wb") as f:
    pickle.dump(model, f)

with open("/app/vectorizer_new.pkl", "wb") as f:
    pickle.dump(vectorizer, f)

print("\nNew model and vectorizer saved successfully!")
print("Total time:", round(time.time() - start, 2), "seconds")
