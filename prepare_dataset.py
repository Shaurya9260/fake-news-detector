
from pathlib import Path
import pandas as pd

DATA_DIR = Path("Data")

fake = pd.read_csv(DATA_DIR / "Fake.csv")
real = pd.read_csv(DATA_DIR / "True.csv")

fake["label"] = 0
real["label"] = 1

df = pd.concat([fake, real], ignore_index=True)

print("Original rows:", len(df))

# Remove empty article bodies
df = df[df["text"].fillna("").str.strip().ne("")].copy()
print("After removing empty bodies:", len(df))

# Normalize text for duplicate detection
df["normalized_text"] = (
    df["text"]
    .astype(str)
    .str.strip()
    .str.lower()
    .str.replace(r"\s+", " ", regex=True)
)

# Remove articles whose identical text has conflicting labels
conflicting = (
    df.groupby("normalized_text")["label"]
    .nunique()
)
conflicting_texts = conflicting[conflicting > 1].index

df = df[
    ~df["normalized_text"].isin(conflicting_texts)
].copy()

print("After removing conflicting labels:", len(df))

# Keep one copy of each article
df = df.drop_duplicates(subset=["normalized_text"])

df = df.drop(columns=["normalized_text"])
df = df.sample(frac=1, random_state=42).reset_index(drop=True)

output = DATA_DIR / "cleaned_news.csv"
df.to_csv(output, index=False)

print("Final cleaned rows:", len(df))
print("Class distribution:")
print(df["label"].value_counts().sort_index())
print("Saved to:", output)
