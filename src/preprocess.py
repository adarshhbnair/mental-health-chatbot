import os
import pandas as pd
from sklearn.model_selection import train_test_split
import re
import ast

# Ensure output directory exists
os.makedirs("data/processed", exist_ok=True)

# Load raw GoEmotions dataset
df = pd.read_csv("data/raw/goemotions.csv")[["text", "labels"]]
# Drop rows with missing text or labels
df = df.dropna(subset=["text", "labels"])

# Clean text function
def clean_text(t):
    t = t.lower()
    t = re.sub(r"http\S+|www\S+", "", t)
    t = re.sub(r"[^a-z0-9\s,.!?']", " ", t)
    t = re.sub(r"\s+", " ", t).strip()
    return t

df["text"] = df["text"].apply(clean_text)

def parse_labels(x):
    if isinstance(x, list):
        return x

    x = str(x).strip()

    # Case: "[3,5,7]"
    if x.startswith("[") and x.endswith("]"):
        try:
            return ast.literal_eval(x)
        except:
            pass

    # Case: "3,5,7"
    if "," in x:
        return [int(i) for i in x.split(",") if i.strip().isdigit()]

    # Case: "27"
    if x.isdigit():
        return [int(x)]

    return []

# Convert stringified lists → real lists (GoEmotions style "1,5,7")
# Example: "1, 5, 7" → [1,5,7]
df["labels"] = df["labels"].apply(parse_labels)

# Filter out empty labels
df = df[df["labels"].map(len) > 0]

# Train-test split
train, test = train_test_split(df, test_size=0.1, shuffle=True, random_state=42)

#Save
train.to_csv("data/processed/train.csv", index=False)
test.to_csv("data/processed/test.csv", index=False)

print("Sentiment Emotion Dataset Preprocessing done.")

df2 = pd.read_csv("data/raw/empathetic_dialogues.csv")

# Rename columns to simple names
df2 = df2.rename(columns={
    "empathetic_dialogues": "prompt",
    "labels": "response"
})

# Drop rows with missing data
df2 = df2.dropna(subset=["prompt", "response"])

# Clean text
df2["prompt"] = df2["prompt"].astype(str).str.strip()
df2["response"] = df2["response"].astype(str).str.strip()

# Save processed file
df2[["prompt", "response"]].to_csv("data/processed/response_pairs.csv", index=False)

print("Response dataset created.")