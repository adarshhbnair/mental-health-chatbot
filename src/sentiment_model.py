import torch
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    Trainer,
    TrainingArguments
)
import pandas as pd
from sklearn.preprocessing import MultiLabelBinarizer
from datasets import Dataset

df = pd.read_csv("data/processed/train.csv")
df["text"] = df["text"].astype(str)
df["labels"] = df["labels"].apply(eval)

mlb = MultiLabelBinarizer(classes=list(range(28)))
df_labels = mlb.fit_transform(df["labels"])

for i in range(28):
    df[f"label_{i}"] = df_labels[:, i]

dataset = Dataset.from_pandas(df[["text"] + [f"label_{i}" for i in range(28)]])

model_name = "distilbert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(model_name)

def tokenize(batch):
    enc = tokenizer(
        batch["text"],
        truncation=True,
        padding="max_length",
        max_length=64
    )

    labels = []
    for j in range(len(batch["text"])):
        row = [float(batch[f"label_{i}"][j]) for i in range(28)]
        labels.append(row)

    enc["labels"] = labels
    return enc

dataset = dataset.map(tokenize, batched=True)

model = AutoModelForSequenceClassification.from_pretrained(
    model_name,
    num_labels=28,
    problem_type="multi_label_classification"
)

args = TrainingArguments(
    output_dir="models/sentiment_emotion",
    per_device_train_batch_size=16,
    num_train_epochs=2,
    fp16=True,
)

trainer = Trainer(
    model=model,
    args=args,
    train_dataset=dataset
)

trainer.train()
