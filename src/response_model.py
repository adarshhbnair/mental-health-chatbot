import os
os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"

import torch
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, TrainingArguments, Trainer
import pandas as pd

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load dataset
df = pd.read_csv("data/processed/response_pairs.csv")

# Convert to HF dataset
dataset = Dataset.from_pandas(df)

# Model
model_name = "t5-small"
tokenizer = AutoTokenizer.from_pretrained(model_name)

def preprocess(batch):
    inputs = ["respond empathetically: " + p for p in batch["prompt"]]
    targets = batch["response"]

    model_inputs = tokenizer(inputs, truncation=True, padding="max_length", max_length=128)
    labels = tokenizer(targets, truncation=True, padding="max_length", max_length=128).input_ids

    model_inputs["labels"] = labels
    return model_inputs

dataset = dataset.map(preprocess, batched=True)

model = AutoModelForSeq2SeqLM.from_pretrained(model_name).to(device)

args = TrainingArguments(
    output_dir="models/response_model",
    per_device_train_batch_size=8,
    num_train_epochs=2,
    logging_steps=50,
    save_steps=500,
    fp16=True,
)

trainer = Trainer(
    model=model,
    args=args,
    train_dataset=dataset,
)

trainer.train()

model.save_pretrained("models/response_model")
tokenizer.save_pretrained("models/response_model")

print("Training complete.")
