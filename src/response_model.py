import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, Trainer, TrainingArguments
import pandas as pd

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model_name = "google/flan-t5-xs"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name).to(device)

df = pd.read_csv("data/raw/empathetic_dialogues.csv")
inputs = list(df["prompt"])
labels = list(df["response"])

args = TrainingArguments(
    output_dir="models/response_gen",
    per_device_train_batch_size=4,
    num_train_epochs=1,
    fp16=True,
)

class Dataset(torch.utils.data.Dataset):
    def __init__(self, inp, lab):
        self.enc = tokenizer(inp, truncation=True, padding=True, return_tensors="pt")
        self.tgt = tokenizer(lab, truncation=True, padding=True, return_tensors="pt")

    def __getitem__(self, idx):
        return {
            "input_ids": self.enc["input_ids"][idx],
            "attention_mask": self.enc["attention_mask"][idx],
            "labels": self.tgt["input_ids"][idx],
        }

    def __len__(self):
        return len(inputs)

trainer = Trainer(model=model, args=args, train_dataset=Dataset(inputs, labels))
trainer.train()
