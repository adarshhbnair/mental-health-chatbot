import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model_name = "models/response_model"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name).to(device)

def generate(text):
    prompt = f"User: {text}\nSupportive response:"

    enc = tokenizer(prompt, return_tensors="pt").to(device)

    with torch.no_grad():
        out = model.generate(
            **enc,
            max_length=128,
            min_length=20,
            num_beams=5,
            no_repeat_ngram_size=3,
            early_stopping=True,
            length_penalty=1.2
        )

    return tokenizer.decode(out[0], skip_special_tokens=True)

print(generate("I feel overwhelming."))

