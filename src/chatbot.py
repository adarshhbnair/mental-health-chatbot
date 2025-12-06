import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification, AutoModelForSeq2SeqLM

from anonymizer import anonymize
from safety_filter import safe

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

sent_tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
sent_model = AutoModelForSequenceClassification.from_pretrained("models/sentiment_emotion").to(device)

resp_tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-small")
resp_model = AutoModelForSeq2SeqLM.from_pretrained("models/response_gen").to(device)

def chat(text):
    text = anonymize(text)
    text = safe(text)

    inp = resp_tokenizer(text, return_tensors="pt").to(device)
    out = resp_model.generate(**inp, max_length=80)
    return resp_tokenizer.decode(out[0], skip_special_tokens=True)
