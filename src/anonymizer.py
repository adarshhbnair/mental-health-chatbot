import spacy
nlp = spacy.load("en_core_web_sm")

def anonymize(text):
    doc = nlp(text)
    for ent in doc.ents:
        text = text.replace(ent.text, f"[{ent.label_}]")
    return text
