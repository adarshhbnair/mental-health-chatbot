def safe(text):
    dangerous = ["suicide","kill","harm","die","end life"]
    if any(w in text.lower() for w in dangerous):
        return "I'm really sorry you're feeling this way. Please contact emergency services or a trusted person."
    return text
