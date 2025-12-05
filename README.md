# Mental Health Support Chatbot (NLP + Transformers + Safety Filters)

A fully functional mental-health–assistive chatbot built with **PyTorch**, **HuggingFace Transformers**, and **NLP pipelines**.  
The system performs **emotion detection**, **sentiment analysis**, **response generation**, **anonymization**, and **safety filtering** to ensure responsible and empathetic interactions.

This project includes complete GPU-accelerated training (CUDA), dataset processing, model fine-tuning, and a CLI chatbot interface.

---

## 🚀 Features

### 🔹 **1. Emotion & Sentiment Classification**
- Fine-tuned **DistilBERT** on GoEmotions dataset  
- Detects 28 emotion categories  
- Optimized with mixed-precision **fp16** (when GPU available)  

### 🔹 **2. Empathetic Response Generation**
- Fine-tuned **T5-Small** on EmpatheticDialogues  
- Generates context-aware, empathetic, supportive responses  
- Custom training loop or HF Trainer support  

### 🔹 **3. Anonymization / Privacy Layer**
- spaCy-based NER  
- Removes user names, locations, orgs, dates → replaces with `[PERSON]`, `[GPE]`, etc.  

### 🔹 **4. Safety & Ethical Filtering**
- Rule-based detection for self-harm and dangerous content  
- Redirects to safe/helpful crisis responses  
- Ensures no harmful or unethical outputs  

### 🔹 **5. End-to-End Chatbot**
- Combined inference pipeline  
- CLI interface (`src/cli.py`)  
- Runs fully offline after training  

---

## 🏗️ Project Structure

```
mental_health_chatbot/
│
├── data/
│   ├── raw/
│   └── processed/
│
├── models/
│   ├── sentiment_emotion/
│   └── response_gen/
│
├── src/
│   ├── download_datasets.py
│   ├── preprocess.py
│   ├── sentiment_model.py
│   ├── response_model.py
│   ├── anonymizer.py
│   ├── safety_filter.py
│   ├── chatbot.py
│   └── cli.py
│
├── requirements.txt
└── README.md
```

---

## 📦 Setup Instructions (Windows + CUDA)

### 1️⃣ Create a Virtual Environment
```bash
python -m venv venv
venv\Scripts\activate
```

### 2️⃣ Install Dependencies
```bash
pip install -r requirements.txt
python -m spacy download en_core_web_sm
```

### 3️⃣ Install GPU-enabled PyTorch
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

---

## 📥 Download Datasets
```bash
python src/download_datasets.py
```

### Preprocess
```bash
python src/preprocess.py
```

---

## 🏋️ Train Models (GPU Accelerated)
### Train Emotion/Sentiment Model
```bash
python src/sentiment_model.py
```

### Train Response Generation Model
```bash
python src/response_model.py
```

Models will be saved automatically inside `models/`.

---

## 💬 Run the Chatbot
```bash
python src/cli.py
```

---

## 🔐 Ethical Considerations
This project implements:
- Anonymization of user inputs  
- Safety filters for self-harm or dangerous prompts  
- Non-judgmental empathetic response templates  

⚠️ **This is NOT a replacement for professional medical or psychological help.**

---

## 🛠️ Tech Stack
- Python  
- PyTorch  
- HuggingFace Transformers  
- spaCy  
- Datasets (GoEmotions, EmpatheticDialogues)  
- CUDA acceleration  

---

## 📄 License
MIT License — Free to use, modify, and share.

---

## 👤 Author
Adarsh B  
