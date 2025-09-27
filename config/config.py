import os

UPLOAD_FOLDER = "data/uploads/"
OUTPUT_FOLDER = "outputs/"

# HuggingFace models
SENTIMENT_MODEL = "distilbert-base-uncased-finetuned-sst-2-english"
EMOTION_MODEL = "nateraw/bert-base-uncased-emotion"

# LangChain / Ollama
LLM_MODEL = "ollama/llama2-13b"  # replace with your Ollama model
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
