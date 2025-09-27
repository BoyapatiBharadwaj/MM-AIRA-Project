from transformers import pipeline
from config.config import SENTIMENT_MODEL, EMOTION_MODEL

class SentimentEmotionAnalyzer:
    def __init__(self):
        self.sentiment = pipeline(
            "sentiment-analysis",
            model=SENTIMENT_MODEL,
            tokenizer=SENTIMENT_MODEL,
            truncation=True
        )
        self.emotion = pipeline(
            "text-classification",
            model=EMOTION_MODEL,
            tokenizer=EMOTION_MODEL,
            truncation=True,
            return_all_scores=True
        )

    def analyze_sentiment(self, text):
        return self.sentiment(text)

    def analyze_emotion(self, text):
        return self.emotion(text)
