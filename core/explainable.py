from captum.attr import IntegratedGradients
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

class TextExplainer:
    def __init__(self, model_name):
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

    def explain(self, text):
        inputs = self.tokenizer(text, return_tensors="pt")
        ig = IntegratedGradients(self.model)
        attributions, delta = ig.attribute(inputs['input_ids'], target=0, return_convergence_delta=True)
        return attributions
