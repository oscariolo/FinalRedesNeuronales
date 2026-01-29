from transformers import (
    BertForSequenceClassification, BertTokenizer,
    RobertaForSequenceClassification, RobertaTokenizer,
    AutoTokenizer, AutoModelForSequenceClassification
)
from typing import Literal, Tuple, Any
import torch

MODEL_CONFIGS = {
    "SaBert": {
        "model_name": "VerificadoProfesional/SaBERT-Spanish-Sentiment-Analysis",
        "model_class": BertForSequenceClassification,
        "tokenizer_class": BertTokenizer,
        "num_labels": 2
    },
    "Roberta": {
        "model_name": "cardiffnlp/twitter-xlm-roberta-base-sentiment",
        "model_class": RobertaForSequenceClassification,
        "tokenizer_class": RobertaTokenizer,
        "num_labels": 3
    },
    "DistilBert": {
        "model_name": "distilbert-base-uncased",
        "model_class": AutoModelForSequenceClassification,
        "tokenizer_class": AutoTokenizer,
        "num_labels": 2
    },
    "BertMultilingual": {
        "model_name": "bert-base-multilingual-cased",
        "model_class": BertForSequenceClassification,
        "tokenizer_class": BertTokenizer,
        "num_labels": 2
    }
}

def get_model_and_tokenizer(model_name: str, num_labels: int = None) -> Tuple[Any, Any]:
    """Get model and tokenizer for a given model name."""
    if model_name not in MODEL_CONFIGS:
        raise ValueError(f"Model {model_name} not supported. Available models: {list(MODEL_CONFIGS.keys())}")
    
    config = MODEL_CONFIGS[model_name]
    labels = num_labels if num_labels is not None else config["num_labels"]
    
    try:
        model = config["model_class"].from_pretrained(
            config["model_name"], 
            num_labels=labels
        )
        tokenizer = config["tokenizer_class"].from_pretrained(config["model_name"])
    except Exception as e:
        # Fallback to Auto classes
        model = AutoModelForSequenceClassification.from_pretrained(
            config["model_name"], 
            num_labels=labels
        )
        tokenizer = AutoTokenizer.from_pretrained(config["model_name"])
    
    return model, tokenizer

def load_fine_tuned_model(model_path: str, model_name: str = None) -> Tuple[Any, Any]:
    """Load a fine-tuned model from a saved path."""
    if model_name:
        config = MODEL_CONFIGS[model_name]
        model = config["model_class"].from_pretrained(model_path)
        tokenizer = config["tokenizer_class"].from_pretrained(model_path)
    else:
        model = AutoModelForSequenceClassification.from_pretrained(model_path)
        tokenizer = AutoTokenizer.from_pretrained(model_path)
    
    return model, tokenizer

def get_available_models() -> list:
    """Get list of available model names."""
    return list(MODEL_CONFIGS.keys())