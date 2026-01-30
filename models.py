from transformers import (
    BertForSequenceClassification, BertTokenizer,
    RobertaForSequenceClassification, RobertaTokenizer,
    AutoTokenizer, AutoModelForSequenceClassification
)
from typing import Literal, Tuple, Any, Dict
import torch
import os
import shutil

# Standardized sentiment labels
STANDARD_LABELS = {
    "very_negative": -2,
    "negative": -1,
    "neutral": 0,
    "positive": 1,
    "very_positive": 2
}

MODEL_CONFIGS = {
    "SaBert": {
        "model_name": "VerificadoProfesional/SaBERT-Spanish-Sentiment-Analysis",
        "model_class": BertForSequenceClassification,
        "tokenizer_class": BertTokenizer,
        "num_labels": 2,
        "label_meaning": {
            0: "negative",
            1: "positive"
        }
    },
    "Roberta": {
        "model_name": "cardiffnlp/twitter-xlm-roberta-base-sentiment",
        "model_class": RobertaForSequenceClassification,
        "tokenizer_class": RobertaTokenizer,
        "num_labels": 3,
        "label_meaning": {
            0: "negative",
            1: "neutral",
            2: "positive"
        }
    },
    "Tabularisai": {
        "model_name": "tabularisai/multilingual-sentiment-analysis",
        "model_class": AutoModelForSequenceClassification,
        "tokenizer_class": AutoTokenizer,
        "num_labels": 5,
        "label_meaning": {
            0: "very_negative",
            1: "negative",
            2: "neutral",
            3: "positive",
            4: "very_positive"
        }
    }
}

# Cache directory for downloaded models
CACHE_DIR = "model_cache"

def ensure_cache_dir():
    """Ensure cache directory exists."""
    if not os.path.exists(CACHE_DIR):
        os.makedirs(CACHE_DIR)

def get_cache_path(model_name: str) -> str:
    """Get cache path for a model."""
    return os.path.join(CACHE_DIR, model_name.replace("/", "_"))

def is_model_cached(model_name: str) -> bool:
    """Check if model is already cached."""
    cache_path = get_cache_path(model_name)
    return os.path.exists(cache_path) and len(os.listdir(cache_path)) > 0

def get_model_and_tokenizer(model_name: str, num_labels: int = None) -> Tuple[Any, Any]:
    """Get model and tokenizer for a given model name with caching."""
    if model_name not in MODEL_CONFIGS:
        raise ValueError(f"Model {model_name} not supported. Available models: {list(MODEL_CONFIGS.keys())}")
    
    config = MODEL_CONFIGS[model_name]
    labels = num_labels if num_labels is not None else config["num_labels"]
    model_hf_name = config["model_name"]
    
    ensure_cache_dir()
    cache_path = get_cache_path(model_hf_name)
    
    # Check if model is cached
    if is_model_cached(model_hf_name):
        print(f"Loading {model_name} from cache: {cache_path}")
        try:
            if model_name in ["SaBert", "BertMultilingual"]:
                model = config["model_class"].from_pretrained(cache_path, num_labels=labels)
                tokenizer = config["tokenizer_class"].from_pretrained(cache_path)
            else:
                model = AutoModelForSequenceClassification.from_pretrained(cache_path, num_labels=labels)
                tokenizer = AutoTokenizer.from_pretrained(cache_path)
            return model, tokenizer
        except Exception as e:
            print(f"Error loading from cache: {e}. Downloading fresh model...")
            # If cache is corrupted, remove it and download fresh
            if os.path.exists(cache_path):
                shutil.rmtree(cache_path)
    
    print(f"Downloading {model_name} and caching to: {cache_path}")
    
    try:
        model = config["model_class"].from_pretrained(
            model_hf_name, 
            num_labels=labels
        )
        tokenizer = config["tokenizer_class"].from_pretrained(model_hf_name)
    except Exception as e:
        # Fallback to Auto classes
        model = AutoModelForSequenceClassification.from_pretrained(
            model_hf_name, 
            num_labels=labels
        )
        tokenizer = AutoTokenizer.from_pretrained(model_hf_name)
    
    # Save to cache
    try:
        model.save_pretrained(cache_path)
        tokenizer.save_pretrained(cache_path)
        print(f"Model {model_name} cached successfully")
    except Exception as e:
        print(f"Warning: Could not cache model {model_name}: {e}")
    
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

def get_label_meaning(model_name: str) -> Dict[int, str]:
    """Get label meaning mapping for a model."""
    if model_name not in MODEL_CONFIGS:
        raise ValueError(f"Model {model_name} not supported.")
    return MODEL_CONFIGS[model_name].get("label_meaning", {0: "negative", 1: "positive"})

def get_standardized_sentiment(model_name: str, predicted_class: int) -> Tuple[str, int]:
    """Convert model-specific prediction to standardized sentiment.
    
    Returns:
        Tuple of (sentiment_label, sentiment_score)
        sentiment_score: -2 (very_negative) to +2 (very_positive)
    """
    label_meaning = get_label_meaning(model_name)
    sentiment_label = label_meaning.get(predicted_class, "unknown")
    sentiment_score = STANDARD_LABELS.get(sentiment_label, 0)
    return sentiment_label, sentiment_score