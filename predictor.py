import torch
import numpy as np
from typing import List, Dict, Any, Union, Tuple
from models import load_fine_tuned_model

class Predictor:
    """Prediction utilities for sentiment analysis models."""
    
    def __init__(self, model, tokenizer, device: str = None):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device if device else ('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)
        self.model.eval()
    
    @classmethod
    def from_pretrained_path(cls, model_path: str, model_name: str = None):
        """Load predictor from a fine-tuned model path."""
        model, tokenizer = load_fine_tuned_model(model_path, model_name)
        return cls(model, tokenizer)
    
    def predict_single(self, text: str, threshold: float = 0.5, max_length: int = 128) -> Dict[str, Any]:
        """Predict sentiment for a single text."""
        inputs = self.tokenizer(
            text, 
            return_tensors="pt", 
            padding=True, 
            truncation=True, 
            max_length=max_length
        )
        
        # Move inputs to device
        inputs = {key: value.to(self.device) for key, value in inputs.items()}
        
        with torch.no_grad():
            outputs = self.model(**inputs)
        
        logits = outputs.logits
        probabilities = torch.softmax(logits, dim=1).squeeze()
        
        if probabilities.dim() == 0:  # Handle single dimension case
            probabilities = probabilities.unsqueeze(0)
        
        predicted_class = torch.argmax(logits, dim=1).item()
        confidence = probabilities[predicted_class].item()
        
        # Apply threshold logic for binary classification
        if len(probabilities) == 2 and confidence <= threshold and predicted_class == 1:
            predicted_class = 0
            confidence = probabilities[0].item()
        
        return {
            'predicted_class': predicted_class,
            'confidence': confidence,
            'probabilities': probabilities.cpu().tolist(),
            'text': text
        }
    
    def predict_batch(self, texts: List[str], threshold: float = 0.5, 
                     max_length: int = 128, batch_size: int = 32) -> List[Dict[str, Any]]:
        """Predict sentiment for a batch of texts."""
        results = []
        
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i + batch_size]
            
            inputs = self.tokenizer(
                batch_texts,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=max_length
            )
            
            # Move inputs to device
            inputs = {key: value.to(self.device) for key, value in inputs.items()}
            
            with torch.no_grad():
                outputs = self.model(**inputs)
            
            logits = outputs.logits
            probabilities = torch.softmax(logits, dim=1)
            predicted_classes = torch.argmax(logits, dim=1)
            
            for j, text in enumerate(batch_texts):
                probs = probabilities[j]
                pred_class = predicted_classes[j].item()
                confidence = probs[pred_class].item()
                
                # Apply threshold logic for binary classification
                if len(probs) == 2 and confidence <= threshold and pred_class == 1:
                    pred_class = 0
                    confidence = probs[0].item()
                
                results.append({
                    'predicted_class': pred_class,
                    'confidence': confidence,
                    'probabilities': probs.cpu().tolist(),
                    'text': text
                })
        
        return results
    
    def predict_with_labels(self, texts: List[str], label_mapping: Dict[int, str] = None, 
                          threshold: float = 0.5, max_length: int = 128) -> List[Dict[str, Any]]:
        """Predict with human-readable labels."""
        results = self.predict_batch(texts, threshold, max_length)
        
        if label_mapping is None:
            # Default mapping for binary sentiment
            label_mapping = {0: 'negative', 1: 'positive'}
        
        for result in results:
            result['predicted_label'] = label_mapping.get(
                result['predicted_class'], 
                f"class_{result['predicted_class']}"
            )
        
        return results
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the current model."""
        return {
            'model_type': type(self.model).__name__,
            'tokenizer_type': type(self.tokenizer).__name__,
            'device': str(self.device),
            'num_parameters': sum(p.numel() for p in self.model.parameters()),
            'vocab_size': len(self.tokenizer.vocab) if hasattr(self.tokenizer, 'vocab') else 'Unknown'
        }

def create_predictor_from_model_name(model_name: str, model_path: str = None) -> Predictor:
    """Create predictor from model name or fine-tuned model path."""
    if model_path:
        return Predictor.from_pretrained_path(model_path, model_name)
    else:
        from models import get_model_and_tokenizer
        model, tokenizer = get_model_and_tokenizer(model_name)
        return Predictor(model, tokenizer)