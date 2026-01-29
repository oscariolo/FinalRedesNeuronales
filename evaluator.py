import numpy as np
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix, classification_report
from typing import List, Dict, Any, Tuple
import matplotlib.pyplot as plt
import seaborn as sns
from data_loader import DataLoader as SentimentDataLoader
import pandas as pd

class ModelEvaluator:
    """Model evaluation utilities for sentiment analysis."""
    
    def __init__(self, predictor, label_mapping: Dict[int, str] = None):
        self.predictor = predictor
        self.label_mapping = label_mapping or {0: 'negative', 1: 'positive'}
    
    def evaluate_predictions(self, true_labels: List[int], predictions: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Evaluate predictions against true labels."""
        predicted_labels = [pred['predicted_class'] for pred in predictions]
        
        # Calculate metrics
        accuracy = accuracy_score(true_labels, predicted_labels)
        precision, recall, f1, support = precision_recall_fscore_support(
            true_labels, predicted_labels, average='weighted'
        )
        
        # Per-class metrics
        precision_per_class, recall_per_class, f1_per_class, support_per_class = precision_recall_fscore_support(
            true_labels, predicted_labels, average=None
        )
        
        # Confusion matrix
        cm = confusion_matrix(true_labels, predicted_labels)
        
        # Classification report
        unique_labels = sorted(set(true_labels))
        class_names = [self.label_mapping.get(i, f'class_{i}') for i in unique_labels]
        report = classification_report(true_labels, predicted_labels, target_names=class_names, output_dict=True)
        
        return {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'support': support,
            'per_class_metrics': {
                'precision': precision_per_class.tolist(),
                'recall': recall_per_class.tolist(),
                'f1_score': f1_per_class.tolist(),
                'support': support_per_class.tolist()
            },
            'confusion_matrix': cm.tolist(),
            'classification_report': report
        }
    
    def evaluate_dataset(self, texts: List[str], true_labels: List[int], 
                        threshold: float = 0.5, batch_size: int = 32) -> Dict[str, Any]:
        """Evaluate model on a dataset."""
        print(f"Evaluating model on {len(texts)} samples...")
        
        # Get predictions
        predictions = self.predictor.predict_batch(
            texts, threshold=threshold, batch_size=batch_size
        )
        
        # Calculate metrics
        metrics = self.evaluate_predictions(true_labels, predictions)
        
        print(f"Accuracy: {metrics['accuracy']:.4f}")
        print(f"F1-Score: {metrics['f1_score']:.4f}")
        print(f"Precision: {metrics['precision']:.4f}")
        print(f"Recall: {metrics['recall']:.4f}")
        
        return {
            'metrics': metrics,
            'predictions': predictions,
            'num_samples': len(texts)
        }
    
    def compare_models(self, models_and_predictors: Dict[str, Any], 
                      texts: List[str], true_labels: List[int], 
                      threshold: float = 0.5) -> pd.DataFrame:
        """Compare multiple models on the same dataset."""
        results = []
        
        for model_name, predictor in models_and_predictors.items():
            print(f"\\nEvaluating {model_name}...")
            
            # Temporarily switch predictor
            original_predictor = self.predictor
            self.predictor = predictor
            
            eval_result = self.evaluate_dataset(texts, true_labels, threshold)
            metrics = eval_result['metrics']
            
            results.append({
                'Model': model_name,
                'Accuracy': metrics['accuracy'],
                'Precision': metrics['precision'],
                'Recall': metrics['recall'],
                'F1-Score': metrics['f1_score'],
                'Samples': len(texts)
            })
            
            # Restore original predictor
            self.predictor = original_predictor
        
        df = pd.DataFrame(results)
        df = df.sort_values('F1-Score', ascending=False)
        
        print("\\n=== Model Comparison Results ===")
        print(df.to_string(index=False, float_format='%.4f'))
        
        return df
    
    def plot_confusion_matrix(self, true_labels: List[int], predictions: List[Dict[str, Any]], 
                             title: str = "Confusion Matrix", figsize: Tuple[int, int] = (8, 6)):
        """Plot confusion matrix."""
        predicted_labels = [pred['predicted_class'] for pred in predictions]
        cm = confusion_matrix(true_labels, predicted_labels)
        
        plt.figure(figsize=figsize)
        class_names = [self.label_mapping.get(i, f'Class {i}') for i in range(len(cm))]
        
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=class_names, yticklabels=class_names)
        plt.title(title)
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.tight_layout()
        plt.show()
    
    def analyze_confidence_distribution(self, predictions: List[Dict[str, Any]], 
                                      true_labels: List[int] = None, 
                                      figsize: Tuple[int, int] = (12, 4)):
        """Analyze confidence distribution of predictions."""
        confidences = [pred['confidence'] for pred in predictions]
        
        if true_labels:
            correct_predictions = [pred['predicted_class'] == true_label 
                                 for pred, true_label in zip(predictions, true_labels)]
            correct_confidences = [conf for conf, correct in zip(confidences, correct_predictions) if correct]
            incorrect_confidences = [conf for conf, correct in zip(confidences, correct_predictions) if not correct]
        
        fig, axes = plt.subplots(1, 2 if true_labels else 1, figsize=figsize)
        
        if true_labels:
            axes[0].hist(correct_confidences, bins=20, alpha=0.7, label='Correct', color='green')
            axes[0].hist(incorrect_confidences, bins=20, alpha=0.7, label='Incorrect', color='red')
            axes[0].set_xlabel('Confidence')
            axes[0].set_ylabel('Frequency')
            axes[0].set_title('Confidence Distribution by Correctness')
            axes[0].legend()
            
            axes[1].hist(confidences, bins=30, alpha=0.7, color='blue')
            axes[1].set_xlabel('Confidence')
            axes[1].set_ylabel('Frequency')
            axes[1].set_title('Overall Confidence Distribution')
        else:
            axes.hist(confidences, bins=30, alpha=0.7, color='blue')
            axes.set_xlabel('Confidence')
            axes.set_ylabel('Frequency')
            axes.set_title('Confidence Distribution')
        
        plt.tight_layout()
        plt.show()
        
        return {
            'mean_confidence': np.mean(confidences),
            'std_confidence': np.std(confidences),
            'min_confidence': np.min(confidences),
            'max_confidence': np.max(confidences)
        }
    
    def generate_evaluation_report(self, texts: List[str], true_labels: List[int], 
                                  threshold: float = 0.5, save_path: str = None) -> Dict[str, Any]:
        """Generate a comprehensive evaluation report."""
        eval_result = self.evaluate_dataset(texts, true_labels, threshold)
        
        report = {
            'model_info': self.predictor.get_model_info(),
            'evaluation_params': {
                'threshold': threshold,
                'num_samples': len(texts)
            },
            'metrics': eval_result['metrics'],
            'sample_predictions': eval_result['predictions'][:10]  # First 10 predictions
        }
        
        if save_path:
            import json
            with open(save_path, 'w', encoding='utf-8') as f:
                json.dump(report, f, indent=2, ensure_ascii=False)
            print(f"Evaluation report saved to: {save_path}")
        
        return report