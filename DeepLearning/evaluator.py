import numpy as np
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix, classification_report, roc_curve, auc
from sklearn.preprocessing import label_binarize
from typing import List, Dict, Any, Tuple
import matplotlib.pyplot as plt
import seaborn as sns
from data_loader import DataLoader as SentimentDataLoader
import pandas as pd
import torch
from datetime import datetime
import os
from models import normalize_prediction_to_3class, normalize_label_to_3class

class ModelEvaluator:
    """Model evaluation utilities for sentiment analysis."""
    
    def __init__(self, predictor, label_mapping: Dict[int, str] = None, use_standardized: bool = True):
        self.predictor = predictor
        self.use_standardized = use_standardized
        # Use predictor's label_meaning if available, otherwise use provided or default
        if hasattr(predictor, 'label_meaning') and predictor.label_meaning:
            self.label_mapping = predictor.label_meaning
        else:
            self.label_mapping = label_mapping or {0: 'negative', 1: 'positive'}
    
    def evaluate_predictions(self, true_labels: List[int], predictions: List[Dict[str, Any]], 
                              use_sentiment_score: bool = False, normalize_labels: bool = False,
                              model_name: str = None) -> Dict[str, Any]:
        """Evaluate predictions against true labels.
        
        Args:
            true_labels: List of true label indices
            predictions: List of prediction dictionaries
            use_sentiment_score: If True, use standardized sentiment_score for comparison
            normalize_labels: If True, normalize model outputs to 3-class format
            model_name: Name of the model (required if normalize_labels=True)
        """
        if use_sentiment_score and 'sentiment_score' in predictions[0]:
            # Use standardized sentiment scores for evaluation
            predicted_labels = [pred['sentiment_score'] for pred in predictions]
        else:
            predicted_labels = [pred['predicted_class'] for pred in predictions]
        
        # Normalize labels if cross-model evaluation
        if normalize_labels and model_name:
            predicted_labels = [normalize_prediction_to_3class(pred, model_name) 
                               for pred in predicted_labels]
            true_labels = [normalize_label_to_3class(label, source_num_classes=3) 
                        for label in true_labels]
        
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
                        threshold: float = 0.5, batch_size: int = 32,
                        normalize_labels: bool = False) -> Dict[str, Any]:
        """Evaluate model on a dataset.
        
        Args:
            texts: List of input texts
            true_labels: List of true labels
            threshold: Classification threshold
            batch_size: Batch size for prediction
            normalize_labels: If True, normalize to 3-class format for cross-model comparison
        """
        print(f"Evaluating model on {len(texts)} samples...")
        
        # Get predictions
        predictions = self.predictor.predict_batch(
            texts, threshold=threshold, batch_size=batch_size
        )
        
        # Calculate metrics with optional normalization
        metrics = self.evaluate_predictions(
            true_labels, predictions, 
            normalize_labels=normalize_labels,
            model_name=self.predictor.model_name
        )
        
        print(f"Accuracy: {metrics['accuracy']:.4f}")
        print(f"F1-Score: {metrics['f1_score']:.4f}")
        print(f"Precision: {metrics['precision']:.4f}")
        print(f"Recall: {metrics['recall']:.4f}")
        
        # Add sentiment distribution summary
        if predictions and 'sentiment_label' in predictions[0]:
            sentiment_counts = {}
            for pred in predictions:
                label = pred['sentiment_label']
                sentiment_counts[label] = sentiment_counts.get(label, 0) + 1
            print(f"Sentiment Distribution: {sentiment_counts}")
        
        return {
            'metrics': metrics,
            'predictions': predictions,
            'num_samples': len(texts),
            'label_mapping': self.label_mapping
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
    
    def get_prediction_probabilities(self, texts: List[str], batch_size: int = 32) -> np.ndarray:
        """Get prediction probabilities for ROC curve calculation."""
        all_probabilities = []
        
        # Use the model directly for probabilities
        device = next(self.predictor.model.parameters()).device
        self.predictor.model.eval()
        
        with torch.no_grad():
            for i in range(0, len(texts), batch_size):
                batch_texts = texts[i:i+batch_size]
                
                # Tokenize batch
                inputs = self.predictor.tokenizer(
                    batch_texts,
                    return_tensors="pt",
                    truncation=True,
                    padding=True,
                    max_length=128
                )
                
                # Move to device
                inputs = {key: value.to(device) for key, value in inputs.items()}
                
                # Get predictions
                outputs = self.predictor.model(**inputs)
                probabilities = torch.nn.functional.softmax(outputs.logits, dim=-1)
                all_probabilities.extend(probabilities.cpu().numpy())
        
        return np.array(all_probabilities)
    
    def plot_roc_curve(self, true_labels: List[int], texts: List[str], 
                      title: str = "ROC Curve", figsize: Tuple[int, int] = (12, 8),
                      save_path: str = None) -> Dict[str, float]:
        """Plot ROC curve for multiclass classification."""
        try:
            # Get probabilities
            y_proba = self.get_prediction_probabilities(texts)
            
            # Get unique classes and class names
            unique_classes = sorted(set(true_labels))
            n_classes = len(unique_classes)
            class_names = [self.label_mapping.get(i, f'Class {i}') for i in unique_classes]
            
            if n_classes == 2:
                # Binary classification
                fpr, tpr, _ = roc_curve(true_labels, y_proba[:, 1])
                roc_auc = auc(fpr, tpr)
                
                plt.figure(figsize=(8, 6))
                plt.plot(fpr, tpr, color='darkorange', linewidth=2,
                        label=f'ROC curve (AUC = {roc_auc:.2f})')
                plt.plot([0, 1], [0, 1], color='navy', linewidth=2, linestyle='--')
                plt.xlim([0.0, 1.0])
                plt.ylim([0.0, 1.05])
                plt.xlabel('False Positive Rate')
                plt.ylabel('True Positive Rate')
                plt.title(title)
                plt.legend(loc="lower right")
                plt.grid(True, alpha=0.3)
                
                roc_auc_scores = {'overall': roc_auc}
            else:
                # Multiclass classification
                # Binarize the output
                y_true_binary = label_binarize(true_labels, classes=unique_classes)
                
                # Compute ROC curve and ROC area for each class
                fpr = dict()
                tpr = dict()
                roc_auc = dict()
                
                for i in range(n_classes):
                    fpr[i], tpr[i], _ = roc_curve(y_true_binary[:, i], y_proba[:, i])
                    roc_auc[i] = auc(fpr[i], tpr[i])
                
                # Compute micro-average ROC curve and ROC area
                fpr["micro"], tpr["micro"], _ = roc_curve(y_true_binary.ravel(), y_proba.ravel())
                roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])
                
                # Plot ROC curves
                plt.figure(figsize=figsize)
                
                # Plot micro-average ROC curve
                plt.subplot(2, 2, 1)
                plt.plot(fpr["micro"], tpr["micro"],
                        label=f'Micro-average (AUC = {roc_auc["micro"]:.2f})',
                        color='deeppink', linestyle=':', linewidth=4)
                
                # Plot ROC curves for each class
                colors = ['aqua', 'darkorange', 'cornflowerblue', 'green', 'red']
                for i, color in enumerate(colors[:n_classes]):
                    plt.plot(fpr[i], tpr[i], color=color, linewidth=2,
                            label=f'{class_names[i]} (AUC = {roc_auc[i]:.2f})')
                
                plt.plot([0, 1], [0, 1], 'k--', linewidth=2)
                plt.xlim([0.0, 1.0])
                plt.ylim([0.0, 1.05])
                plt.xlabel('False Positive Rate')
                plt.ylabel('True Positive Rate')
                plt.title(f'ROC Curves - {title}')
                plt.legend(loc="lower right")
                plt.grid(True, alpha=0.3)
                
                # Plot individual class ROC curves
                for i in range(min(n_classes, 3)):  # Limit to 3 subplots
                    plt.subplot(2, 2, i+2)
                    plt.plot(fpr[i], tpr[i], color=colors[i % len(colors)], linewidth=2)
                    plt.plot([0, 1], [0, 1], 'k--', linewidth=2)
                    plt.xlim([0.0, 1.0])
                    plt.ylim([0.0, 1.05])
                    plt.xlabel('False Positive Rate')
                    plt.ylabel('True Positive Rate')
                    plt.title(f'{class_names[i]} (AUC = {roc_auc[i]:.2f})')
                    plt.grid(True, alpha=0.3)
                
                roc_auc_scores = roc_auc.copy()
            
            plt.tight_layout()
            
            # Save the plot if path provided
            if save_path:
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
                print(f"ROC curve saved as: {save_path}")
            
            plt.show()
            
            return roc_auc_scores
            
        except Exception as e:
            print(f"Error generating ROC curve: {e}")
            return {}
    
    def plot_roc_curve_multiclass(self, true_labels: List[int], texts: List[str], 
                                 title: str = "ROC Curves", figsize: Tuple[int, int] = (12, 8),
                                 save_path: str = None) -> Dict[str, float]:
        """Plot ROC curve for multiclass classification."""
        print("Generating ROC curves...")
        
        # Convert to numpy arrays for safety
        true_labels = np.array(true_labels)
        
        # Get prediction probabilities
        y_proba = self.get_prediction_probabilities(texts)
        
        # Get actual number of classes from predictions and true labels
        n_classes_pred = y_proba.shape[1]  # From model predictions
        n_classes_true = len(np.unique(true_labels))  # From actual labels
        n_classes = max(n_classes_pred, n_classes_true)
        
        # Get class names - only use as many as we actually have
        class_names = [self.label_mapping.get(i, f'Class {i}') for i in range(n_classes_pred)]
        
        # Binarize the output using actual number of classes in true labels
        y_true_binary = label_binarize(true_labels, classes=list(range(n_classes)))
        
        # Handle binary case
        if n_classes == 2:
            y_true_binary = np.column_stack([1 - true_labels, true_labels])
            y_proba = np.column_stack([1 - y_proba[:, 1], y_proba[:, 1]])
        
        # Compute ROC curve and ROC area for each class
        fpr = dict()
        tpr = dict()
        roc_auc = dict()
        
        for i in range(n_classes):
            if i < y_true_binary.shape[1]:  # Only compute for classes that exist in data
                fpr[i], tpr[i], _ = roc_curve(y_true_binary[:, i], y_proba[:, i])
                roc_auc[i] = auc(fpr[i], tpr[i])
        
        # Compute micro-average ROC curve and ROC area
        fpr["micro"], tpr["micro"], _ = roc_curve(y_true_binary.ravel(), y_proba.ravel())
        roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])
        
        # Plot ROC curves
        plt.figure(figsize=figsize)
        
        if n_classes > 2:
            # Plot micro-average ROC curve
            plt.subplot(2, 2, 1)
            plt.plot(fpr["micro"], tpr["micro"],
                     label=f'Micro-average ROC curve (AUC = {roc_auc["micro"]:.2f})',
                     color='deeppink', linestyle=':', linewidth=4)
            
            # Plot ROC curves for each class
            colors = ['aqua', 'darkorange', 'cornflowerblue', 'green', 'red']
            for i, color in enumerate(colors[:n_classes]):
                plt.plot(fpr[i], tpr[i], color=color, linewidth=2,
                         label=f'{class_names[i]} (AUC = {roc_auc[i]:.2f})')
            
            plt.plot([0, 1], [0, 1], 'k--', linewidth=2)
            plt.xlim([0.0, 1.0])
            plt.ylim([0.0, 1.05])
            plt.xlabel('False Positive Rate')
            plt.ylabel('True Positive Rate')
            plt.title(f'ROC Curves - {title}')
            plt.legend(loc="lower right")
            
            # Plot individual class ROC curves
            for i in range(min(n_classes, 3)):  # Max 3 individual plots
                plt.subplot(2, 2, i+2)
                plt.plot(fpr[i], tpr[i], color=colors[i % len(colors)], linewidth=2)
                plt.plot([0, 1], [0, 1], 'k--', linewidth=2)
                plt.xlim([0.0, 1.0])
                plt.ylim([0.0, 1.05])
                plt.xlabel('False Positive Rate')
                plt.ylabel('True Positive Rate')
                class_label = class_names[i] if i < len(class_names) else f'Class {i}'
                plt.title(f'{class_label} (AUC = {roc_auc[i]:.2f})')
                plt.grid(True, alpha=0.3)
        else:
            # Binary classification - single ROC curve
            if 1 in roc_auc:
                plt.plot(fpr[1], tpr[1], color='darkorange', linewidth=2,
                     label=f'ROC curve (AUC = {roc_auc[1]:.2f})')
            plt.plot([0, 1], [0, 1], 'k--', linewidth=2)
            plt.xlim([0.0, 1.0])
            plt.ylim([0.0, 1.05])
            plt.xlabel('False Positive Rate')
            plt.ylabel('True Positive Rate')
            plt.title(f'ROC Curve - {title}')
            plt.legend(loc="lower right")
            plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # Save the plot if path provided
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"ROC curve saved as: {save_path}")
        else:
            # Auto-generate filename with timestamp
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"roc_curve_{timestamp}.png"
            plt.savefig(filename, dpi=300, bbox_inches='tight')
            print(f"ROC curve saved as: {filename}")
        
        plt.show()
        
        # Print AUC scores
        print("\n=== AUC Scores ===")
        for i in range(n_classes):
            if i in roc_auc:
                class_label = class_names[i] if i < len(class_names) else f'Class {i}'
                print(f"{class_label}: {roc_auc[i]:.4f}")
        if n_classes > 2:
            print(f"Micro-average: {roc_auc['micro']:.4f}")
        
        return roc_auc
    
    def get_prediction_probabilities(self, texts: List[str], batch_size: int = 32) -> np.ndarray:
        """Get prediction probabilities for ROC curve calculation."""
        all_probabilities = []
        
        # Use the model directly for probabilities
        device = next(self.predictor.model.parameters()).device
        self.predictor.model.eval()
        
        with torch.no_grad():
            for i in range(0, len(texts), batch_size):
                batch_texts = texts[i:i+batch_size]
                
                # Tokenize batch
                inputs = self.predictor.tokenizer(
                    batch_texts,
                    return_tensors="pt",
                    truncation=True,
                    padding=True,
                    max_length=128
                )
                
                # Move to device
                inputs = {key: value.to(device) for key, value in inputs.items()}
                
                # Get predictions
                outputs = self.predictor.model(**inputs)
                probabilities = torch.nn.functional.softmax(outputs.logits, dim=-1)
                all_probabilities.extend(probabilities.cpu().numpy())
        
        return np.array(all_probabilities)