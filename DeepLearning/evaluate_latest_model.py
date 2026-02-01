import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix, classification_report, roc_curve, auc
from sklearn.preprocessing import label_binarize
import seaborn as sns
import torch
from datetime import datetime

from models import load_fine_tuned_model, get_label_meaning
from data_loader import DataLoader
from predictor import Predictor
from evaluator import ModelEvaluator

# Define label mapping function for sentiment analysis
def sentiment_label_mapper(label):
    """Map sentiment labels to integers: -1=0 (negative), 0=1 (neutral), 1=2 (positive)"""
    if isinstance(label, str):
        label_mapping = {'N': 0, 'NEU': 1, 'P': 2}
        if label in label_mapping:
            return label_mapping[label]
    
    # Handle numeric labels: -1, 0, 1 -> 0, 1, 2
    try:
        numeric_label = float(label)
        if numeric_label == -1.0:
            return 0  # negative
        elif numeric_label == 0.0:
            return 1  # neutral
        elif numeric_label == 1.0:
            return 2  # positive
        else:
            print(f"Warning: Unexpected numeric label '{label}', assigning as neutral (1)")
            return 1
    except (ValueError, TypeError):
        print(f"Warning: Unknown label '{label}', assigning as neutral (1)")
        return 1  # Default to neutral

def find_latest_model():
    """Find the most recent fine-tuned model based on timestamp."""
    fine_tuned_dir = "fine_tuned_models"
    if not os.path.exists(fine_tuned_dir):
        raise FileNotFoundError(f"Fine-tuned models directory '{fine_tuned_dir}' not found")
    
    model_dirs = [d for d in os.listdir(fine_tuned_dir) if os.path.isdir(os.path.join(fine_tuned_dir, d))]
    if not model_dirs:
        raise FileNotFoundError("No fine-tuned models found")
    
    # Sort by timestamp in filename (assuming format: ModelName_finetuned_YYYYMMDD_HHMMSS)
    model_dirs.sort()
    latest_model = model_dirs[-1]
    latest_model_path = os.path.join(fine_tuned_dir, latest_model)
    
    print(f"Found latest model: {latest_model}")
    print(f"Model path: {latest_model_path}")
    
    return latest_model_path, latest_model

def plot_roc_curve_multiclass(y_true, y_proba, class_names, model_name):
    """Plot ROC curve for multiclass classification."""
    n_classes = len(class_names)
    
    # Binarize the output
    y_true_binary = label_binarize(y_true, classes=list(range(n_classes)))
    
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
    plt.figure(figsize=(12, 8))
    
    # Plot micro-average ROC curve
    plt.subplot(2, 2, 1)
    plt.plot(fpr["micro"], tpr["micro"],
             label=f'Micro-average ROC curve (AUC = {roc_auc["micro"]:.2f})',
             color='deeppink', linestyle=':', linewidth=4)
    
    # Plot ROC curves for each class
    colors = ['aqua', 'darkorange', 'cornflowerblue']
    for i, color in enumerate(colors[:n_classes]):
        plt.plot(fpr[i], tpr[i], color=color, linewidth=2,
                 label=f'ROC curve {class_names[i]} (AUC = {roc_auc[i]:.2f})')
    
    plt.plot([0, 1], [0, 1], 'k--', linewidth=2)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'ROC Curves - {model_name}')
    plt.legend(loc="lower right")
    
    # Plot individual class ROC curves
    for i in range(n_classes):
        plt.subplot(2, 2, i+2)
        plt.plot(fpr[i], tpr[i], color=colors[i % len(colors)], linewidth=2)
        plt.plot([0, 1], [0, 1], 'k--', linewidth=2)
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title(f'ROC Curve - {class_names[i]} (AUC = {roc_auc[i]:.2f})')
        plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save the plot
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"roc_curve_{model_name.replace('/', '_')}_{timestamp}.png"
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    print(f"ROC curve saved as: {filename}")
    plt.show()
    
    return roc_auc

def get_prediction_probabilities(predictor, texts, batch_size=32):
    """Get prediction probabilities for ROC curve calculation."""
    all_probabilities = []
    
    # Use the model directly for probabilities
    device = next(predictor.model.parameters()).device
    predictor.model.eval()
    
    with torch.no_grad():
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i+batch_size]
            
            # Tokenize batch
            inputs = predictor.tokenizer(
                batch_texts,
                return_tensors="pt",
                truncation=True,
                padding=True,
                max_length=128
            )
            
            # Move to device
            inputs = {key: value.to(device) for key, value in inputs.items()}
            
            # Get predictions
            outputs = predictor.model(**inputs)
            probabilities = torch.nn.functional.softmax(outputs.logits, dim=-1)
            all_probabilities.extend(probabilities.cpu().numpy())
    
    return np.array(all_probabilities)

def evaluate_latest_model():
    """Evaluate the latest fine-tuned model."""
    print("=== Evaluating Latest Fine-Tuned Model ===\n")
    
    # Find latest model
    try:
        latest_model_path, model_name = find_latest_model()
    except FileNotFoundError as e:
        print(f"Error: {e}")
        return
    
    # Load the model
    print("Loading model...")
    try:
        model, tokenizer = load_fine_tuned_model(latest_model_path)
        print(f"Successfully loaded model from: {latest_model_path}")
    except Exception as e:
        print(f"Error loading model: {e}")
        return
    
    # Create predictor
    predictor = Predictor(model, tokenizer, model_name=model_name)
    
    # Load test data
    print("Loading test data...")
    data_loader = DataLoader(tokenizer)
    try:
        texts, labels = data_loader.load_from_csv(
            "JoinedData/FULLDATA.csv", 
            "fullText", 
            "label", 
            label_mapper=sentiment_label_mapper
        )
        print(f"Loaded {len(texts)} samples")
        print(f"Label distribution: {dict(zip(*np.unique(labels, return_counts=True)))}")
    except Exception as e:
        print(f"Error loading data: {e}")
        return
    
    # Split data for evaluation (use a subset for faster evaluation)
    from sklearn.model_selection import train_test_split
    _, test_texts, _, test_labels = train_test_split(
        texts, labels, test_size=0.3, random_state=42, stratify=labels
    )
    
    print(f"Using {len(test_texts)} samples for evaluation")
    
    # Create evaluator
    evaluator = ModelEvaluator(predictor)
    
    # Evaluate the model
    print("\\n=== Running Evaluation ===")
    eval_results = evaluator.evaluate_dataset(test_texts, test_labels, batch_size=16)
    
    # Print detailed metrics
    metrics = eval_results['metrics']
    print(f"\\n=== Detailed Metrics ===")
    print(f"Accuracy: {metrics['accuracy']:.4f}")
    print(f"Precision (weighted): {metrics['precision']:.4f}")
    print(f"Recall (weighted): {metrics['recall']:.4f}")
    print(f"F1-Score (weighted): {metrics['f1_score']:.4f}")
    
    # Per-class metrics
    print("\\n=== Per-Class Metrics ===")
    class_names = ['Negative', 'Neutral', 'Positive']
    per_class = metrics['per_class_metrics']
    for i, class_name in enumerate(class_names):
        if i < len(per_class['precision']):
            print(f"{class_name}:")
            print(f"  Precision: {per_class['precision'][i]:.4f}")
            print(f"  Recall: {per_class['recall'][i]:.4f}")
            print(f"  F1-Score: {per_class['f1_score'][i]:.4f}")
            print(f"  Support: {per_class['support'][i]}")
    
    # Plot confusion matrix
    print("\\n=== Confusion Matrix ===")
    evaluator.plot_confusion_matrix(test_labels, eval_results['predictions'], 
                                  title=f"Confusion Matrix - {model_name}")
    
    # Generate ROC curve
    print("\\n=== Generating ROC Curve ===")
    try:
        probabilities = get_prediction_probabilities(predictor, test_texts)
        roc_auc_scores = plot_roc_curve_multiclass(
            test_labels, probabilities, class_names, model_name
        )
        print(f"AUC scores: {roc_auc_scores}")
    except Exception as e:
        print(f"Error generating ROC curve: {e}")
    
    # Analyze confidence distribution
    print("\\n=== Confidence Analysis ===")
    conf_stats = evaluator.analyze_confidence_distribution(
        eval_results['predictions'], test_labels
    )
    print(f"Confidence Statistics: {conf_stats}")
    
    # Save evaluation report
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_filename = f"evaluation_report_{model_name.replace('/', '_')}_{timestamp}.json"
    try:
        evaluator.generate_evaluation_report(test_texts, test_labels, save_path=report_filename)
    except Exception as e:
        print(f"Warning: Could not save report: {e}")
    
    print(f"\\n=== Evaluation Complete ===")
    print(f"Model: {model_name}")
    print(f"Model Path: {latest_model_path}")
    print(f"Test Samples: {len(test_texts)}")
    print(f"Overall Accuracy: {metrics['accuracy']:.4f}")
    print(f"Overall F1-Score: {metrics['f1_score']:.4f}")

if __name__ == "__main__":
    evaluate_latest_model()