

from models import MODEL_CONFIGS, get_model_and_tokenizer, get_available_models, load_fine_tuned_model
from data_loader import DataLoader, create_sample_data
from fine_tuner import FineTuner
from predictor import Predictor, create_predictor_from_model_name
from evaluator import ModelEvaluator
import os
from datetime import datetime
from datetime import datetime
from sklearn.model_selection import train_test_split

# Define label mapping function for sentiment analysis
def sentiment_label_mapper(label):
    """Map sentiment labels to integers: -1=0 (negative), 0=1 (neutral), 1=2 (positive)
    
    PyTorch requires labels to be non-negative integers starting from 0.
    This function converts your dataset's -1, 0, 1 format to 0, 1, 2.
    """
    # Handle both numeric and string labels
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


def evaluate_latest_model_complete():
    """Evaluate the latest fine-tuned model with complete metrics and ROC curve."""
    print("\n=== Evaluating Latest Fine-Tuned Model ===\n")
    
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
        print(f"Label distribution: {dict(zip(*__import__('numpy').unique(labels, return_counts=True)))}") 
    except Exception as e:
        print(f"Error loading data: {e}")
        return
    
    # Split data for evaluation (use a subset for faster evaluation)
    _, test_texts, _, test_labels = train_test_split(
        texts, labels, test_size=0.3, random_state=42, stratify=labels
    )
    
    print(f"Using {len(test_texts)} samples for evaluation")
    
    # Create evaluator
    evaluator = ModelEvaluator(predictor)
    
    # Evaluate the model
    print("\n=== Running Evaluation ===")
    eval_results = evaluator.evaluate_dataset(test_texts, test_labels, batch_size=16)
    
    # Print detailed metrics
    metrics = eval_results['metrics']
    print(f"\n=== Detailed Metrics ===")
    print(f"Accuracy: {metrics['accuracy']:.4f}")
    print(f"Precision (weighted): {metrics['precision']:.4f}")
    print(f"Recall (weighted): {metrics['recall']:.4f}")
    print(f"F1-Score (weighted): {metrics['f1_score']:.4f}")
    
    # Per-class metrics
    print("\n=== Per-Class Metrics ===")
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
    print("\n=== Confusion Matrix ===")
    evaluator.plot_confusion_matrix(test_labels, eval_results['predictions'], 
                                  title=f"Confusion Matrix - {model_name}")
    
    # Generate ROC curve
    print("\n=== Generating ROC Curve ===")
    try:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        roc_filename = f"roc_curve_{model_name.replace('/', '_')}_{timestamp}.png"
        roc_auc_scores = evaluator.plot_roc_curve_multiclass(
            test_labels, test_texts, title=model_name, save_path=roc_filename
        )
    except Exception as e:
        print(f"Error generating ROC curve: {e}")
    
    # Analyze confidence distribution
    print("\n=== Confidence Analysis ===")
    conf_stats = evaluator.analyze_confidence_distribution(
        eval_results['predictions'], test_labels
    )
    print(f"Confidence Statistics: {conf_stats}")
    
    print(f"\n=== Evaluation Summary ===")
    print(f"Model: {model_name}")
    print(f"Model Path: {latest_model_path}")
    print(f"Test Samples: {len(test_texts)}")
    print(f"Overall Accuracy: {metrics['accuracy']:.4f}")
    print(f"Overall F1-Score: {metrics['f1_score']:.4f}")
    print(f"ROC curve saved as: {roc_filename}")

def evaluate_fine_tunned_models():
    
    fine_tuned_models = ["Roberta_finetuned_20260131_222853", "Tabularisai_finetuned_20260131_223634"]
    
    for model_name in fine_tuned_models:
        print(f"\n=== Evaluating Fine-tuned Model: {model_name} ===")
        
        # Load fine-tuned model
        model, tokenizer = load_fine_tuned_model(model_name)
        
        # Create predictor
        predictor = Predictor(model, tokenizer, model_name=model_name)
        
        # Create evaluator
        evaluator = ModelEvaluator(predictor)
        
        # Load evaluation data
        data_loader = DataLoader(tokenizer)
        texts, labels = data_loader.load_from_csv("JoinedData/FULLDATA.csv", "fullText", "label", label_mapper=sentiment_label_mapper)
        print(f"Loaded {len(texts)} texts for evaluation with {len(set(labels))} unique labels: {sorted(set(labels))}")
        
        # Evaluate model
        eval_results = evaluator.evaluate_dataset(texts, labels)
        print(f"Evaluation Results for {model_name}: {eval_results}")
    
    

def main():
    print("=== Sentiment Analysis Framework ===\n")
    
    # # 1. Show available models
    # print("Available models:", get_available_models())
    
    # # 2. Load sample data (replace with your own data loading logic)
    # texts, labels = create_sample_data()
    # print(f"Loaded {len(texts)} sample texts with {len(set(labels))} classes")
    
    # # 3. Example: Test a pre-trained model
    # print("\n=== Testing Pre-trained Model ===")
    # model_name = "Tabularisai"  # Change to any available model
    # model, tokenizer = get_model_and_tokenizer(model_name)
    
    # # Create data loader and predictor
    # data_loader = DataLoader(tokenizer)
    # predictor = Predictor(model, tokenizer, model_name=model_name)
    
    # # Test single prediction
    # test_text = "Un agente ICE confirma a una vecina que le graba legalmente que la va a incluir en su base de datos como terrorista. Viva la libertad carajo!"
    # result = predictor.predict_single(test_text, threshold=0.3)
    # print(f"Text: {test_text}")
    # print(f"Prediction: {result}")
    
    # # 4. Example: Evaluate model
    # print("\n=== Model Evaluation ===")
    # evaluator = ModelEvaluator(predictor)
    # eval_results = evaluator.evaluate_dataset(texts, labels)
    
    # #5. Fine-tune both compatible models
    # print("\n=== Fine-tuning Models ===")
    
    # # IMPORTANT: For 3-class classification (-1, 0, 1), use Roberta or Tabularisai
    # # SaBert only supports binary classification (2 classes)
    
    
    for model_name in MODEL_CONFIGS.keys():
        print(f"\n--- Fine-tuning {model_name} ---")
        
        # Load model suitable for n-class classification
        model, tokenizer = get_model_and_tokenizer(model_name, num_labels=MODEL_CONFIGS[model_name]['num_labels'])
        
        # Recreate data_loader with new tokenizer
        data_loader = DataLoader(tokenizer, test_size=0.2, random_state=42)
        
        # Load raw texts and labels from CSV
        texts, labels = data_loader.load_from_csv("JoinedData/FULLDATA.csv", "fullText", "label", label_mapper=sentiment_label_mapper)
        print(f"Loaded {len(texts)} texts with {len(set(labels))} unique labels: {sorted(set(labels))}")
        
        # Create train and validation datasets
        train_dataset, val_dataset = data_loader.create_datasets(texts, labels)
        
        # Fine-tune the model
        fine_tuner = FineTuner(model, tokenizer)
        fine_tune_results = fine_tuner.fine_tune(
            train_dataset, 
            eval_dataset=val_dataset,
            num_train_epochs=3,
            model_name=f"{model_name}_finetuned"
        )
        print("Output directory:", fine_tuner.output_dir)
        
        # Load the fine-tuned model for evaluation
        from transformers import AutoTokenizer, AutoModelForSequenceClassification
        eval_model = AutoModelForSequenceClassification.from_pretrained(fine_tune_results['output_dir'])
        eval_tokenizer = AutoTokenizer.from_pretrained(fine_tune_results['output_dir'])
        
        # Create predictor for evaluation
        eval_predictor = Predictor(eval_model, eval_tokenizer, model_name=model_name)
        
        # Comprehensive evaluation
        print(f"\n=== Evaluating {model_name} Fine-tuned Model ===")
        evaluator = ModelEvaluator(eval_predictor)
             
        # Evaluate the model
        eval_results = evaluator.evaluate_dataset(val_dataset.texts, val_dataset.labels, batch_size=16, normalize_labels=True)
        metrics = eval_results['metrics']
        
        # Print detailed metrics
        print(f"\n=== {model_name} Performance Metrics ===")
        print(f"Accuracy: {metrics['accuracy']:.4f}")
        print(f"Precision (weighted): {metrics['precision']:.4f}")
        print(f"Recall (weighted): {metrics['recall']:.4f}")
        print(f"F1-Score (weighted): {metrics['f1_score']:.4f}")
        
        # Per-class metrics
        print(f"\n=== {model_name} Per-Class Metrics ===")
        class_names = ['Negative', 'Neutral', 'Positive']
        per_class = metrics['per_class_metrics']
        for i, class_name in enumerate(class_names):
            if i < len(per_class['precision']):
                print(f"{class_name}:")
                print(f"  Precision: {per_class['precision'][i]:.4f}")
                print(f"  Recall: {per_class['recall'][i]:.4f}")
                print(f"  F1-Score: {per_class['f1_score'][i]:.4f}")
                print(f"  Support: {per_class['support'][i]}")
        
        # Generate and save ROC curve
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        roc_filename = f"roc_curve_{model_name}_finetuned_{timestamp}.png"
        try:
            roc_auc_scores = evaluator.plot_roc_curve(
                val_dataset.labels, val_dataset.texts, 
                title=f"{model_name} Fine-tuned Model",
                save_path=roc_filename
            )
            print(f"\n=== {model_name} AUC Scores ===")
            for key, value in roc_auc_scores.items():
                print(f"{key}: {value:.4f}")
        except Exception as e:
            print(f"Warning: Could not generate ROC curve: {e}")
        
        # Plot confusion matrix
        try:
            evaluator.plot_confusion_matrix(
                val_dataset.labels, eval_results['predictions'], 
                title=f"Confusion Matrix - {model_name} Fine-tuned"
            )
        except Exception as e:
            print(f"Warning: Could not generate confusion matrix: {e}")
        
        print(f"\n{model_name} fine-tuning and evaluation completed!")
        print(f"Training Results: {fine_tune_results}")

    print("\n=== All models fine-tuned and saved ===")
    
    # Evaluate the latest model with complete metrics and ROC curve
    # evaluate_latest_model_complete()
    

if __name__ == "__main__":
    main()