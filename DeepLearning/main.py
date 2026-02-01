

from models import get_model_and_tokenizer, get_available_models, load_fine_tuned_model
from data_loader import DataLoader, create_sample_data
from fine_tuner import FineTuner
from predictor import Predictor, create_predictor_from_model_name
from evaluator import ModelEvaluator
import os

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
    
    models_to_finetune = ["Roberta", "Tabularisai"]
    
    for model_name in models_to_finetune:
        print(f"\n--- Fine-tuning {model_name} ---")
        
        # Load model suitable for 3-class classification
        model, tokenizer = get_model_and_tokenizer(model_name, num_labels=3)
        
        # Recreate data_loader with new tokenizer
        data_loader = DataLoader(tokenizer)
        
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
        
        print(f"{model_name} fine-tuning completed!")
        print(f"Results: {fine_tune_results}")

    print( "\=== All models fine-tuned and saved ===")
    

if __name__ == "__main__":
    main()