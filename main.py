

from models import get_model_and_tokenizer, get_available_models, load_fine_tuned_model
from data_loader import DataLoader, create_sample_data
from fine_tuner import FineTuner
from predictor import Predictor, create_predictor_from_model_name
from evaluator import ModelEvaluator
import os

# Define label mapping function for sentiment analysis
def sentiment_label_mapper(label):
    """Map sentiment labels to integers: N=0 (negative), NEU=1 (neutral), P=2 (positive)"""
    label_mapping = {'N': 0, 'NEU': 1, 'P': 2}
    if label in label_mapping:
        return label_mapping[label]
    else:
        print(f"Warning: Unknown label '{label}', assigning as neutral (1)")
        return 1  # Default to neutral



def main():
    print("=== Sentiment Analysis Framework ===\n")
    
    # 1. Show available models
    print("Available models:", get_available_models())
    
    # 2. Load sample data (replace with your own data loading logic)
    texts, labels = create_sample_data()
    print(f"Loaded {len(texts)} sample texts with {len(set(labels))} classes")
    
    # 3. Example: Test a pre-trained model
    print("\n=== Testing Pre-trained Model ===")
    model_name = "Tabularisai"  # Change to any available model
    model, tokenizer = get_model_and_tokenizer(model_name)
    
    # Create data loader and predictor
    data_loader = DataLoader(tokenizer)
    predictor = Predictor(model, tokenizer, model_name=model_name)
    
    # Test single prediction
    test_text = "Un agente ICE confirma a una vecina que le graba legalmente que la va a incluir en su base de datos como terrorista. Viva la libertad carajo!"
    result = predictor.predict_single(test_text, threshold=0.3)
    print(f"Text: {test_text}")
    print(f"Prediction: {result}")
    
    # # 4. Example: Evaluate model
    # print("\n=== Model Evaluation ===")
    # evaluator = ModelEvaluator(predictor)
    # eval_results = evaluator.evaluate_dataset(texts, labels)
    
    # #5. Example: Fine-tune a model (uncomment to use)
    # print("\n=== Fine-tuning Model ===")
    
    # # Load raw texts and labels from CSV
    # texts, labels = data_loader.load_from_csv("Data/ia_tweets.csv", "text", "polarity", sentiment_label_mapper)
    # # Create train and validation datasets
    # train_dataset, val_dataset = data_loader.create_datasets(texts, labels)
    
    # fine_tuner = FineTuner(model, tokenizer)
    # fine_tune_results = fine_tuner.fine_tune(
    #     train_dataset, 
    #     eval_dataset=val_dataset,
    #     num_train_epochs=2,
    #     model_name="SaBert_finetuned"
    # )
    # print(f"Fine-tuning results: {fine_tune_results}")
    

if __name__ == "__main__":
    main()