

from models import get_model_and_tokenizer, get_available_models, load_fine_tuned_model
from data_loader import DataLoader, create_sample_data
from fine_tuner import FineTuner
from predictor import Predictor, create_predictor_from_model_name
from evaluator import ModelEvaluator
import os

def main():
    print("=== Sentiment Analysis Framework ===\n")
    
    # 1. Show available models
    print("Available models:", get_available_models())
    
    # 2. Load sample data (replace with your own data loading logic)
    texts, labels = create_sample_data()
    print(f"Loaded {len(texts)} sample texts with {len(set(labels))} classes")
    
    # 3. Example: Test a pre-trained model
    print("\n=== Testing Pre-trained Model ===")
    model_name = "SaBert"
    model, tokenizer = get_model_and_tokenizer(model_name)
    
    # Create data loader and predictor
    data_loader = DataLoader(tokenizer)
    predictor = Predictor(model, tokenizer)
    
    # Test single prediction
    test_text = "Me encanta este producto, es fantástico"
    result = predictor.predict_single(test_text)
    print(f"Text: {test_text}")
    print(f"Prediction: {result}")
    
    # 4. Example: Evaluate model
    print("\n=== Model Evaluation ===")
    evaluator = ModelEvaluator(predictor)
    eval_results = evaluator.evaluate_dataset(texts, labels)
    
    #5. Example: Fine-tune a model (uncomment to use)
    print("\n=== Fine-tuning Model ===")
    train_dataset, val_dataset = data_loader.create_datasets(texts, labels)
    fine_tuner = FineTuner(model, tokenizer)
    fine_tune_results = fine_tuner.fine_tune(
        train_dataset, 
        eval_dataset=val_dataset,
        num_train_epochs=2,
        model_name="moreira_finetuned"
    )
    print(f"Fine-tuning results: {fine_tune_results}")
    

if __name__ == "__main__":
    main()