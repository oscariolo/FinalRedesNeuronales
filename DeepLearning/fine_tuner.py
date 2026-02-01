import torch
from transformers import TrainingArguments, Trainer
from typing import Any, Dict, Optional
import os
from datetime import datetime
import torch.nn as nn

class FineTuner:
    """Fine-tuning utilities for sentiment analysis models."""
    
    def __init__(self, model, tokenizer, output_dir: str = "./fine_tuned_models"):
        self.model = model
        self.tokenizer = tokenizer
        self.output_dir = output_dir
        self.trainer = None
        self.class_weights = None
        
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
    
    def setup_training_arguments(self, 
                               num_train_epochs: int = 3,
                               per_device_train_batch_size: int = 16,
                               per_device_eval_batch_size: int = 16,
                               learning_rate: float = 2e-5,
                               warmup_steps: int = 500,
                               weight_decay: float = 0.01,
                               logging_steps: int = 10,
                               save_steps: int = 500,
                               eval_steps: int = 500,
                               evaluation_strategy: str = "steps",
                               save_strategy: str = "steps",
                               load_best_model_at_end: bool = True,
                               metric_for_best_model: str = "eval_loss",
                               model_name: str = None,
                               **kwargs) -> TrainingArguments:
        """Setup training arguments for fine-tuning."""
        
        # Use provided model_name or fallback to 'model'
        if model_name is None:
            model_name = kwargs.get('model_name', 'model')
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        run_name = f"{model_name}_{timestamp}"
        
        return TrainingArguments(
            output_dir=os.path.join(self.output_dir, run_name),
            num_train_epochs=num_train_epochs,
            per_device_train_batch_size=per_device_train_batch_size,
            per_device_eval_batch_size=per_device_eval_batch_size,
            learning_rate=learning_rate,
            warmup_steps=warmup_steps,
            weight_decay=weight_decay,
            logging_steps=logging_steps,
            save_steps=save_steps,
            eval_steps=eval_steps,
            eval_strategy=evaluation_strategy,
            save_strategy=save_strategy,
            load_best_model_at_end=load_best_model_at_end,
            metric_for_best_model=metric_for_best_model,
            greater_is_better=False if metric_for_best_model == "eval_loss" else True,
            dataloader_num_workers=0,  # Windows compatibility
            **kwargs
        )
    
    def fine_tune(self, 
                  train_dataset,
                  eval_dataset=None,
                  training_args: TrainingArguments = None,
                  model_name: str = None,
                  class_weights=None,
                  num_train_epochs: int = 3,
                  **kwargs) -> Dict[str, Any]:
        """Fine-tune the model.
        
        Args:
            train_dataset: Training dataset
            eval_dataset: Evaluation dataset (optional)
            training_args: Custom training arguments (optional)
            model_name: Name for the fine-tuned model (optional)
            class_weights: Weights for each class (optional)
            **kwargs: Additional arguments passed to setup_training_arguments
        """
        
        if training_args is None:
            if model_name is not None:
                kwargs['model_name'] = model_name
            training_args = self.setup_training_arguments(**kwargs)
        
        if class_weights is not None:
            # Accept list/np array/torch tensor
            # Ensure weights tensor matches the number of classes in the model
            self.class_weights = torch.tensor(class_weights, dtype=torch.float)
        else:
            self.class_weights = None

        class WeightedTrainer(Trainer):
            def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
                labels = inputs.get("labels")
                outputs = model(**{k: v for k, v in inputs.items() if k != "labels"})
                logits = outputs.get("logits")
                
                if self.class_weights is not None:
                    weights = self.class_weights.to(logits.device)
                    
                    # Pad weights if model has more classes than training labels
                    num_model_classes = logits.shape[-1]
                    num_weight_classes = weights.shape[0]
                    
                    if num_weight_classes < num_model_classes:
                        # Pad with 1.0 for unused classes
                        padding = torch.ones(num_model_classes - num_weight_classes, device=logits.device)
                        weights = torch.cat([weights, padding])
                    
                    loss_fct = nn.CrossEntropyLoss(weight=weights)
                else:
                    loss_fct = nn.CrossEntropyLoss()
                
                loss = loss_fct(logits, labels)
                return (loss, outputs) if return_outputs else loss

        trainer_class = WeightedTrainer if self.class_weights is not None else Trainer

        self.trainer = trainer_class(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset
        )
        if self.class_weights is not None:
            self.trainer.class_weights = self.class_weights
        
        # Train the model
        print(f"Starting fine-tuning with {len(train_dataset)} training samples...")
        train_result = self.trainer.train()
        
        # Save the model
        self.trainer.save_model()
        self.trainer.save_state()
        
        print(f"Fine-tuning completed. Model saved to: {training_args.output_dir}")
        
        return {
            'train_loss': train_result.training_loss,
            'train_runtime': train_result.metrics.get('train_runtime'),
            'train_samples_per_second': train_result.metrics.get('train_samples_per_second'),
            'output_dir': training_args.output_dir
        }
    
    def evaluate(self, eval_dataset) -> Dict[str, float]:
        """Evaluate the model."""
        if self.trainer is None:
            raise ValueError("Model must be fine-tuned first or trainer must be initialized")
        
        print("Evaluating model...")
        eval_result = self.trainer.evaluate(eval_dataset=eval_dataset)
        print(f"Evaluation results: {eval_result}")
        
        return eval_result
    
    def save_model(self, save_path: str = None, model_name: str = None):
        """Save the model and tokenizer.
        
        Args:
            save_path: Custom save path (optional)
            model_name: Name for the model directory (optional)
        """
        if save_path is None:
            # Generate save path with model name
            name = model_name if model_name else "fine_tuned_model"
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            save_path = os.path.join(self.output_dir, f"{name}_{timestamp}")
        
        os.makedirs(save_path, exist_ok=True)
        self.model.save_pretrained(save_path)
        self.tokenizer.save_pretrained(save_path)
        print(f"Model saved to: {save_path}")
        
        return save_path
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the current model."""
        return {
            'model_type': type(self.model).__name__,
            'tokenizer_type': type(self.tokenizer).__name__,
            'num_parameters': sum(p.numel() for p in self.model.parameters()),
            'trainable_parameters': sum(p.numel() for p in self.model.parameters() if p.requires_grad),
            'device': next(self.model.parameters()).device
        }