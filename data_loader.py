import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer
from typing import List, Tuple, Dict, Any
import os
from sklearn.model_selection import train_test_split

class SentimentDataset(Dataset):
    """Custom dataset for sentiment analysis."""
    
    def __init__(self, texts: List[str], labels: List[int], tokenizer, max_length: int = 128):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = str(self.texts[idx])
        label = self.labels[idx]
        
        encoding = self.tokenizer(
            text,
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'
        )
        
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(label, dtype=torch.long)
        }

class DataLoader:
    """Data loading and preprocessing utilities."""
    
    def __init__(self, tokenizer, max_length: int = 128, test_size: float = 0.2, random_state: int = 42):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.test_size = test_size
        self.random_state = random_state
    
    def load_from_csv(self, file_path: str, text_column: str, label_column: str, label_mapper=None) -> Tuple[List[str], List[int]]:
        """Load data from CSV file."""
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File {file_path} not found")
        
        df = pd.read_csv(file_path)
        texts = df[text_column].tolist()
        raw_labels = df[label_column].tolist()
        
        # Use the label mapper if provided to convert labels
        if label_mapper is not None:
            labels = [label_mapper(label) for label in raw_labels]
        else:
            # Assume labels are already in the correct format
            labels = raw_labels
        
        return texts, labels
    
    def load_from_lists(self, texts: List[str], labels: List[int]) -> Tuple[List[str], List[int]]:
        """Load data from lists."""
        return texts, labels
    
    def create_datasets(self, texts: List[str], labels: List[int], 
                       split_data: bool = True) -> Tuple[SentimentDataset, SentimentDataset]:
        """Create train and validation datasets."""
        if split_data:
            train_texts, val_texts, train_labels, val_labels = train_test_split(
                texts, labels, test_size=self.test_size, random_state=self.random_state, stratify=labels
            )
        else:
            # Basic split if split_data=False
            split_idx = int(len(texts) * (1 - self.test_size))
            train_texts, train_labels = texts[:split_idx], labels[:split_idx]
            val_texts, val_labels = texts[split_idx:], labels[split_idx:]
        
        train_dataset = SentimentDataset(train_texts, train_labels, self.tokenizer, self.max_length)
        val_dataset = SentimentDataset(val_texts, val_labels, self.tokenizer, self.max_length)
        
        return train_dataset, val_dataset
    
    def create_single_dataset(self, texts: List[str], labels: List[int]) -> SentimentDataset:
        """Create a single dataset without splitting."""
        return SentimentDataset(texts, labels, self.tokenizer, self.max_length)
    
    def create_dataloader(self, dataset: SentimentDataset, batch_size: int = 16, shuffle: bool = True) -> torch.utils.data.DataLoader:
        """Create a PyTorch DataLoader."""
        return torch.utils.data.DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=0  # Set to 0 for Windows compatibility
        )

def create_sample_data() -> Tuple[List[str], List[int]]:
    """Create sample data for testing purposes."""
    texts = [
        "Me encanta este producto, es fantástico",
        "Odio este servicio, es terrible",
        "El producto está bien, nada especial",
        "Excelente calidad y buen precio",
        "Muy malo, no lo recomiendo",
        "Perfecto para mis necesidades",
        "No funciona como esperaba",
        "Buena relación calidad-precio"
    ]
    labels = [1, 0, 1, 1, 0, 1, 0, 1]  # 1: positive, 0: negative
    
    return texts, labels