import torch
import os
import logging
from transformers import AutoTokenizer, AutoModelForQuestionAnswering, pipeline
from typing import Dict, Any, Optional, List, Tuple
import time

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ModelManager:
    """Manages loading, caching, and optimizing models for question answering."""
    
    def __init__(self):
        """Initialize the model manager."""
        self.models_cache = {}
        self.tokenizers_cache = {}
        
        # Check for MPS (Metal Performance Shaders) on Apple Silicon
        self.device = self._get_optimal_device()
        logger.info(f"Using device: {self.device}")
        
        # Model configurations
        self.available_models = {
            "distilbert-base-uncased-distilled-squad": {
                "name": "DistilBERT",
                "description": "Lightweight model, good balance of speed and accuracy",
                "size_mb": 265
            },
            "deepset/roberta-base-squad2": {
                "name": "RoBERTa Base",
                "description": "Higher accuracy on SQuAD 2.0 dataset",
                "size_mb": 480
            },
            "bert-large-uncased-whole-word-masking-finetuned-squad": {
                "name": "BERT Large",
                "description": "High accuracy but slower performance",
                "size_mb": 1250
            },
            "google/electra-small-discriminator": {
                "name": "ELECTRA Small",
                "description": "Small and fast model",
                "size_mb": 55
            }
        }
    
    def _get_optimal_device(self) -> str:
        """Determine the best available device for inference."""
        if torch.cuda.is_available():
            return "cuda"
        elif hasattr(torch, 'mps') and torch.backends.mps.is_available():
            # MPS for Apple Silicon (M1/M2/M3)
            return "mps"
        else:
            return "cpu"
    
    def get_model_info(self) -> Dict[str, Dict[str, Any]]:
        """Get information about all available models."""
        return self.available_models
    
    def load_model(self, model_name: str) -> Tuple[Any, Any]:
        """
        Load a model and tokenizer, with caching.
        
        Args:
            model_name: HuggingFace model identifier
            
        Returns:
            Tuple of (model, tokenizer)
        """
        # Check if already loaded
        if model_name in self.models_cache:
            logger.info(f"Using cached model: {model_name}")
            return self.models_cache[model_name], self.tokenizers_cache[model_name]
        
        # Log loading start time
        start_time = time.time()
        logger.info(f"Loading model: {model_name}")
        
        # Load tokenizer
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        
        # Load model with appropriate device settings
        if self.device == "mps":
            # For MPS, load to CPU first then transfer
            model = AutoModelForQuestionAnswering.from_pretrained(model_name)
            model = model.to(self.device)
        else:
            model = AutoModelForQuestionAnswering.from_pretrained(model_name).to(self.device)
        
        # Cache the loaded model and tokenizer
        self.models_cache[model_name] = model
        self.tokenizers_cache[model_name] = tokenizer
        
        # Log load time
        load_time = time.time() - start_time
        logger.info(f"Model loaded in {load_time:.2f} seconds")
        
        return model, tokenizer
    
    def get_pipeline(self, model_name: str) -> Any:
        """
        Create an optimized question-answering pipeline.
        
        Args:
            model_name: HuggingFace model identifier
            
        Returns:
            HuggingFace pipeline for question answering
        """
        model, tokenizer = self.load_model(model_name)
        
        # Create pipeline with loaded model and tokenizer
        qa_pipeline = pipeline(
            "question-answering",
            model=model,
            tokenizer=tokenizer,
            device=0 if self.device == "cuda" else -1 if self.device == "cpu" else self.device
        )
        
        return qa_pipeline
    
    def get_best_model_for_context_size(self, context_size: int) -> str:
        """
        Recommend the best model based on context size.
        
        Args:
            context_size: Length of the context in characters
            
        Returns:
            Model name string
        """
        # Simple heuristic for model selection based on context size
        if context_size > 10000:
            return "google/electra-small-discriminator"  # Fast for large contexts
        elif context_size > 5000:
            return "distilbert-base-uncased-distilled-squad"  # Good balance
        else:
            return "deepset/roberta-base-squad2"  # Better accuracy for smaller contexts
    
    def cleanup(self):
        """Free memory by clearing model cache."""
        self.models_cache.clear()
        self.tokenizers_cache.clear()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()