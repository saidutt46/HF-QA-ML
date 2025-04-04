import logging
from typing import Dict, List, Any, Optional, Union
import time
from improved_utils import chunk_text_by_sentences, rank_answers

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class AdvancedQA:
    """Advanced question answering with multiple strategies for improved results."""
    
    def __init__(self, model_manager):
        """Initialize with a model manager instance."""
        self.model_manager = model_manager
    
    def process_question(
        self, 
        question: str, 
        context: str, 
        model_name: str = "distilbert-base-uncased-distilled-squad",
        strategy: str = "auto"
    ) -> Dict[str, Any]:
        """
        Process a question with advanced strategies.
        
        Args:
            question: The question to answer
            context: The context to search for answers
            model_name: Name of the model to use
            strategy: Strategy to use ('auto', 'direct', 'chunked', 'ensemble')
            
        Returns:
            Dictionary with answer and metadata
        """
        start_time = time.time()
        
        # Determine best strategy if set to auto
        if strategy == "auto":
            strategy = self._determine_strategy(question, context)
        
        # Apply the selected strategy
        if strategy == "direct":
            result = self._direct_qa(question, context, model_name)
        elif strategy == "chunked":
            result = self._chunked_qa(question, context, model_name)
        elif strategy == "ensemble":
            result = self._ensemble_qa(question, context)
        else:
            logger.warning(f"Unknown strategy '{strategy}', falling back to direct")
            result = self._direct_qa(question, context, model_name)
        
        # Add processing metadata
        if result:
            result["processing_time"] = time.time() - start_time
            result["strategy_used"] = strategy
            result["model_used"] = model_name
        
        return result
    
    def _determine_strategy(self, question: str, context: str) -> str:
        """
        Automatically determine the best strategy based on question and context.
        
        Args:
            question: The question to answer
            context: The context text
            
        Returns:
            Strategy name as string
        """
        context_length = len(context.split())
        
        # For very short contexts, direct approach is fastest
        if context_length < 200:
            return "direct"
        
        # For medium contexts, use chunking
        elif context_length < 1000:
            return "chunked"
        
        # For long contexts, use ensemble approach
        else:
            return "ensemble"
    
    def _direct_qa(
        self, 
        question: str, 
        context: str, 
        model_name: str
    ) -> Dict[str, Any]:
        """
        Direct question answering without chunking.
        
        Args:
            question: The question to answer
            context: The context text
            model_name: Model to use
            
        Returns:
            Answer dictionary
        """
        logger.info(f"Using direct QA approach with model {model_name}")
        
        try:
            qa_pipeline = self.model_manager.get_pipeline(model_name)
            result = qa_pipeline(question=question, context=context)
            return result
        except Exception as e:
            logger.error(f"Error in direct QA: {e}")
            return None
    
    def _chunked_qa(
        self, 
        question: str, 
        context: str, 
        model_name: str,
        max_words: int = 300,
        overlap: int = 50
    ) -> Dict[str, Any]:
        """
        Chunked question answering for longer contexts.
        
        Args:
            question: The question to answer
            context: The context text
            model_name: Model to use
            max_words: Maximum words per chunk
            overlap: Word overlap between chunks
            
        Returns:
            Best answer dictionary
        """
        logger.info(f"Using chunked QA approach with model {model_name}")
        
        try:
            # Split text into chunks by sentence boundaries with overlap
            chunks = chunk_text_by_sentences(context, max_words, overlap)
            logger.info(f"Split context into {len(chunks)} chunks")
            
            qa_pipeline = self.model_manager.get_pipeline(model_name)
            
            # Process each chunk
            all_results = []
            for idx, chunk in enumerate(chunks):
                chunk_result = qa_pipeline(question=question, context=chunk)
                chunk_result["chunk_index"] = idx
                all_results.append(chunk_result)
            
            # Find best result
            ranked_results = rank_answers(all_results)
            best_result = ranked_results[0] if ranked_results else None
            
            return best_result
        except Exception as e:
            logger.error(f"Error in chunked QA: {e}")
            return None
    
    def _ensemble_qa(
        self, 
        question: str, 
        context: str
    ) -> Dict[str, Any]:
        """
        Ensemble approach using multiple models and strategies.
        
        Args:
            question: The question to answer
            context: The context text
            
        Returns:
            Best answer dictionary with confidence score
        """
        logger.info("Using ensemble QA approach with multiple models")
        
        try:
            # Use different models
            model_results = []
            
            # Small, fast model with chunking for initial pass
            electra_result = self._chunked_qa(
                question, 
                context, 
                "google/electra-small-discriminator"
            )
            if electra_result:
                model_results.append(electra_result)
            
            # More accurate model for verification
            roberta_result = self._chunked_qa(
                question,
                context,
                "deepset/roberta-base-squad2"
            )
            if roberta_result:
                model_results.append(roberta_result)
            
            # If results available, select best answer
            if model_results:
                # Rank and get best answer
                ranked_results = rank_answers(model_results)
                best_result = ranked_results[0]
                
                # Include alternate answers
                best_result["alternate_answers"] = [
                    result["answer"] for result in ranked_results[1:] 
                    if result["answer"] != best_result["answer"]
                ]
                
                return best_result
            else:
                # Fallback to direct approach with DistilBERT
                return self._direct_qa(
                    question,
                    context,
                    "distilbert-base-uncased-distilled-squad"
                )
        except Exception as e:
            logger.error(f"Error in ensemble QA: {e}")
            return None