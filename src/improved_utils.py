import re
import nltk
from nltk.tokenize import sent_tokenize
import logging
from typing import List, Dict, Any, Tuple

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Download NLTK data (if not already present)
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt', quiet=True)


def preprocess_text(text: str) -> str:
    """
    Clean and preprocess text for better QA performance.
    
    Args:
        text (str): Raw text input
        
    Returns:
        str: Preprocessed text
    """
    # Convert multiple whitespaces to single space
    text = re.sub(r'\s+', ' ', text)
    
    # Remove special characters that might confuse the model
    text = re.sub(r'[^\w\s.,?!;:()\[\]\"\'`-]', ' ', text)
    
    # Normalize quotes
    text = re.sub(r'[\u201c\u201d\u0022]', '"', text)
    text = re.sub(r'[\u2018\u2019\u0027]', "'", text)
    
    return text.strip()


def chunk_text_by_sentences(text: str, max_words: int = 300, overlap: int = 50) -> List[str]:
    """
    Split text into overlapping chunks based on sentence boundaries.
    This is more sophisticated than simple word-based chunking.
    
    Args:
        text (str): Input text
        max_words (int): Maximum words per chunk
        overlap (int): Number of words to overlap between chunks
        
    Returns:
        List[str]: List of text chunks
    """
    # Preprocess text
    text = preprocess_text(text)
    
    # Tokenize into sentences and words
    sentences = sent_tokenize(text)
    chunks = []
    current_chunk = []
    current_length = 0
    
    for sentence in sentences:
        # Count words in this sentence
        sentence_words = len(sentence.split())
        
        # If adding this sentence exceeds max_words and we already have content,
        # finalize the current chunk and start a new one
        if current_length + sentence_words > max_words and current_chunk:
            chunk_text = " ".join(current_chunk)
            chunks.append(chunk_text)
            
            # Start new chunk with overlap
            overlap_words = " ".join(current_chunk[-overlap:]) if overlap > 0 and len(current_chunk) > overlap else ""
            current_chunk = []
            if overlap_words:
                current_chunk = [overlap_words]
                current_length = len(overlap_words.split())
            else:
                current_length = 0
        
        # Add the sentence to the current chunk
        current_chunk.append(sentence)
        current_length += sentence_words
    
    # Don't forget the last chunk
    if current_chunk:
        chunk_text = " ".join(current_chunk)
        chunks.append(chunk_text)
    
    logger.info(f"Split text into {len(chunks)} chunks with max {max_words} words each")
    return chunks


def get_context_window(text: str, answer: str, window_size: int = 200) -> str:
    """
    Extract a window of text around the answer for better context display.
    
    Args:
        text (str): Full text context
        answer (str): The answer string
        window_size (int): Number of characters before and after answer
        
    Returns:
        str: Text window with highlighted answer
    """
    # Find the answer in the text
    answer_pos = text.find(answer)
    if answer_pos == -1:
        return text
    
    # Calculate window boundaries
    start = max(0, answer_pos - window_size)
    end = min(len(text), answer_pos + len(answer) + window_size)
    
    # Extract window
    window = text[start:end]
    
    # Add ellipsis if we're not at the beginning/end
    if start > 0:
        window = "..." + window
    if end < len(text):
        window = window + "..."
        
    return window


def rank_answers(answers: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Rank a list of potential answers based on multiple factors.
    
    Args:
        answers (List[Dict]): List of answer dictionaries from the QA pipeline
        
    Returns:
        List[Dict]: Ranked list of answers
    """
    # First sort by score
    sorted_answers = sorted(answers, key=lambda x: x.get('score', 0), reverse=True)
    
    # Return top 3 answers
    return sorted_answers[:3]