def chunk_text(text, max_words=300):
    """
    Split the text into chunks of approximately max_words.
    
    Args:
        text (str): The input context.
        max_words (int): Maximum number of words per chunk.
    
    Returns:
        List[str]: A list of text chunks.
    """
    words = text.split()
    chunks = []
    for i in range(0, len(words), max_words):
        chunk = " ".join(words[i:i + max_words])
        chunks.append(chunk)
    return chunks
