import argparse
import logging
from transformers import pipeline
from src.utils import chunk_text

# Set up logging for better visibility during execution.
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def load_qa_pipeline(model_name="distilbert-base-uncased-distilled-squad"):
    """
    Load the Hugging Face question-answering pipeline using the specified model.
    """
    logger.info(f"Loading QA pipeline with model: {model_name}")
    return pipeline("question-answering", model=model_name)

def get_answer(qa_pipeline, question, context):
    """
    Get an answer from the QA pipeline given a question and context.
    
    Returns:
        dict: The result with answer and score.
    """
    try:
        result = qa_pipeline(question=question, context=context)
        return result
    except Exception as e:
        logger.error(f"Error during QA inference: {e}")
        return None

def process_question(qa_pipeline, question, context):
    """
    Process the question by checking if the context needs chunking.
    
    If the context is long (more than 300 words), it splits it into chunks,
    runs the QA pipeline on each, and returns the answer with the highest score.
    
    Returns:
        dict: The best result found.
    """
    words = context.split()
    if len(words) > 300:
        logger.info("Context is long; splitting into chunks...")
        chunks = chunk_text(context, max_words=300)
        best_result = None
        best_score = 0
        for idx, chunk in enumerate(chunks):
            result = get_answer(qa_pipeline, question, chunk)
            if result and result["score"] > best_score:
                best_score = result["score"]
                best_result = result
                best_result["chunk_index"] = idx
        return best_result
    else:
        return get_answer(qa_pipeline, question, context)

def main():
    parser = argparse.ArgumentParser(description="Advanced Question-Answering System")
    parser.add_argument("--context", type=str, help="Path to a text file with context")
    parser.add_argument("--question", type=str, required=True, help="The question to ask")
    parser.add_argument("--model", type=str, default="distilbert-base-uncased-distilled-squad", help="Name of the model to use")
    args = parser.parse_args()

    # Load context either from a file or use a default context.
    if args.context:
        try:
            with open(args.context, "r", encoding="utf-8") as f:
                context = f.read()
        except Exception as e:
            logger.error(f"Error reading context file: {e}")
            return
    else:
        context = ("Artificial intelligence (AI) is intelligence demonstrated by machines, "
                   "as opposed to natural intelligence displayed by humans.")

    qa = load_qa_pipeline(args.model)
    result = process_question(qa, args.question, context)

    if result:
        print(f"Question: {args.question}")
        print(f"Answer: {result['answer']}")
        print(f"Score: {result['score']:.4f}")
        if "chunk_index" in result:
            print(f"(Found in chunk: {result['chunk_index']})")
    else:
        print("No answer found.")

if __name__ == "__main__":
    main()
