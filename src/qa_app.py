from transformers import pipeline

# Load QA pipeline
qa = pipeline("question-answering", model="distilbert-base-uncased-distilled-squad")

# Example context and question
context = """Artificial intelligence (AI) is intelligence demonstrated by machines, as opposed to natural intelligence displayed by humans."""
question = "What is AI?"

# Get answer
result = qa(question=question, context=context)

print(f"Question: {question}")
print(f"Answer: {result['answer']}")
