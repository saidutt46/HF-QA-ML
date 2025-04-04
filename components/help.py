import streamlit as st

def render_help():
    st.markdown("""
    ### Getting Started
    
    Answerly helps you get answers from any text. Here's how to use it:
    
    #### 1. Choose your source text
    
    - **Enter Text**: Paste your own text directly
    - **Upload File**: Upload a TXT document (max 200MB)
    - **Sample Text**: Use our built-in examples to try the system
    
    #### 2. Ask a question
    
    Type a specific question about the text in the question box. The more precise your question, the better the answer will be.
    
    #### 3. Choose a model
    
    Select from different AI models depending on your needs:
    
    - **DistilBERT (Fast)**: Quick responses, good for general questions
    - **RoBERTa (Balanced)**: Good balance of speed and accuracy
    - **BERT Large (Accurate)**: More accurate but slower responses
    - **ELECTRA Small (Lightweight)**: Efficient for simple questions
    
    #### 4. Review your answer
    
    The system will highlight the answer in the text and show you a confidence score.
    
    ### Tips for better results
    
    - Ask clear, specific questions
    - Use proper names when referring to people or places
    - Rephrase your question if you don't get a satisfactory answer
    - Try different models for complex questions
    
    ### Limitations
    
    - Questions must be answerable from the provided text
    - Performance varies based on text length and complexity
    - Very complex or ambiguous questions may receive lower confidence scores
    """) 