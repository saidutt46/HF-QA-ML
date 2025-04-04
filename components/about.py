import streamlit as st

def render_about():
    st.markdown("""
    ### About Answerly
    
    Answerly is an advanced Question Answering (QA) system built on modern natural language processing technology.
    
    #### Technology
    
    Answerly leverages state-of-the-art transformer models from Hugging Face to understand text and answer questions with high accuracy. The system processes text using these key components:
    
    - **Text chunking**: Breaks down long documents into manageable pieces
    - **Question analysis**: Processes questions to find relevant information
    - **Answer extraction**: Identifies the most likely answer from the context
    - **Confidence scoring**: Assigns a reliability score to each answer
    
    #### Features
    
    - Process text of any length
    - Choose from multiple AI models
    - Get highlighted answers in context
    - View confidence scores
    - Store question history
    
    #### Future Development
    
    We're constantly working to improve Answerly with:
    
    - Support for more file formats (PDF, DOCX)
    - Multi-language support
    - Advanced filtering options
    - Batch processing capabilities
    
    #### Contact
    
    For questions or feedback, please contact support@answerly.ai
    """) 