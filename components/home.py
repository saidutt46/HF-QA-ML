import streamlit as st
import time
from .utils import process_context, display_results

def render_home(model_name, qa_pipeline):
    # Question input with modern styling
    st.markdown('<div class="search-container">', unsafe_allow_html=True)
    question = st.text_input(
        "Ask a question", 
        placeholder="Type your question here...",
        label_visibility="collapsed"
    )
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Main content
    col1, col2 = st.columns([3, 2])
    
    with col1:
        context = process_context()
        
        # Process button
        process_button = st.button("Find Answer", type="primary", use_container_width=False)
    
    with col2:
        # Results display
        if question and context and process_button:
            # Store question in session state
            st.session_state.current_question = question
            
            # Show loading spinner
            with st.spinner("Finding your answer..."):
                # Create placeholder for animated loading
                results_placeholder = st.empty()
                
                # Process the question
                result = qa_pipeline(question, context)
                
                # Display results
                if result:
                    display_results(results_placeholder, result, context, model_name)
                else:
                    results_placeholder.error("No answer found. Try reformulating your question.")
        else:
            # Initial state or when no question/context
            with st.container():
                st.subheader("How it works")
                st.markdown("1. Provide text in the context area")
                st.markdown("2. Ask a specific question about the text")
                st.markdown("3. Get precise answers with confidence scores")
                st.markdown("4. Try different models for better results")
                st.caption("*Need help? Check the Help section in the sidebar.*") 