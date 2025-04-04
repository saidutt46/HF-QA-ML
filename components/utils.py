import streamlit as st
import time

def process_context():
    """Process and return the context based on user selection"""
    st.markdown("#### Context Source")
    st.markdown('<div class="stRadio">', unsafe_allow_html=True)
    context_source = st.radio(
        "Select your context source",
        ["Enter Text", "Upload File", "Sample Text"],
        horizontal=True,
        label_visibility="collapsed"
    )
    st.markdown('</div>', unsafe_allow_html=True)
    
    if context_source == "Enter Text":
        return st.text_area(
            "Enter context text",
            height=300,
            placeholder="Paste or type the text you want to ask questions about..."
        )
    elif context_source == "Upload File":
        st.markdown("""
        <div class="uploadfile-container">
            <div class="upload-icon">📄</div>
            <p>Drag and drop file here</p>
            <p style="font-size: 12px; color: #6c757d;">Limit 200MB per file • TXT</p>
        </div>
        """, unsafe_allow_html=True)
        uploaded_file = st.file_uploader("Upload a text file", type=['txt'], label_visibility="collapsed")
        if uploaded_file:
            context = uploaded_file.getvalue().decode("utf-8")
            with st.expander("Context Preview"):
                st.text(context[:500] + "..." if len(context) > 500 else context)
            return context
    else:  # Sample Text
        try:
            with open("data/sample_context.txt", "r") as f:
                context = f.read()
            st.success("Using sample context")
            with st.expander("Context Preview"):
                st.text(context[:500] + "..." if len(context) > 500 else context)
            return context
        except Exception as e:
            st.error(f"Error loading sample: {e}")
            return "Artificial intelligence (AI) is intelligence demonstrated by machines."
    
    return None

def display_results(placeholder, result, context, model_name):
    """Display the QA results in a formatted way"""
    # Get context window around answer
    answer_start = max(0, result.get('start', 0) - 100)
    answer_end = min(len(context), result.get('end', 0) + 100)
    
    if answer_start > 0:
        context_snippet = "..." + context[answer_start:answer_end]
    else:
        context_snippet = context[answer_start:answer_end]
        
    if answer_end < len(context):
        context_snippet += "..."
    
    # Highlight the answer in context
    highlighted_context = context_snippet.replace(
        result['answer'],
        f"<span class='answer-highlight'>{result['answer']}</span>"
    )
    
    # Display model info and confidence
    st.markdown(
        f"""
        <div class="results-header">
            <div><strong>Model:</strong> {model_name}</div>
            <div>
                <strong>Confidence:</strong> 
                <span class="confidence-indicator confidence-{'high' if result['score'] >= 0.8 else 'medium' if result['score'] >= 0.5 else 'low'}"></span>
                {int(result['score']*100)}%
            </div>
            <div><strong>Time:</strong> {result.get('processing_time', 0.5):.1f}s</div>
        </div>
        """,
        unsafe_allow_html=True
    )
    
    # Display answer
    st.subheader("Answer")
    st.markdown(f"<div class='results-answer'>{result['answer']}</div>", unsafe_allow_html=True)
    
    # Display source
    st.subheader("Source")
    st.markdown(
        f"<div class='results-source'>{highlighted_context}</div>",
        unsafe_allow_html=True
    )
    
    # Add to history
    st.session_state.history.append({
        "question": st.session_state.current_question,
        "answer": result['answer'],
        "score": result['score'],
        "model": model_name,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
    }) 