import streamlit as st
import os
import sys
from transformers import pipeline
import time
from PIL import Image
import numpy as np

# Add current directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Import components
from components.home import render_home
from components.history import render_history
from components.help import render_help
from components.about import render_about
from src.qa_app import load_qa_pipeline, process_question

# Available models
MODELS = {
    "DistilBERT (Fast)": "distilbert-base-uncased-distilled-squad",
    "RoBERTa (Balanced)": "deepset/roberta-base-squad2",
    "BERT Large (Accurate)": "bert-large-uncased-whole-word-masking-finetuned-squad",
    "ELECTRA Small (Lightweight)": "google/electra-small-discriminator"
}

# Page configuration
st.set_page_config(
    page_title="Answerly",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Load CSS
with open('static/styles.css') as f:
    st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)

# Load fonts
st.markdown("""
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link href="https://fonts.googleapis.com/css2?family=Lato:ital,wght@0,100;0,300;0,400;0,700;0,900;1,100;1,300;1,400;1,700;1,900&display=swap" rel="stylesheet">
""", unsafe_allow_html=True)

# Session state initialization
if 'page' not in st.session_state:
    st.session_state.page = 'home'
if 'history' not in st.session_state:
    st.session_state.history = []
if 'current_context' not in st.session_state:
    st.session_state.current_context = None
if 'current_question' not in st.session_state:
    st.session_state.current_question = ""

# Function to handle navigation
def navigate_to(page):
    st.session_state.page = page
    st.query_params.page = page

# Sidebar navigation
with st.sidebar:
    st.markdown('<div class="logo-text">Answerly</div>', unsafe_allow_html=True)
    
    st.markdown("#### Navigation")
    
    # Home nav item
    if st.sidebar.button("🏠 Home", key="home_btn", use_container_width=True):
        navigate_to('home')
    
    # History nav item
    if st.sidebar.button("📜 History", key="history_btn", use_container_width=True):
        navigate_to('history')
    
    # Help nav item
    if st.sidebar.button("❓ Help", key="help_btn", use_container_width=True):
        navigate_to('help')
    
    # About nav item
    if st.sidebar.button("ℹ️ About", key="about_btn", use_container_width=True):
        navigate_to('about')
    
    st.markdown("---")
    st.caption("© 2025 Answerly")

# Check for query parameters to update page
query_params = st.query_params
if "page" in query_params:
    st.session_state.page = query_params["page"]

# Title with model selector in header
if st.session_state.page in ['home', 'history']:
    col1, col2 = st.columns([2, 1])
    with col1:
        st.title("Ask Questions About Any Text")
    with col2:
        st.markdown('<div class="header-model-select">', unsafe_allow_html=True)
        model_name = st.selectbox(
            "Model:",
            list(MODELS.keys()),
            label_visibility="visible"
        )
        st.markdown('</div>', unsafe_allow_html=True)
else:
    st.title({
        'help': "Help & Documentation",
        'about': "About Answerly"
    }.get(st.session_state.page, "Answerly"))

# Load QA pipeline
qa_pipeline = load_qa_pipeline(MODELS[model_name]) if 'model_name' in locals() else None

# Render appropriate page
if st.session_state.page == 'home':
    render_home(model_name, qa_pipeline)
elif st.session_state.page == 'history':
    render_history()
elif st.session_state.page == 'help':
    render_help()
elif st.session_state.page == 'about':
    render_about()