```markdown
# A Question-Answering App

A lean project built with Hugging Face Transformers and PyTorch, designed to answer questions from a given text context. uses on devide NLP.

## Tech Stack

- **Python 3.11**
- **PyTorch** (with MPS for Apple Silicon)
- **Hugging Face Transformers**

## Quick Start

1. **Clone & Setup Environment**
   ```bash
   git clone <repository-url>
   cd huggingface_qa_project
   python3 -m venv venv
   source venv/bin/activate
   ```

2. **Install Dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the App**
   ```bash
   python src/qa_app.py
   ```
   Expect a simple output like:
   ```
   Question: What is AI?
   Answer: artificial intelligence
   ```

## Project Structure

```
QA-Indie/
├── data/               # Context files & text samples
├── src/                # Main app & utility scripts
│   ├── qa_app.py       # QA pipeline example
│   └── utils.py        # Helper functions
├── venv/               # Virtual environment (ignore)
├── requirements.txt    # Frozen dependency versions
└── README.md           # This file
```