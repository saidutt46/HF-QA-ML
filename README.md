# QA System

This project implements a question-answering system using Hugging Face's Transformers library. It supports dynamic model selection and handles long textual contexts by splitting them into manageable chunks. The application uses Python's argparse for configuration and logging for traceability.

## Features

- **Dynamic Model Selection:** Specify the pre-trained model at runtime.
- **Long-Context Handling:** Automatically splits long texts into chunks and aggregates responses based on confidence scores.
- **Command-Line Interface:** Configure the question, context file, and model via command-line arguments.
- **Logging:** Provides detailed execution logs for debugging and analysis.

## Usage

1. **Clone the repository and set up the environment:**

   ```bash
   git clone <repository-url>
   cd QA-System
   python3 -m venv venv
   source venv/bin/activate
   pip install -r requirements.txt
   ```

2. **Run the application:**

   ```bash
   python src/qa_app.py --question "Your question here" --context data/sample_context.txt --model "deepset/roberta-base-squad2"
   ```

   - If the `--context` argument is omitted, a default context is used.
   - The `--model` argument allows switching between different Hugging Face models.

## Project Structure

```
QA-System/
├── data/
│   └── sample_context.txt       # Text file with context for QA.
├── src/
│   ├── qa_app.py                # Main application script.
│   └── utils.py                 # Utility functions (e.g., text chunking).
├── venv/                        # Python virtual environment.
├── requirements.txt             # Frozen dependency list.
└── README.md                    # Project documentation.
```

## Setup and Execution

- **Environment:** Python 3.11 with a virtual environment is recommended.
- **Dependencies:** All required packages are listed in `requirements.txt`. Install them using:
  
  ```bash
  pip install -r requirements.txt
  ```

- **Running the App:** Use the command provided in the Usage section to execute the QA system.

## Testing and Evaluation

This project is designed to serve as a robust foundation for further exploration and enhancement in question-answering applications. The modular design and dynamic configuration support quick iterations and model experimentation.

---