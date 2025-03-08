# ReAG - Retrieval Augmented Generation

A question-answering application that uses OpenAI's language models to extract relevant information from PDF documents and generate concise answers to user queries.

## Project Overview

ReAG (Retrieval Augmented Generation) is designed to analyze PDF documents and answer specific questions using OpenAI's language models. It employs a two-step process:

1. **Relevancy Assessment**: The application analyzes each page of a document to determine if it contains information relevant to the user's question.
2. **Answer Generation**: Using only the relevant sections, it generates concise, accurate answers limited to three sentences.

## Features

- PDF document loading and processing
- Parallel processing of document pages for improved performance
- Relevancy assessment using OpenAI's models
- Concise answer generation based on relevant context
- Support for local PDF files and remote URLs

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/ReAG.git
cd ReAG
```

2. Install required packages:
```bash
pip install -r requirements.txt
```

3. Create a `.env` file in the project root directory and add your OpenAI API key:
```
OPENAI_API_KEY=your_openai_api_key_here
```

## Usage

Run the application using:
```bash
python app.py
```

The default example processes a PDF about Fibromyalgia and answers two questions:
1. "What is Fibromyalgia?"
2. "What are the causes of Fibromyalgia?"

### Customizing for Your Own Questions

To use the application with your own questions or documents, modify the main execution block in `app.py`:

```python
if __name__ == "__main__":
    # Load your document
    file_path = "path/to/your/document.pdf"  # Can be a local file or URL
    loader = PyMuPDFLoader(file_path)
    docs = loader.load()
    
    # Ask your question
    question = "Your question here?"
    context = extract_relevant_context(question, docs)
    answer = generate_response(question, context)
    
    # Print the result
    print("\nQuestion:", question)
    print("Answer:", answer)
```

## Requirements

- Python 3.8 or higher
- OpenAI API key
- Required packages listed in requirements.txt:
  - langchain
  - langchain_openai
  - langchain_community
  - pymupdf
  - pypdf
  - pydantic
  - python-dotenv
  - openai

## Project Structure

```
ReAG/
│
├── app.py              # Main application code
├── requirements.txt    # Required Python packages
├── .env               # Environment variables (create this file)
├── .gitignore         # Git ignore file
├── README.md          # This documentation
└── data/
    └── fibromyalgia.pdf  # Example PDF document
```

## How It Works

1. **Document Loading**: PDF documents are loaded and split into pages.
2. **Relevancy Analysis**: Each page is analyzed to determine if it contains information relevant to the user's question.
3. **Context Extraction**: Relevant content is extracted and compiled.
4. **Answer Generation**: A concise answer is generated using the extracted context.

## License

[MIT License](LICENSE)

## Acknowledgements

This project uses the following libraries:
- LangChain for the RAG implementation
- OpenAI for language models
- PyMuPDF and PyPDF for PDF processing