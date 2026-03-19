# RAG Ticket Intelligence System

AI assistant that ingests historical IT support tickets, classifies them by severity/category, and uses RAG to suggest resolutions based on past similar tickets.

## Stack
- **Embeddings:** HuggingFace `sentence-transformers` (BGE)
- **Vector DB:** ChromaDB
- **RAG Framework:** LangChain
- **LLM:** OpenAI GPT-4o-mini
- **ML Tracking:** MLflow
- **API:** Flask | **UI:** Streamlit

## Project Structure
```
rag-ticket-intelligence/
├── data/
│   ├── raw/          # synthetic ITSM dataset (gitignored)
│   ├── processed/    # cleaned data
│   └── chroma_db/    # vector store (gitignored)
├── notebooks/        # weekly demo notebooks
├── src/              # core modules (embeddings, retriever, pipeline)
├── models/           # saved classifiers
├── api/              # Flask app
├── ui/               # Streamlit app
└── requirements.txt
```

## Setup
```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

## Dataset
10,000 synthetic ITSM tickets generated via `scripts/generate_dataset.py`. Also available on Kaggle.
