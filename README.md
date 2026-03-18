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
│   ├── raw/          # downloaded Kaggle CSV (gitignored)
│   └── processed/    # cleaned data
├── notebooks/
│   └── week1_eda.ipynb
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
Download from [Kaggle ITSM Dataset](https://www.kaggle.com/datasets/imrandude/itsm) and place CSV in `data/raw/`.

## Progress
- [x] Week 1 — Setup & Data Exploration
- [ ] Week 2 — Embeddings & Vector DB
- [ ] Week 3 — RAG Pipeline
- [ ] Week 4 — Classification Layer
- [ ] Week 5 — API + UI
- [ ] Week 6 — Evaluation & Depth
- [ ] Week 7 — Polish & Deploy
- [ ] Week 8 — Interview Prep
