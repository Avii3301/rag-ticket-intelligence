# Week 2: Embeddings & Vector DB

## Context
Week 1 produced `data/processed/tickets_clean.csv` (9,769 rows). Week 2 builds the embedding pipeline that forms the retrieval backbone of the RAG system. The goal: embed ticket descriptions with a BGE sentence-transformer, load them into a persistent ChromaDB collection, and expose a similarity search function that retrieves top-k resolved tickets plus their resolutions for any new query ticket.

## Deliverables
1. `src/embeddings.py` — the embedding + retrieval module
2. `notebooks/week2_embeddings.ipynb` — demo notebook showing the pipeline end-to-end

---

## Implementation Plan

### 1. `src/embeddings.py`

**Model:** `BAAI/bge-small-en-v1.5` via `sentence_transformers.SentenceTransformer`
- Fits in memory, fast to encode 7,200 tickets, strong retrieval quality for short texts
- BGE requires prepending `"Represent this sentence for searching relevant passages: "` to query strings only (not documents) — important for retrieval quality

**ChromaDB persist path:** `data/chroma_db/`

**Collection name:** `itsm_tickets`

**Documents stored:** `description` field

**Metadata stored per document:**
- `ticket_id`, `category`, `sub_category`, `priority`, `department`, `status`
- `resolution` (the key retrieval payload)
- `subject`, `satisfaction_rating` (cast to string for Chroma compatibility)

**Functions:**

```
load_corpus(path) -> pd.DataFrame
    Loads tickets_clean.csv, filters to status in [Closed, Resolved] (~7,200 rows)

get_embedding_model(model_name) -> SentenceTransformer
    Loads BGE model; cached with a module-level variable

embed_texts(texts, model, batch_size=64) -> list[list[float]]
    Encodes a list of strings; shows tqdm progress bar

build_collection(df, chroma_path, collection_name, model) -> chromadb.Collection
    - Creates persistent ChromaDB client at chroma_path
    - Deletes existing collection if present (idempotent rebuild)
    - Batch-encodes df["description"]
    - Upserts with ids=ticket_id, embeddings=..., documents=description, metadatas=...
    - Returns the collection

search(query, collection, model, top_k=5, where=None) -> list[dict]
    - Prepends BGE query prefix to query string
    - Embeds query
    - Calls collection.query(query_embeddings=..., n_results=top_k, where=where)
    - Returns list of dicts: {ticket_id, subject, category, priority, resolution, distance}

if __name__ == "__main__":
    CLI entrypoint: load corpus → build collection → print summary
```

---

### 2. `notebooks/week2_embeddings.ipynb`

Cells:
1. **Setup** — imports, path config
2. **Load corpus** — call `load_corpus()`, show shape and resolved-ticket stats
3. **Build ChromaDB** — call `build_collection()`, confirm row count in Chroma matches corpus
4. **Example searches** — 3–4 diverse example queries (one per category), show top-3 results with resolution text
5. **Metadata filtering** — demo `where={"department": "HR"}` filter on a query
6. **Commentary** — markdown cells explaining design decisions

---

## Key Design Decisions
- **Embed `description` only** — descriptions are self-contained (~225 chars), consistent quality; subject adds noise
- **Corpus = resolved tickets only** — open tickets have no resolution, useless for RAG retrieval
- **Idempotent build** — delete + recreate collection so the script is safe to re-run
- **BGE query prefix** — only on query strings, not on indexed documents (per BGE spec)
- **`satisfaction_rating` as metadata** — stored for future retrieval weighting, not used yet

## Verification
1. Run `python src/embeddings.py` — should print "Indexed N tickets" (expect ~7,200)
2. Run all cells in `notebooks/week2_embeddings.ipynb` — example searches should return topically relevant tickets
3. Check `data/chroma_db/` was created
4. Verify a department filter query returns only tickets from that department
