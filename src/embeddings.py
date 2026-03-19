import chromadb
import pandas as pd
from sentence_transformers import SentenceTransformer


def load_corpus(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    return df[df["status"].isin(["Closed", "Resolved"])]


_model = None


def get_embedding_model(model_name: str = "BAAI/bge-small-en-v1.5") -> SentenceTransformer:
    global _model
    if _model is None:
        # device="mps" routes tensor operations to the M-series GPU via Apple's Metal framework.
        # The M4's CPU and GPU share unified memory, so there's no data transfer overhead —
        # matrix multiplications (the core of transformer inference) run significantly faster on GPU.
        _model = SentenceTransformer(model_name, device="mps")
    return _model


def embed_texts(texts: list[str], model: SentenceTransformer, batch_size: int = 64) -> list[list[float]]:
    return model.encode(texts, batch_size=batch_size, show_progress_bar=True).tolist()


def build_collection(
    df: pd.DataFrame,
    chroma_path: str,
    collection_name: str,
    model: SentenceTransformer,
) -> chromadb.Collection:
    # Create a persistent ChromaDB client — saves the vector store to disk at chroma_path
    # so embeddings survive between Python sessions (no need to re-embed every run)
    client = chromadb.PersistentClient(path=chroma_path)

    # Delete the collection if it already exists so this function is safe to re-run.
    # Without this, upserting the same ticket_ids twice would cause duplicates.
    existing = [c.name for c in client.list_collections()]
    if collection_name in existing:
        client.delete_collection(collection_name)

    collection = client.create_collection(collection_name)

    # Embed all ticket descriptions — returns a list of 384-dimensional vectors
    embeddings = embed_texts(df["description"].tolist(), model)

    # Build metadata dicts for each ticket — ChromaDB only accepts string values,
    # so satisfaction_rating (float, sometimes NaN) must be cast to str
    metadatas = [
        {
            "ticket_id": row["ticket_id"],
            "category": row["category"],
            "sub_category": row["sub_category"],
            "priority": row["priority"],
            "department": row["department"],
            "status": row["status"],
            "resolution": row["resolution"] if pd.notna(row["resolution"]) else "",
            "subject": row["subject"],
            "satisfaction_rating": str(row["satisfaction_rating"]),
        }
        for _, row in df.iterrows()
    ]

    # Upsert in batches — ChromaDB has a max batch size of 5461
    ids = df["ticket_id"].tolist()
    documents = df["description"].tolist()
    batch_size = 5000
    for i in range(0, len(ids), batch_size):
        collection.upsert(
            ids=ids[i:i + batch_size],
            embeddings=embeddings[i:i + batch_size],
            documents=documents[i:i + batch_size],
            metadatas=metadatas[i:i + batch_size],
        )

    return collection


def search(
    query: str,
    collection: chromadb.Collection,
    model: SentenceTransformer,
    top_k: int = 5,
    where: dict = None,
) -> list[dict]:
    # BGE was trained with this prefix on query strings only — not on documents.
    # Skipping it gives noticeably worse retrieval quality.
    prefixed_query = "Represent this sentence for searching relevant passages: " + query

    # Embed the query into a 384-dimensional vector, same space as our stored tickets
    query_embedding = model.encode(prefixed_query).tolist()

    # Query ChromaDB — finds the top_k tickets whose embeddings are closest to the query embedding
    # where= is an optional metadata filter e.g. {"department": "HR"}
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=top_k,
        where=where,
    )

    # ChromaDB returns parallel lists — ids[0], documents[0], metadatas[0], distances[0]
    # The [0] is because we sent one query; if you batched multiple queries you'd get multiple result sets
    hits = []
    for i in range(len(results["ids"][0])):
        meta = results["metadatas"][0][i]
        hits.append({
            "ticket_id": meta["ticket_id"],
            "subject": meta["subject"],
            "category": meta["category"],
            "priority": meta["priority"],
            "resolution": meta["resolution"],
            "distance": results["distances"][0][i],
        })

    return hits


if __name__ == "__main__":
    import os

    corpus_path = "data/processed/tickets_clean.csv"
    chroma_path = "data/chroma_db"
    collection_name = "itsm_tickets"

    print("Loading corpus...")
    df = load_corpus(corpus_path)
    print(f"  {len(df)} resolved tickets loaded")

    print("Loading model...")
    model = get_embedding_model()

    print("Building ChromaDB collection...")
    collection = build_collection(df, chroma_path, collection_name, model)
    print(f"  Indexed {collection.count()} tickets into '{collection_name}'")
    print(f"  Saved to {os.path.abspath(chroma_path)}")
