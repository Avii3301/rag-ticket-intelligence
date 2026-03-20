import chromadb
from src.embeddings import get_embedding_model, search

# Default paths — match what embeddings.py used when building the database
CHROMA_PATH = "data/chroma_db"
COLLECTION_NAME = "itsm_tickets"

# Module-level variable to hold the collection after the first load.
# This is a simple cache — ChromaDB doesn't need to re-read the database
# from disk on every call. We load it once and reuse the same object.
_collection = None


def get_collection(
    chroma_path: str = CHROMA_PATH,
    collection_name: str = COLLECTION_NAME,
) -> chromadb.Collection:
    """
    Load and return the ChromaDB collection from disk.

    PersistentClient reads the database that was written by embeddings.py.
    Calling this a second time returns the cached collection — no disk I/O.
    """
    global _collection
    if _collection is None:
        # PersistentClient opens the on-disk database at chroma_path.
        # This is read-only from retriever's perspective — we never call
        # build_collection() here, only query().
        client = chromadb.PersistentClient(path=chroma_path)
        _collection = client.get_collection(collection_name)
    return _collection


def retrieve(
    query: str,
    top_k: int = 20,
    where: dict = None,
) -> list[dict]:
    """
    Given a natural-language query (new ticket description), return the
    top_k most similar resolved tickets from ChromaDB, deduplicated by
    resolution text.

    We fetch top_k * 3 from ChromaDB first, then deduplicate — because
    synthetic (and real) datasets often have many tickets with identical
    resolutions. Without deduplication, the reranker receives 20 copies
    of the same resolution, which wastes context and skews the LLM's output.

    Deduplication keeps the highest-ranked (lowest distance) ticket for
    each unique resolution, then returns the top_k diverse results.

    where is an optional metadata filter, e.g. {"department": "HR"}.
    Pass None to search all tickets.

    Returns a list of dicts, each with:
        ticket_id, subject, category, priority, resolution, distance
    """
    model = get_embedding_model()
    collection = get_collection()

    # Fetch 3x candidates to give deduplication room to work.
    # If top_k=20 and we fetched exactly 20, deduplication might leave us
    # with far fewer than 20 diverse results.
    candidates = search(query, collection, model, top_k=top_k * 3, where=where)

    # Deduplicate by resolution text — keep first occurrence (lowest distance)
    # of each unique resolution. dict.fromkeys() preserves insertion order.
    seen_resolutions = set()
    diverse = []
    for candidate in candidates:
        resolution = candidate["resolution"].strip()
        if resolution not in seen_resolutions:
            seen_resolutions.add(resolution)
            diverse.append(candidate)
        if len(diverse) == top_k:
            break

    return diverse
