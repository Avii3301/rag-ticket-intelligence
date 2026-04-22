import chromadb
from src.embeddings import get_embedding_model, search
from src.logger import get_logger

log = get_logger(__name__)

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
    global _collection
    if _collection is None:
        log.info("Opening ChromaDB at %s (collection=%s)", chroma_path, collection_name)
        client = chromadb.PersistentClient(path=chroma_path)
        _collection = client.get_collection(collection_name)
        log.info("Collection loaded — %d documents indexed", _collection.count())
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
    log.info("retrieve | top_k=%d | query=%.80r", top_k, query)
    model = get_embedding_model()
    collection = get_collection()

    # Fetch 3x candidates to give deduplication room to work.
    # If top_k=20 and we fetched exactly 20, deduplication might leave us
    # with far fewer than 20 diverse results.
    candidates = search(query, collection, model, top_k=top_k * 3, where=where)
    log.info("Vector search returned %d raw candidates", len(candidates))

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

    log.info(
        "Deduplication: %d candidates → %d diverse results (removed %d duplicates)",
        len(candidates), len(diverse), len(candidates) - len(diverse),
    )
    return diverse
