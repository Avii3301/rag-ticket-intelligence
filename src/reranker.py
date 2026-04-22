from sentence_transformers import CrossEncoder
from src.logger import get_logger

log = get_logger(__name__)

# The model we're using is trained on MS MARCO — a large dataset of
# (query, passage, relevance_label) triples from Bing search logs.
# "MiniLM-L-6" means 6 transformer layers — small and fast while still
# being accurate enough for our use case.
MODEL_NAME = "cross-encoder/ms-marco-MiniLM-L-6-v2"

# Module-level cache — same pattern as embeddings.py and retriever.py.
# CrossEncoder takes a few seconds to load; we only want to do that once.
_cross_encoder = None


def get_cross_encoder(model_name: str = MODEL_NAME) -> CrossEncoder:
    global _cross_encoder
    if _cross_encoder is None:
        log.info("Loading cross-encoder: %s", model_name)
        _cross_encoder = CrossEncoder(model_name)
        log.info("Cross-encoder ready")
    return _cross_encoder


def rerank(
    query: str,
    candidates: list[dict],
    top_k: int = 5,
    document_key: str = "resolution",
) -> list[dict]:
    """
    Rerank a list of candidate tickets by relevance to the query.

    candidates is the list of dicts returned by retriever.retrieve() —
    each dict has keys: ticket_id, subject, category, priority, resolution, distance.

    document_key controls which field the cross-encoder scores against.
    We use "resolution" — we want to surface tickets whose resolutions are
    most relevant to the new query, not just tickets with similar descriptions.

    Returns the top_k candidates sorted by cross-encoder score (highest first),
    with a "rerank_score" field added to each dict.
    """
    if not candidates:
        log.warning("rerank called with empty candidates list — returning []")
        return []

    log.info("rerank | %d candidates → top_%d | query=%.80r", len(candidates), top_k, query)
    model = get_cross_encoder()

    # The cross-encoder expects a list of [query, document] pairs.
    # It scores each pair independently — for N candidates, we get N scores.
    # We use the resolution text as the document because that's what we want
    # the LLM to use: we're asking "which resolutions are most useful for this query?"
    #
    # Some resolutions might be empty strings (tickets with no resolution text).
    # The cross-encoder handles empty strings fine — they'll just score low.
    pairs = [[query, c[document_key]] for c in candidates]

    # predict() runs the cross-encoder on all pairs in one batch.
    # Returns a numpy array of float scores — higher means more relevant.
    # These are raw logits, not probabilities — the absolute values don't
    # matter, only the relative ordering.
    scores = model.predict(pairs)

    for candidate, score in zip(candidates, scores):
        candidate["rerank_score"] = float(score)

    reranked = sorted(candidates, key=lambda c: c["rerank_score"], reverse=True)
    top = reranked[:top_k]

    if top:
        log.info(
            "Rerank complete | top score=%.3f  bottom score=%.3f  spread=%.3f",
            top[0]["rerank_score"],
            top[-1]["rerank_score"],
            top[0]["rerank_score"] - top[-1]["rerank_score"],
        )
        for rank, src in enumerate(top, 1):
            log.info(
                "  #%d  score=% .3f  category=%-12s  subject=%.60s",
                rank, src["rerank_score"], src["category"], src["subject"],
            )

    return top
