"""
MLflow evaluation harness for the RAG Ticket Intelligence pipeline.

Usage
-----
# 1. Start MLflow UI (run once in a separate terminal):
#    mlflow ui --backend-store-uri sqlite:///mlflow.db --port 5050
#    Then open the link printed at the end of the run.

# 2. Run evaluation with default settings (Ollama/mistral):
#    python scripts/evaluate.py

# 3. Override provider / model / top_k via CLI:
#    python scripts/evaluate.py --provider ollama --model mistral --top-k-retrieve 20 --top-k-rerank 5

# 4. Score interactively:
#    python scripts/evaluate.py --score
#    (After generation, you'll be prompted to enter 1-5 quality score per query)

# 5. Compare experiments in MLflow UI by changing top_k or model and re-running.
"""

import sys
import os
import json
import time
import sqlite3
import argparse

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import mlflow
from mlflow.tracking import MlflowClient
from src.logger import get_logger
from src.reranker import MODEL_NAME as RERANKER_MODEL

log = get_logger("evaluate")

EMBEDDING_MODEL = "BAAI/bge-small-en-v1.5"
CHUNK_SIZE = 500

MLFLOW_TRACKING_URI = "sqlite:///mlflow.db"
EXPERIMENT_NAME = "RAG Ticket Intelligence System"

DEFAULT_TOP_K_RETRIEVE = 20
DEFAULT_TOP_K_RERANK = 5
DEFAULT_PROVIDER = "ollama"
DEFAULT_MODEL = "mistral"

TEST_QUERIES = [
    "My laptop won't connect to the office WiFi after the Windows update last night.",
    "Outlook keeps crashing when I try to open attachments larger than 5MB.",
    "I need access to the shared finance drive but get 'Access Denied' every time.",
    "My VPN disconnects every 10 minutes and I can't stay connected during video calls.",
    "The printer on floor 3 is offline and nobody can print — it's been down since morning.",
    "I accidentally deleted a folder from the shared project drive. Can it be recovered?",
    "New employee onboarding — need accounts created for Sarah Johnson starting Monday.",
    "SAP is throwing error code DBIF_RSQL_INVALID_REQUEST when running the month-end report.",
    "My screen resolution resets to 1024x768 every time I restart my computer.",
    "Two-factor authentication is not sending the SMS code to my phone.",
]


# ---------------------------------------------------------------------------
# MLflow experiment bootstrap
# ---------------------------------------------------------------------------

def _setup_experiment() -> str:
    """
    Ensure the experiment exists in SQLite with a local file:// artifact root.

    SQLite-backed MLflow without a running server uses mlflow-artifacts:// by
    default, which requires an HTTP server to resolve. We patch the artifact
    location directly in SQLite so artifacts write to a local directory instead.
    """
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    artifact_root = f"file://{os.path.abspath('mlflow_artifacts')}"
    db_path = MLFLOW_TRACKING_URI.replace("sqlite:///", "")

    with sqlite3.connect(db_path) as conn:
        conn.execute(
            "UPDATE experiments SET artifact_location = ?, lifecycle_stage = 'active' WHERE name = ?",
            (artifact_root, EXPERIMENT_NAME),
        )

    client = MlflowClient()
    existing = client.get_experiment_by_name(EXPERIMENT_NAME)
    if existing is None:
        log.info("Creating MLflow experiment: %s", EXPERIMENT_NAME)
        experiment_id = client.create_experiment(EXPERIMENT_NAME, artifact_location=artifact_root)
    else:
        experiment_id = existing.experiment_id
        log.info("Using existing MLflow experiment: %s (id=%s)", EXPERIMENT_NAME, experiment_id)

    mlflow.set_experiment(experiment_id=experiment_id)
    return experiment_id


# ---------------------------------------------------------------------------
# Pipeline stages
# ---------------------------------------------------------------------------

def _retrieve(query: str, top_k: int) -> tuple[list[dict], float]:
    from src.retriever import retrieve
    t0 = time.perf_counter()
    candidates = retrieve(query, top_k=top_k)
    return candidates, (time.perf_counter() - t0) * 1000


def _rerank(query: str, candidates: list[dict], top_k: int) -> tuple[list[dict], float]:
    from src.reranker import rerank
    t0 = time.perf_counter()
    top_sources = rerank(query, candidates, top_k=top_k)
    return top_sources, (time.perf_counter() - t0) * 1000


def _generate(
    query: str,
    top_sources: list[dict],
    provider: str,
    model_name: str,
) -> tuple[str, str, float]:
    """Return (prompt, response, latency_ms)."""
    context_parts = []
    for i, src in enumerate(top_sources, 1):
        context_parts.append(
            f"[Ticket {i}]\n"
            f"Category: {src.get('category', '')}\n"
            f"Priority: {src.get('priority', '')}\n"
            f"Issue: {src.get('subject', '')}\n"
            f"Resolution: {src.get('resolution', '')}"
        )
    context = "\n\n".join(context_parts)

    prompt = (
        f"You are an IT support specialist. A new ticket has been submitted:\n\n"
        f"TICKET:\n{query}\n\n"
        f"SIMILAR RESOLVED TICKETS (use these as reference):\n{context}\n\n"
        f"Based on the similar resolved tickets above, provide:\n"
        f"1. Root cause (likely reason for this issue)\n"
        f"2. Suggested resolution (step-by-step fix)\n"
        f"3. Escalation needed? (yes/no and why)\n\n"
        f"Be concise and practical."
    )

    log.info("Invoking LLM | provider=%s  model=%s  context_tickets=%d", provider, model_name, len(top_sources))
    t0 = time.perf_counter()

    if provider == "ollama":
        from langchain_community.chat_models import ChatOllama
        from langchain_core.messages import HumanMessage
        response_text = ChatOllama(model=model_name).invoke([HumanMessage(content=prompt)]).content
    elif provider == "openai":
        from langchain_openai import ChatOpenAI
        from langchain_core.messages import HumanMessage
        response_text = ChatOpenAI(model=model_name, temperature=0).invoke([HumanMessage(content=prompt)]).content
    else:
        raise ValueError(f"Unknown provider: {provider}")

    latency = (time.perf_counter() - t0) * 1000
    log.info("LLM response received | %d chars | %.0fms", len(response_text), latency)
    return prompt, response_text, latency


# ---------------------------------------------------------------------------
# Interactive scoring
# ---------------------------------------------------------------------------

def _prompt_quality_score(query: str, response: str) -> float:
    print("\n" + "─" * 60)
    print(f"QUERY:\n{query}\n")
    print(f"RESPONSE:\n{response}\n")
    while True:
        try:
            score = float(input("Quality score (1=bad, 5=perfect): ").strip())
            if 1.0 <= score <= 5.0:
                return score
            print("  Enter a number between 1 and 5.")
        except ValueError:
            print("  Invalid input.")


# ---------------------------------------------------------------------------
# Main evaluation loop
# ---------------------------------------------------------------------------

def evaluate(
    queries: list[str] = TEST_QUERIES,
    top_k_retrieve: int = DEFAULT_TOP_K_RETRIEVE,
    top_k_rerank: int = DEFAULT_TOP_K_RERANK,
    provider: str = DEFAULT_PROVIDER,
    model_name: str = DEFAULT_MODEL,
    interactive_score: bool = False,
):
    experiment_id = _setup_experiment()

    log.info(
        "Starting evaluation | experiment=%s | provider=%s/%s | top_k=%d→%d | queries=%d",
        EXPERIMENT_NAME, provider, model_name, top_k_retrieve, top_k_rerank, len(queries),
    )

    print(f"\n{'='*60}")
    print(f"  Experiment : {EXPERIMENT_NAME}")
    print(f"  Provider   : {provider} / {model_name}")
    print(f"  top_k      : retrieve={top_k_retrieve}, rerank={top_k_rerank}")
    print(f"  Queries    : {len(queries)}")
    print(f"  MLflow UI  : mlflow ui --backend-store-uri sqlite:///mlflow.db --port 5050")
    print(f"{'='*60}\n")

    total_latencies = []

    with mlflow.start_run(run_name="evaluation"):

        # ── Tags ─────────────────────────────────────────────────────────────
        mlflow.set_tags({
            "provider": provider,
            "model": model_name,
            "num_queries": str(len(queries)),
        })

        # ── Params (logged once for the whole run) ────────────────────────────
        mlflow.log_params({
            "top_k_retrieve": top_k_retrieve,
            "top_k_rerank": top_k_rerank,
            "chunk_size": CHUNK_SIZE,
            "embedding_model": EMBEDDING_MODEL,
            "reranker_model": RERANKER_MODEL,
            "provider": provider,
            "model_name": model_name,
            "num_queries": len(queries),
        })
        for i, q in enumerate(queries, 1):
            mlflow.log_param(f"query_{i:02d}", q[:250])

        for i, query in enumerate(queries, 1):
            log.info("--- Query %d/%d ---", i, len(queries))
            print(f"[{i}/{len(queries)}] {query[:70]}...")

            # ── Per-query span (groups retrieve → rerank → generate) ──────────
            with mlflow.start_span(name=f"query_{i:02d}", span_type="CHAIN") as qspan:
                qspan.set_inputs({"query": query, "index": i})

                # ── Retrieve ────────────────────────────────────────────────
                with mlflow.start_span(name="retrieve", span_type="RETRIEVER") as span:
                    span.set_inputs({"query": query, "top_k": top_k_retrieve})
                    candidates, retrieval_latency = _retrieve(query, top_k_retrieve)
                    span.set_outputs({
                        "count": len(candidates),
                        "candidates": [
                            {
                                "ticket_id": c["ticket_id"],
                                "category": c["category"],
                                "priority": c["priority"],
                                "distance": round(c["distance"], 4),
                                "subject": c["subject"][:80],
                            }
                            for c in candidates
                        ],
                    })
                    span.set_attribute("latency_ms", round(retrieval_latency, 2))

                # ── Rerank ──────────────────────────────────────────────────
                with mlflow.start_span(name="rerank", span_type="RERANKER") as span:
                    span.set_inputs({
                        "query": query,
                        "num_candidates": len(candidates),
                        "top_k": top_k_rerank,
                        "document_key": "resolution",
                    })
                    top_sources, rerank_latency = _rerank(query, candidates, top_k_rerank)
                    top_score = top_sources[0]["rerank_score"] if top_sources else 0.0
                    score_spread = (
                        top_sources[0]["rerank_score"] - top_sources[-1]["rerank_score"]
                        if len(top_sources) > 1 else 0.0
                    )
                    span.set_outputs({
                        "count": len(top_sources),
                        "top_score": round(top_score, 4),
                        "score_spread": round(score_spread, 4),
                        "sources": [
                            {
                                "rank": r,
                                "ticket_id": s["ticket_id"],
                                "category": s["category"],
                                "priority": s["priority"],
                                "rerank_score": round(s["rerank_score"], 4),
                                "subject": s["subject"][:80],
                            }
                            for r, s in enumerate(top_sources, 1)
                        ],
                    })
                    span.set_attribute("latency_ms", round(rerank_latency, 2))

                # ── Generate ────────────────────────────────────────────────
                with mlflow.start_span(name="generate", span_type="LLM") as span:
                    span.set_inputs({
                        "query": query,
                        "model": model_name,
                        "provider": provider,
                        "num_sources": len(top_sources),
                    })
                    prompt, response, generation_latency = _generate(
                        query, top_sources, provider, model_name
                    )
                    span.set_outputs({"response": response})
                    span.set_attribute("latency_ms", round(generation_latency, 2))
                    span.set_attribute("prompt_length_chars", len(prompt))
                    span.set_attribute("response_length_chars", len(response))

                qspan.set_outputs({"response_preview": response[:300]})

            # ── Metrics logged with step=i → time-series charts in UI ─────────
            total_latency = retrieval_latency + rerank_latency + generation_latency
            total_latencies.append(total_latency)

            mlflow.log_metrics({
                "retrieval_latency_ms": round(retrieval_latency, 2),
                "rerank_latency_ms": round(rerank_latency, 2),
                "generation_latency_ms": round(generation_latency, 2),
                "total_latency_ms": round(total_latency, 2),
                "num_candidates": len(candidates),
                "num_sources": len(top_sources),
                "top_rerank_score": round(top_score, 4),
                "rerank_score_spread": round(score_spread, 4),
                "response_length_chars": len(response),
            }, step=i)

            # ── Quality score ─────────────────────────────────────────────────
            if interactive_score:
                quality = _prompt_quality_score(query, response)
                mlflow.log_metric("manual_quality_score", quality, step=i)

            # ── Artifacts ─────────────────────────────────────────────────────
            mlflow.log_text(response, f"responses/response_{i:02d}.txt")
            mlflow.log_text(prompt, f"prompts/prompt_{i:02d}.txt")

            trace = {"query": query, "sources": top_sources, "response": response}
            trace_path = f"/tmp/rag_trace_{i:02d}.json"
            with open(trace_path, "w") as f:
                json.dump(trace, f, indent=2)
            mlflow.log_artifact(trace_path, artifact_path="traces")

            log.info(
                "Query %d done | retrieve=%.0fms  rerank=%.0fms  generate=%.0fms  total=%.0fms",
                i, retrieval_latency, rerank_latency, generation_latency, total_latency,
            )
            print(
                f"       retrieve={retrieval_latency:.0f}ms  "
                f"rerank={rerank_latency:.0f}ms  "
                f"generate={generation_latency:.0f}ms  "
                f"total={total_latency:.0f}ms"
            )

        # ── Summary metric (average total latency across all queries) ────────
        if total_latencies:
            mlflow.log_metric(
                "avg_total_latency_ms",
                round(sum(total_latencies) / len(total_latencies), 2),
            )

    avg_latency = sum(total_latencies) / len(total_latencies) if total_latencies else 0
    log.info(
        "Evaluation complete | %d queries | avg total latency=%.0fms",
        len(queries), avg_latency,
    )
    print(f"\nDone. Average total latency: {avg_latency:.0f}ms")
    print(f"  Start UI : mlflow ui --backend-store-uri sqlite:///mlflow.db --port 5050")
    print(f"  View runs: http://localhost:5050/#/experiments/{experiment_id}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate RAG pipeline and log to MLflow")
    parser.add_argument("--provider", default=DEFAULT_PROVIDER, help="LLM provider: ollama or openai")
    parser.add_argument("--model", default=DEFAULT_MODEL, dest="model_name", help="Model name")
    parser.add_argument("--top-k-retrieve", default=DEFAULT_TOP_K_RETRIEVE, type=int)
    parser.add_argument("--top-k-rerank", default=DEFAULT_TOP_K_RERANK, type=int)
    parser.add_argument("--score", action="store_true", help="Interactively score each response 1–5")
    parser.add_argument("--queries", nargs="+", help="Custom queries (space-separated strings)")
    args = parser.parse_args()

    evaluate(
        queries=args.queries if args.queries else TEST_QUERIES,
        top_k_retrieve=args.top_k_retrieve,
        top_k_rerank=args.top_k_rerank,
        provider=args.provider,
        model_name=args.model_name,
        interactive_score=args.score,
    )
