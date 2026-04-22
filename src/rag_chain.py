import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_community.chat_models import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

from src.retriever import retrieve
from src.reranker import rerank
from src.logger import get_logger

log = get_logger(__name__)

load_dotenv()


def _format_context(tickets: list[dict]) -> str:
    lines = []
    for i, t in enumerate(tickets, start=1):
        lines.append(
            f"Ticket {i}:\n"
            f"  Category: {t['category']}\n"
            f"  Priority: {t['priority']}\n"
            f"  Subject: {t['subject']}\n"
            f"  Resolution: {t['resolution']}"
        )
    return "\n\n".join(lines)


_PROMPT = ChatPromptTemplate.from_messages([
    (
        "system",
        """You are an expert IT support analyst. You will be given a new support ticket
and a set of similar resolved tickets from the past.

Your job is to suggest a resolution for the new ticket based on the patterns
you see in the resolved tickets. Be specific and actionable.

Structure your response exactly like this:
Root cause: <one sentence explaining the likely cause>
Suggested resolution: <step-by-step fix>
Confidence: <high / medium / low, with one sentence explaining why>""",
    ),
    (
        "human",
        """Here are {n_context} similar resolved tickets from our history:

{context}

New ticket: {query}

Based on these examples, suggest a resolution for the new ticket.""",
    ),
])


def build_rag_chain(
    model_name: str = "mistral",
    temperature: float = 0.2,
    provider: str = "ollama",
):
    log.info("Building RAG chain | provider=%s  model=%s  temperature=%s", provider, model_name, temperature)
    if provider == "ollama":
        llm = ChatOllama(model=model_name, temperature=temperature)
    elif provider == "openai":
        llm = ChatOpenAI(
            model=model_name,
            temperature=temperature,
            api_key=os.getenv("OPENAI_API_KEY"),
        )
    else:
        raise ValueError(f"Unknown provider '{provider}'. Use 'ollama' or 'openai'.")

    chain = _PROMPT | llm | StrOutputParser()
    log.info("RAG chain built")
    return chain


def run(
    query: str,
    retrieve_top_k: int = 20,
    rerank_top_k: int = 5,
    provider: str = "ollama",
    model_name: str = "mistral",
) -> dict:
    log.info("RAG run | query=%.80r", query)

    candidates = retrieve(query, top_k=retrieve_top_k)
    top_tickets = rerank(query, candidates, top_k=rerank_top_k)
    context = _format_context(top_tickets)

    chain = build_rag_chain(model_name=model_name, provider=provider)
    log.info("Invoking LLM with %d source tickets", len(top_tickets))
    response = chain.invoke({
        "context": context,
        "query": query,
        "n_context": len(top_tickets),
    })
    log.info("LLM response received (%d chars)", len(response))

    return {
        "query": query,
        "response": response,
        "sources": top_tickets,
    }
