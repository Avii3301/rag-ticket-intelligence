import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_community.chat_models import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

from src.retriever import retrieve
from src.reranker import rerank

# Load environment variables from a .env file in the project root.
# This is how we keep the OpenAI API key out of source code —
# it lives in .env (which is git-ignored) and gets read here at runtime.
load_dotenv()


def _format_context(tickets: list[dict]) -> str:
    """
    Convert a list of reranked ticket dicts into a single formatted string
    that gets injected into the prompt as context.

    The LLM reads this text — so clarity matters more than efficiency here.
    Each ticket is numbered so the LLM can refer to them ("as seen in ticket 2...").
    """
    lines = []
    for i, t in enumerate(tickets, start=1):
        lines.append(
            f"Ticket {i}:\n"
            f"  Category: {t['category']}\n"
            f"  Priority: {t['priority']}\n"
            f"  Subject: {t['subject']}\n"
            f"  Resolution: {t['resolution']}"
        )
    # Join tickets with a blank line between them for readability
    return "\n\n".join(lines)


# Build the prompt template once at module load time — not inside the function.
# ChatPromptTemplate.from_messages() is cheap to call, but there's no reason
# to rebuild the same template object on every request.
#
# The prompt uses two variables:
#   {context}   — the formatted string of retrieved tickets
#   {query}     — the new ticket description we want a resolution for
#
# System message: sets the LLM's role and tells it exactly what to output.
# Human message: provides the actual input for this specific request.
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
    """
    Build and return the RAG chain as a LangChain pipeline.

    provider="ollama"  — runs locally via Ollama (free, private, no API key needed).
                         Ollama must be running: `ollama serve`
                         Model must be pulled: `ollama pull mistral`

    provider="openai"  — calls OpenAI's API (costs money, needs OPENAI_API_KEY in .env).
                         Set model_name="gpt-4o-mini" when using this provider.

    temperature=0.2 keeps the output focused and consistent — we want
    reliable resolution suggestions, not creative writing. 0.0 would be
    fully deterministic; 0.2 allows slight variation while staying grounded.

    The chain is three steps piped together with the | operator:
        prompt → LLM → output parser
    The | operator is LangChain's way of composing steps. Each step's output
    becomes the next step's input automatically.
    """
    if provider == "ollama":
        # ChatOllama talks to the local Ollama server at localhost:11434.
        # No API key needed — it's running on your machine.
        llm = ChatOllama(model=model_name, temperature=temperature)
    elif provider == "openai":
        llm = ChatOpenAI(
            model=model_name,
            temperature=temperature,
            api_key=os.getenv("OPENAI_API_KEY"),
        )
    else:
        raise ValueError(f"Unknown provider '{provider}'. Use 'ollama' or 'openai'.")

    # StrOutputParser extracts the plain text string from the LLM's response
    # object. Without it, the chain returns a LangChain AIMessage object —
    # with it, you get a plain Python string you can print or store directly.
    chain = _PROMPT | llm | StrOutputParser()
    return chain


def run(
    query: str,
    retrieve_top_k: int = 20,
    rerank_top_k: int = 5,
    provider: str = "ollama",
    model_name: str = "mistral",
) -> dict:
    """
    Run the full RAG pipeline for a single query.

    Steps:
        1. Retrieve top retrieve_top_k candidates from ChromaDB
        2. Rerank → keep top rerank_top_k
        3. Format context
        4. Call LLM
        5. Return result + the source tickets used (for transparency)

    Returns a dict with:
        query         — the original input
        response      — the LLM's generated resolution suggestion
        sources       — the reranked tickets that were passed as context
    """
    # Step 1 — vector search
    candidates = retrieve(query, top_k=retrieve_top_k)

    # Step 2 — rerank
    top_tickets = rerank(query, candidates, top_k=rerank_top_k)

    # Step 3 — format context string
    context = _format_context(top_tickets)

    # Step 4 — build chain and invoke
    chain = build_rag_chain(model_name=model_name, provider=provider)
    response = chain.invoke({
        "context": context,
        "query": query,
        "n_context": len(top_tickets),
    })

    # Step 5 — return result with sources so the caller can inspect
    # which tickets were used. This is important for debugging and for
    # the notebook — you want to see "here's what the LLM was shown."
    return {
        "query": query,
        "response": response,
        "sources": top_tickets,
    }
