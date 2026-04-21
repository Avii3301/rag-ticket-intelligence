"""
Flask API — exposes the classifier and RAG chain over HTTP.

Two endpoints:
  POST /classify  →  predicted category + priority + confidence scores
  POST /resolve   →  RAG-generated resolution suggestion + sources

Both accept JSON: {"description": "ticket text here"}
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from flask import Flask, request, jsonify
from src.classifier import load_data, fit_tfidf_lr
from src.rag_chain import run

app = Flask(__name__)

# ── Startup: train classifiers ────────────────────────────────────────────────
# TF-IDF+LR trains in ~1 second. We do it once at startup and keep the models
# in memory for the lifetime of the server. No need to save/load from disk.

print("Training classifiers on startup...")
X_train, X_test, yc_train, yc_test, yp_train, yp_test = load_data()
cat_vec, cat_clf = fit_tfidf_lr(X_train, yc_train)
pri_vec, pri_clf = fit_tfidf_lr(X_train, yp_train)
print("Classifiers ready.")


# ── Endpoints ─────────────────────────────────────────────────────────────────

@app.route("/health", methods=["GET"])
def health():
    """Simple liveness check — useful to verify the server is running."""
    return jsonify({"status": "ok"})


@app.route("/classify", methods=["POST"])
def classify():
    """
    Predict the category and priority of a new ticket.

    Input JSON:  {"description": "My VPN won't connect..."}

    Output JSON: {
        "category":  "Network",
        "priority":  "High",
        "category_confidence": 0.91,   ← probability of the predicted class
        "priority_confidence": 0.78,
        "category_scores": {"Network": 0.91, "Software": 0.05, ...},
        "priority_scores": {"High": 0.78, "Critical": 0.12, ...}
    }

    predict_proba() returns a probability for each class — we use the max
    as the confidence score and return all scores for transparency.
    """
    body = request.get_json()
    if not body or "description" not in body:
        return jsonify({"error": "Missing 'description' in request body"}), 400

    description = body["description"]

    # Vectorise and predict category
    cat_vec_input = cat_vec.transform([description])
    cat_pred = cat_clf.predict(cat_vec_input)[0]
    cat_proba = cat_clf.predict_proba(cat_vec_input)[0]
    cat_scores = dict(zip(cat_clf.classes_, [round(float(p), 3) for p in cat_proba]))

    # Vectorise and predict priority
    pri_vec_input = pri_vec.transform([description])
    pri_pred = pri_clf.predict(pri_vec_input)[0]
    pri_proba = pri_clf.predict_proba(pri_vec_input)[0]
    pri_scores = dict(zip(pri_clf.classes_, [round(float(p), 3) for p in pri_proba]))

    return jsonify({
        "category":             cat_pred,
        "priority":             pri_pred,
        "category_confidence":  round(float(max(cat_proba)), 3),
        "priority_confidence":  round(float(max(pri_proba)), 3),
        "category_scores":      cat_scores,
        "priority_scores":      pri_scores,
    })


@app.route("/resolve", methods=["POST"])
def resolve():
    """
    Run the full RAG chain and return a resolution suggestion.

    Input JSON:  {"description": "My VPN won't connect..."}

    Output JSON: {
        "response": "Root cause: ...\nSuggested resolution: ...",
        "sources": [
            {"subject": "VPN certificate expired", "category": "Network",
             "priority": "Medium", "resolution": "Renewed the client VPN...",
             "rerank_score": -2.1},
            ...
        ]
    }

    This endpoint is slower than /classify — it embeds the query, searches
    ChromaDB, runs the cross-encoder reranker, and calls Mistral via Ollama.
    Expect 10-30 seconds on first call (model load) and 5-15s after warmup.
    """
    body = request.get_json()
    if not body or "description" not in body:
        return jsonify({"error": "Missing 'description' in request body"}), 400

    description = body["description"]

    result = run(description, provider="ollama", model_name="mistral")

    # Strip internal keys not needed by the UI
    sources = [
        {
            "subject":       s.get("subject", ""),
            "category":      s.get("category", ""),
            "priority":      s.get("priority", ""),
            "resolution":    s.get("resolution", ""),
            "rerank_score":  round(float(s.get("rerank_score", 0)), 3),
        }
        for s in result["sources"]
    ]

    return jsonify({
        "response": result["response"],
        "sources":  sources,
    })


if __name__ == "__main__":
    # debug=False in production — True gives auto-reload on code changes
    # but also exposes the Werkzeug debugger which is a security risk.
    app.run(host="0.0.0.0", port=5001, debug=False)
