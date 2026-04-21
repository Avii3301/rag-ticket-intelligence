import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, f1_score, classification_report
from transformers import DistilBertTokenizerFast, DistilBertForSequenceClassification
from transformers import get_linear_schedule_with_warmup
import mlflow
import mlflow.sklearn

# ── Data ──────────────────────────────────────────────────────────────────────

def load_data(path: str = "data/processed/tickets_clean.csv"):
    """
    Load the cleaned ticket dataset and return train/test splits.

    We use the description field as X (input text) and predict
    category and priority separately — two independent classifiers.

    test_size=0.2 means 80% of rows go to training, 20% held out for eval.
    random_state=42 makes the split reproducible — every run gets the same
    train/test rows, so you can compare experiments fairly.
    """
    df = pd.read_csv(path)

    X = df["description"].astype(str).tolist()
    y_category = df["category"].tolist()
    y_priority = df["priority"].tolist()

    X_train, X_test, yc_train, yc_test, yp_train, yp_test = train_test_split(
        X, y_category, y_priority,
        test_size=0.2,
        random_state=42,
    )

    return X_train, X_test, yc_train, yc_test, yp_train, yp_test


# ── Baseline: TF-IDF + Logistic Regression ────────────────────────────────────

def train_tfidf_lr(
    X_train: list[str],
    X_test: list[str],
    y_train: list[str],
    y_test: list[str],
    target: str,
    max_features: int = 10_000,
    C: float = 1.0,
):
    """
    Train a TF-IDF + Logistic Regression classifier and log results to MLflow.

    Parameters
    ----------
    target      : "category" or "priority" — used to name the MLflow run
    max_features: how many unique words (tokens) TF-IDF keeps.
                  10,000 is a reasonable default — covers the vocabulary
                  well without creating a massive sparse matrix.
    C           : regularisation strength for Logistic Regression.
                  Lower C = stronger regularisation (penalises large weights
                  harder, pushes the model towards simplicity).
                  Higher C = less regularisation (model fits training data
                  more closely, risk of overfitting).
                  C=1.0 is the sklearn default and a safe starting point.

    MLflow tracks this run so you can compare it against DistilBERT later.
    """

    with mlflow.start_run(run_name=f"tfidf_lr_{target}"):

        # ── Log hyperparameters ───────────────────────────────────────────
        # "Parameters" in MLflow = things you set before training.
        # These show up as columns in the MLflow UI so you can compare runs.
        mlflow.log_param("model", "TF-IDF + LogisticRegression")
        mlflow.log_param("target", target)
        mlflow.log_param("max_features", max_features)
        mlflow.log_param("C", C)

        # ── TF-IDF vectoriser ─────────────────────────────────────────────
        # fit_transform on train: learns the vocabulary from training data
        # and converts each ticket description to a sparse vector.
        # transform on test: applies the SAME vocabulary (no data leakage —
        # the model must not see test vocabulary during training).
        #
        # ngram_range=(1, 2) means we keep single words AND two-word phrases.
        # "VPN" is informative, but "VPN certificate" is even more specific.
        vectoriser = TfidfVectorizer(
            max_features=max_features,
            ngram_range=(1, 2),
            sublinear_tf=True,  # apply log(1 + TF) to dampen high-freq terms
        )
        X_train_vec = vectoriser.fit_transform(X_train)
        X_test_vec = vectoriser.transform(X_test)

        # ── Logistic Regression ───────────────────────────────────────────
        # max_iter=1000: the optimiser may need more than the default 100
        # iterations to converge on 8 classes. 1000 is safe.
        # solver="lbfgs": the default solver, works well for multi-class.
        clf = LogisticRegression(C=C, max_iter=1000, solver="lbfgs")
        clf.fit(X_train_vec, y_train)

        # ── Evaluate ──────────────────────────────────────────────────────
        y_pred = clf.predict(X_test_vec)

        accuracy = accuracy_score(y_test, y_pred)

        # weighted F1 accounts for class imbalance — it weights each class's
        # F1 score by how many samples that class has in the test set.
        # Better than macro F1 when classes aren't balanced (priority is slightly
        # skewed: Medium has 2x more samples than Critical).
        f1 = f1_score(y_test, y_pred, average="weighted")

        report = classification_report(y_test, y_pred)

        # ── Log metrics ───────────────────────────────────────────────────
        # "Metrics" in MLflow = numbers that measure performance.
        mlflow.log_metric("accuracy", accuracy)
        mlflow.log_metric("f1_weighted", f1)

        # Save the full classification report as a text artifact.
        # Artifacts are files attached to a run — visible in the MLflow UI.
        report_path = f"/tmp/report_{target}.txt"
        with open(report_path, "w") as f:
            f.write(report)
        mlflow.log_artifact(report_path)

        # Log the trained model so you can reload it later without retraining.
        # mlflow.sklearn.log_model saves both the vectoriser pipeline and the
        # classifier — but here we're logging them separately for clarity.
        mlflow.sklearn.log_model(clf, artifact_path=f"model_{target}")

        print(f"\n{'='*50}")
        print(f"  TF-IDF + LR  →  {target}")
        print(f"{'='*50}")
        print(f"  Accuracy : {accuracy:.4f}")
        print(f"  F1 (weighted): {f1:.4f}")
        print(f"\n{report}")

        return {
            "accuracy": accuracy,
            "f1_weighted": f1,
            "vectoriser": vectoriser,
            "model": clf,
        }


def fit_tfidf_lr(
    X_train: list[str],
    y_train: list[str],
    max_features: int = 10_000,
    C: float = 1.0,
):
    """
    Train TF-IDF + LR without MLflow — used by the Flask API on startup.
    Returns (vectoriser, clf) ready for inference.
    """
    vectoriser = TfidfVectorizer(
        max_features=max_features,
        ngram_range=(1, 2),
        sublinear_tf=True,
    )
    X_vec = vectoriser.fit_transform(X_train)
    clf = LogisticRegression(C=C, max_iter=1000, solver="lbfgs")
    clf.fit(X_vec, y_train)
    return vectoriser, clf


# ── DistilBERT fine-tune ───────────────────────────────────────────────────────

class _TicketDataset(Dataset):
    """
    A minimal PyTorch Dataset that wraps tokenised ticket text + integer labels.

    PyTorch's DataLoader needs a Dataset object — something that knows its
    length and can return a single (input, label) pair by index.
    This is the standard pattern; nothing special here.
    """
    def __init__(self, encodings, labels):
        self.encodings = encodings  # dict of input_ids, attention_mask tensors
        self.labels = labels        # list of integer class indices

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        item = {k: v[idx] for k, v in self.encodings.items()}
        item["labels"] = torch.tensor(self.labels[idx], dtype=torch.long)
        return item


def train_distilbert(
    X_train: list[str],
    X_test: list[str],
    y_train: list[str],
    y_test: list[str],
    target: str = "priority",
    model_name: str = "distilbert-base-uncased",
    epochs: int = 3,
    batch_size: int = 16,
    lr: float = 2e-5,
    max_length: int = 128,
):
    """
    Fine-tune DistilBERT for sequence classification and log to MLflow.

    DistilBERT is a distilled (compressed) version of BERT — 40% smaller,
    60% faster, 97% of BERT's performance on GLUE benchmarks.
    Fine-tuning means we start from pretrained weights (already trained on
    Wikipedia + BookCorpus) and continue training on our ticket data.
    The model adapts its representations to understand IT support language.

    Parameters
    ----------
    max_length  : token limit per ticket. DistilBERT supports up to 512.
                  128 covers ~95% of our tickets and is much faster to train.
    lr          : 2e-5 is the standard learning rate for fine-tuning BERT-family
                  models — small enough not to destroy pretrained weights.
    batch_size  : 16 is safe on 8GB unified memory (M4 Air).
    """

    # ── Device ────────────────────────────────────────────────────────────────
    # MPS = Apple Metal — GPU acceleration on M-series Macs.
    # Falls back to CPU if MPS isn't available.
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Using device: {device}")

    # ── Label encoding ────────────────────────────────────────────────────────
    # DistilBERT expects integer class indices (0, 1, 2, ...), not strings.
    # LabelEncoder converts "Critical" → 0, "High" → 1, etc. consistently.
    le = LabelEncoder()
    le.fit(y_train)
    y_train_enc = le.transform(y_train).tolist()
    y_test_enc = le.transform(y_test).tolist()
    num_labels = len(le.classes_)
    print(f"Classes: {list(le.classes_)}")

    # ── Tokenisation ──────────────────────────────────────────────────────────
    # The tokeniser converts raw text → input_ids (integer token IDs) and
    # attention_mask (1 for real tokens, 0 for padding).
    # truncation=True: chop anything beyond max_length tokens.
    # padding=True: pad shorter sequences to the same length in each batch.
    tokeniser = DistilBertTokenizerFast.from_pretrained(model_name)

    train_enc = tokeniser(X_train, truncation=True, padding=True, max_length=max_length, return_tensors="pt")
    test_enc  = tokeniser(X_test,  truncation=True, padding=True, max_length=max_length, return_tensors="pt")

    train_dataset = _TicketDataset(train_enc, y_train_enc)
    test_dataset  = _TicketDataset(test_enc,  y_test_enc)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader  = DataLoader(test_dataset,  batch_size=batch_size)

    # ── Class weights ─────────────────────────────────────────────────────────
    # Medium makes up ~41% of training data. Without correction the model
    # finds it easier to predict Medium every time and gets stuck there.
    # Class weights invert the frequency: rare classes (Critical) get a high
    # weight so the loss penalises getting them wrong more heavily.
    # Formula: weight[i] = total_samples / (num_classes * count[i])
    counts = np.bincount(y_train_enc)
    weights = len(y_train_enc) / (num_labels * counts)
    class_weights = torch.tensor(weights, dtype=torch.float).to(device)
    loss_fn = torch.nn.CrossEntropyLoss(weight=class_weights)
    print(f"Class weights: { {le.classes_[i]: round(w, 2) for i, w in enumerate(weights)} }")

    # ── Model ─────────────────────────────────────────────────────────────────
    # from_pretrained loads the distilbert-base-uncased weights from HuggingFace.
    # num_labels tells it to add a classification head with the right output size.
    model = DistilBertForSequenceClassification.from_pretrained(model_name, num_labels=num_labels)
    model.to(device)

    optimiser = torch.optim.AdamW(model.parameters(), lr=lr)

    # Linear warmup + decay schedule: lr ramps up for the first 10% of steps,
    # then linearly decays to 0. Standard practice for fine-tuning transformers.
    total_steps = len(train_loader) * epochs
    scheduler = get_linear_schedule_with_warmup(
        optimiser,
        num_warmup_steps=int(0.1 * total_steps),
        num_training_steps=total_steps,
    )

    with mlflow.start_run(run_name=f"distilbert_{target}"):

        mlflow.log_param("model", model_name)
        mlflow.log_param("target", target)
        mlflow.log_param("epochs", epochs)
        mlflow.log_param("batch_size", batch_size)
        mlflow.log_param("lr", lr)
        mlflow.log_param("max_length", max_length)
        mlflow.log_param("class_weights", {le.classes_[i]: round(w, 3) for i, w in enumerate(weights)})

        # ── Training loop ─────────────────────────────────────────────────
        for epoch in range(epochs):
            model.train()
            total_loss = 0

            for batch in train_loader:
                optimiser.zero_grad()
                input_ids      = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                labels         = batch["labels"].to(device)

                # Pass labels=None so the model returns raw logits without
                # computing its own unweighted loss. We compute weighted loss manually.
                outputs = model(input_ids=input_ids, attention_mask=attention_mask)
                loss = loss_fn(outputs.logits, labels)
                loss.backward()

                # Gradient clipping prevents exploding gradients — a common
                # instability when fine-tuning large pretrained models.
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

                optimiser.step()
                scheduler.step()
                total_loss += loss.item()

            avg_loss = total_loss / len(train_loader)
            print(f"  Epoch {epoch+1}/{epochs}  loss: {avg_loss:.4f}")
            mlflow.log_metric("train_loss", avg_loss, step=epoch)

        # ── Evaluation ────────────────────────────────────────────────────
        model.eval()
        all_preds = []

        with torch.no_grad():
            for batch in test_loader:
                input_ids      = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                outputs = model(input_ids=input_ids, attention_mask=attention_mask)
                preds = torch.argmax(outputs.logits, dim=1).cpu().numpy()
                all_preds.extend(preds)

        y_pred_labels = le.inverse_transform(all_preds)
        accuracy = accuracy_score(y_test, y_pred_labels)
        f1 = f1_score(y_test, y_pred_labels, average="weighted")
        report = classification_report(y_test, y_pred_labels)

        mlflow.log_metric("accuracy", accuracy)
        mlflow.log_metric("f1_weighted", f1)

        report_path = f"/tmp/report_distilbert_{target}.txt"
        with open(report_path, "w") as f:
            f.write(report)
        mlflow.log_artifact(report_path)

        print(f"\n{'='*50}")
        print(f"  DistilBERT  →  {target}")
        print(f"{'='*50}")
        print(f"  Accuracy : {accuracy:.4f}")
        print(f"  F1 (weighted): {f1:.4f}")
        print(f"\n{report}")

        return {
            "accuracy": accuracy,
            "f1_weighted": f1,
            "model": model,
            "label_encoder": le,
        }
