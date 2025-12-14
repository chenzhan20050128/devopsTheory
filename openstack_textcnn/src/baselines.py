"""Classic baselines for OpenStack log anomaly detection.

These baselines are designed to be drop-in comparable with the existing TextCNN
pipeline in `train.py`:

- Uses the same `load_dataset` + `split_dataset` preprocessing output.
- Evaluates on the same metrics (accuracy/precision/recall/f1/auc/cm).
- Writes outputs that match the existing artifacts layout.

Implemented baselines
---------------------
1) Rule-based keyword matching
2) Statistical keyword frequency threshold (per window)
3) Traditional ML: TF-IDF + Linear SVM
4) Traditional ML: TF-IDF + Logistic Regression

We intentionally keep these baselines simple and classical for thesis
comparison.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Optional, Tuple

import numpy as np

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC


@dataclass
class BaselineResult:
    probs: np.ndarray  # shape (n,)
    preds: np.ndarray  # shape (n,)


def _sigmoid(x: np.ndarray) -> np.ndarray:
    # Numerically stable sigmoid
    x = np.asarray(x, dtype=np.float64)
    out = np.empty_like(x)
    pos = x >= 0
    out[pos] = 1.0 / (1.0 + np.exp(-x[pos]))
    exp_x = np.exp(x[~pos])
    out[~pos] = exp_x / (1.0 + exp_x)
    return out.astype(np.float64)


def keyword_rule_scores(
    texts: Iterable[str],
    keywords: Iterable[str],
) -> np.ndarray:
    """Return anomaly probability as {0,1} by keyword existence."""
    keywords_l = tuple(k.lower() for k in keywords)
    probs = []
    for t in texts:
        tl = (t or "").lower()
        probs.append(1.0 if any(k in tl for k in keywords_l) else 0.0)
    return np.asarray(probs, dtype=np.float64)


def keyword_frequency_scores(
    texts: Iterable[str],
    keywords: Iterable[str],
    *,
    mode: str = "ratio",
) -> np.ndarray:
    """Compute anomaly score based on abnormal keyword frequency.

    Parameters
    ----------
    mode:
        - "count": number of keyword occurrences in a window text
        - "ratio": count / max(token_count,1)

    Returns
    -------
    score in [0, +inf) for "count" or [0,1] for "ratio".
    """

    keywords_l = tuple(k.lower() for k in keywords)
    scores: List[float] = []
    for t in texts:
        tl = (t or "").lower()
        count = 0
        for k in keywords_l:
            # count overlapping is not needed, use str.count
            count += tl.count(k)
        if mode == "count":
            scores.append(float(count))
        elif mode == "ratio":
            token_count = max(len(tl.split()), 1)
            scores.append(float(count) / float(token_count))
        else:
            raise ValueError(f"Unknown mode={mode}")
    return np.asarray(scores, dtype=np.float64)


def apply_threshold(scores: np.ndarray, threshold: float) -> BaselineResult:
    scores = np.asarray(scores, dtype=np.float64)
    preds = (scores >= threshold).astype(int)

    # If scores already in [0,1], treat as probabilities.
    # Otherwise, map to probability with sigmoid for AUC computation.
    if np.nanmin(scores) >= 0.0 and np.nanmax(scores) <= 1.0:
        probs = scores
    else:
        probs = _sigmoid(scores)

    return BaselineResult(probs=probs, preds=preds)


def build_tfidf_linearsvm_pipeline(
    *,
    min_df: int = 2,
    max_features: Optional[int] = 50000,
    ngram_range: Tuple[int, int] = (1, 2),
    C: float = 1.0,
) -> Pipeline:
    """TF-IDF + LinearSVC.

    LinearSVC doesn't provide `predict_proba`, so we use `decision_function`
    and apply sigmoid as a pseudo-probability for AUC.
    """

    vectorizer = TfidfVectorizer(
        lowercase=True,
        min_df=min_df,
        max_features=max_features,
        ngram_range=ngram_range,
    )
    clf = LinearSVC(C=C)
    return Pipeline([("tfidf", vectorizer), ("clf", clf)])


def build_tfidf_logreg_pipeline(
    *,
    min_df: int = 2,
    max_features: Optional[int] = 50000,
    ngram_range: Tuple[int, int] = (1, 2),
    C: float = 1.0,
    max_iter: int = 2000,
) -> Pipeline:
    vectorizer = TfidfVectorizer(
        lowercase=True,
        min_df=min_df,
        max_features=max_features,
        ngram_range=ngram_range,
    )
    clf = LogisticRegression(
        C=C,
        max_iter=max_iter,
        solver="liblinear",
        class_weight="balanced",
    )
    return Pipeline([("tfidf", vectorizer), ("clf", clf)])


def fit_predict_tfidf_model(
    model: Pipeline,
    train_texts: List[str],
    train_labels: np.ndarray,
    test_texts: List[str],
    *,
    threshold: float = 0.5,
) -> BaselineResult:
    model.fit(train_texts, train_labels)

    clf = model.named_steps["clf"]
    if hasattr(clf, "predict_proba"):
        probs = model.predict_proba(test_texts)[:, 1]
    elif hasattr(clf, "decision_function"):
        scores = model.decision_function(test_texts)
        probs = _sigmoid(scores)
    else:
        # Fallback: hard predictions
        hard = np.asarray(model.predict(test_texts))
        return BaselineResult(
            probs=hard.astype(np.float64),
            preds=hard.astype(int),
        )

    preds = (probs >= threshold).astype(int)
    return BaselineResult(
        probs=np.asarray(probs, dtype=np.float64),
        preds=preds,
    )
