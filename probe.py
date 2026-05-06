"""
probe.py - Hallucination probe classifier (student-implemented).

Implements ``HallucinationProbe``, a binary classifier that classifies feature
vectors as truthful (0) or hallucinated (1). Called from ``solution.py`` via
``evaluate.run_evaluation``. All four public methods (``fit``,
``fit_hyperparameters``, ``predict``, ``predict_proba``) must be implemented
and their signatures must not change.
"""

from __future__ import annotations

import numpy as np
import torch.nn as nn

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


class HallucinationProbe(nn.Module):
    """Binary hallucination detector backed by scaled logistic regression."""

    def __init__(self) -> None:
        super().__init__()
        self._threshold: float = 0.5
        self._clf: Pipeline | None = None
        self._clfs: list[Pipeline] = []
        self._X_train: np.ndarray | None = None
        self._y_train: np.ndarray | None = None
        self._best_c = 0.1
        self._ensemble_top_k = 5
        self._default_c_grid = np.array([0.01, 0.03, 0.1, 0.3, 1.0])

    def _make_clf(self, C: float) -> Pipeline:
        steps = [("scaler", StandardScaler())]

        steps.append(
            (
                "logreg",
                LogisticRegression(
                    C=C,
                    class_weight="balanced",
                    solver="liblinear",
                    max_iter=5000,
                    random_state=42,
                ),
            )
        )
        return Pipeline(steps)

    def fit(self, X: np.ndarray, y: np.ndarray) -> "HallucinationProbe":
        """Train the probe on labelled feature vectors."""
        self._X_train = X.copy()
        self._y_train = y.copy()

        self._clfs = []
        for C in self._default_c_grid:
            clf = self._make_clf(C=float(C))
            clf.fit(X, y)
            self._clfs.append(clf)

        self._clf = self._clfs[0]
        self._best_c = float(self._default_c_grid[2])
        self._threshold = 0.5
        return self

    def fit_hyperparameters(
        self, X_val: np.ndarray, y_val: np.ndarray
    ) -> "HallucinationProbe":
        """Tune regularization strength and decision threshold on validation."""
        if self._X_train is None or self._y_train is None:
            raise RuntimeError("Call fit() before fit_hyperparameters().")

        model_records = []

        for C in np.logspace(-4, 2, 25):
            clf = self._make_clf(C=C)
            clf.fit(self._X_train, self._y_train)
            probs = clf.predict_proba(X_val)[:, 1]

            try:
                auroc = roc_auc_score(y_val, probs)
            except ValueError:
                auroc = 0.0

            acc = accuracy_score(y_val, (probs >= 0.5).astype(int))
            model_records.append(
                {
                    "auroc": auroc,
                    "accuracy": acc,
                    "regularization_tie_break": -abs(np.log10(C / 0.1)),
                    "C": float(C),
                    "clf": clf,
                }
            )

        ranked = sorted(
            model_records,
            key=lambda record: (
                record["auroc"],
                record["accuracy"],
                record["regularization_tie_break"],
            ),
            reverse=True,
        )
        selected = ranked[: self._ensemble_top_k]

        self._clfs = [record["clf"] for record in selected]
        self._clf = self._clfs[0]
        self._best_c = selected[0]["C"]

        probs = self.predict_proba(X_val)[:, 1]
        sorted_probs = np.unique(probs)
        midpoints = (sorted_probs[:-1] + sorted_probs[1:]) / 2.0
        candidates = np.unique(np.concatenate([[0.0, 1.0], sorted_probs, midpoints]))

        best_threshold = 0.5
        best_acc = -1.0
        for t in candidates:
            pred = (probs >= t).astype(int)
            score = accuracy_score(y_val, pred)
            is_better = score > best_acc
            is_tie_better = score == best_acc and abs(t - 0.5) < abs(
                best_threshold - 0.5
            )
            if is_better or is_tie_better:
                best_acc = score
                best_threshold = float(t)

        self._threshold = best_threshold
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict binary labels for feature vectors."""
        return (self.predict_proba(X)[:, 1] >= self._threshold).astype(int)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Return class probabilities with column 1 as hallucination probability."""
        if not self._clfs:
            raise RuntimeError("Probe is not fitted.")
        return np.mean([clf.predict_proba(X) for clf in self._clfs], axis=0)
