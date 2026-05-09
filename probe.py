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

from sklearn.decomposition import PCA
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
        self._best_c = 0.03
        self._best_class_weight = None
        self._ensemble_top_k = 3
        self._default_c_grid = np.array([0.01, 0.03, 0.1])
        self._search_c_grid = np.logspace(-4, 0, 13)
        self._class_weight_options = (None, "balanced")
        self._pca_feature_threshold = 512
        self._pca_components = 64

    def _make_clf(
        self,
        C: float,
        class_weight: str | None = None,
        n_components: int | None = None,
    ) -> Pipeline:
        steps = [("scaler", StandardScaler())]
        if n_components is not None:
            steps.append(
                (
                    "pca",
                    PCA(
                        n_components=n_components,
                        whiten=True,
                        svd_solver="randomized",
                        random_state=42,
                    ),
                )
            )

        steps.append(
            (
                "logreg",
                LogisticRegression(
                    C=C,
                    class_weight=class_weight,
                    solver="liblinear",
                    max_iter=3000,
                    random_state=42,
                ),
            )
        )
        return Pipeline(steps)

    def _pca_n_components(self, X: np.ndarray) -> int | None:
        if X.shape[1] <= self._pca_feature_threshold:
            return None
        n_components = min(self._pca_components, X.shape[0] - 1, X.shape[1])
        return int(n_components) if n_components >= 2 else None

    def fit(self, X: np.ndarray, y: np.ndarray) -> "HallucinationProbe":
        """Train the probe on labelled feature vectors."""
        self._X_train = X.copy()
        self._y_train = y.copy()

        n_components = self._pca_n_components(X)
        self._clfs = []
        for C in self._default_c_grid:
            for class_weight in self._class_weight_options:
                clf = self._make_clf(
                    C=float(C),
                    class_weight=class_weight,
                    n_components=n_components,
                )
                clf.fit(X, y)
                self._clfs.append(clf)

        self._clf = self._clfs[0]
        self._best_c = float(self._default_c_grid[1])
        self._best_class_weight = None
        self._threshold = 0.5
        return self

    def fit_hyperparameters(
        self, X_val: np.ndarray, y_val: np.ndarray
    ) -> "HallucinationProbe":
        """Tune regularization strength and decision threshold on validation."""
        if self._X_train is None or self._y_train is None:
            raise RuntimeError("Call fit() before fit_hyperparameters().")

        model_records = []

        n_components = self._pca_n_components(self._X_train)
        for C in self._search_c_grid:
            for class_weight in self._class_weight_options:
                clf = self._make_clf(
                    C=float(C),
                    class_weight=class_weight,
                    n_components=n_components,
                )
                clf.fit(self._X_train, self._y_train)
                probs = clf.predict_proba(X_val)[:, 1]

                threshold, acc = self._best_accuracy_threshold(y_val, probs)
                if np.unique(y_val).size < 2:
                    auroc = 0.0
                else:
                    auroc = roc_auc_score(y_val, probs)
                    if not np.isfinite(auroc):
                        auroc = 0.0

                model_records.append(
                    {
                        "accuracy": acc,
                        "auroc": auroc,
                        "regularization_tie_break": -abs(np.log10(C / 0.03)),
                        "class_weight_tie_break": 1 if class_weight is None else 0,
                        "C": float(C),
                        "class_weight": class_weight,
                        "clf": clf,
                        "threshold": threshold,
                    }
                )

        ranked = sorted(
            model_records,
            key=lambda record: (
                record["accuracy"],
                record["auroc"],
                record["regularization_tie_break"],
                record["class_weight_tie_break"],
            ),
            reverse=True,
        )
        selected = ranked[: self._ensemble_top_k]

        self._clfs = [record["clf"] for record in selected]
        self._clf = self._clfs[0]
        self._best_c = selected[0]["C"]
        self._best_class_weight = selected[0]["class_weight"]

        probs = self.predict_proba(X_val)[:, 1]
        best_threshold, _ = self._best_accuracy_threshold(y_val, probs)
        self._threshold = best_threshold
        return self

    @staticmethod
    def _best_accuracy_threshold(
        y_true: np.ndarray,
        probs: np.ndarray,
    ) -> tuple[float, float]:
        sorted_probs = np.unique(probs)
        midpoints = (sorted_probs[:-1] + sorted_probs[1:]) / 2.0
        candidates = np.unique(np.concatenate([[0.0, 1.0], sorted_probs, midpoints]))

        best_threshold = 0.5
        best_acc = -1.0
        for threshold in candidates:
            pred = (probs >= threshold).astype(int)
            score = accuracy_score(y_true, pred)
            tie_break = score == best_acc and abs(threshold - 0.5) < abs(
                best_threshold - 0.5
            )
            if score > best_acc or tie_break:
                best_acc = score
                best_threshold = float(threshold)

        return best_threshold, float(best_acc)

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict binary labels for feature vectors."""
        return (self.predict_proba(X)[:, 1] >= self._threshold).astype(int)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Return class probabilities with column 1 as hallucination probability."""
        if not self._clfs:
            raise RuntimeError("Probe is not fitted.")
        return np.mean([clf.predict_proba(X) for clf in self._clfs], axis=0)
