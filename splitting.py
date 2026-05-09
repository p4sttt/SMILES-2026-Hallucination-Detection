"""
splitting.py — Train / validation / test split utilities (student-implementable).

``split_data`` receives the label array ``y`` and, optionally, the full
DataFrame ``df`` (for group-aware splits).  It must return a list of
``(idx_train, idx_val, idx_test)`` tuples of integer index arrays.

Contract
--------
* ``idx_train``, ``idx_val``, ``idx_test`` are 1-D NumPy arrays of integer
  indices into the full dataset.
* ``idx_val`` may be ``None`` if no separate validation fold is needed.
* All indices must be non-overlapping; together they must cover every sample.
* Return a **list** — one element for a single split, K elements for k-fold.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.model_selection import (
    StratifiedGroupKFold,
    StratifiedKFold,
    train_test_split,
)


def split_data(
    y: np.ndarray,
    df: pd.DataFrame | None = None,
    test_size: float = 0.15,
    val_size: float = 0.15,
    random_state: int = 42,
) -> list[tuple[np.ndarray, np.ndarray | None, np.ndarray]]:
    """Split dataset indices into train, validation, and test subsets.

    The default strategy performs 5-fold stratified cross-validation. If the
    source DataFrame is provided, exact duplicate responses are kept in the same
    fold to avoid leaking repeated answers across train/validation/test.
    Inside each training fold, a small validation split is carved out for probe
    hyperparameter and threshold tuning.

    Args:
        y:            Label array of shape ``(N,)`` with values in ``{0, 1}``.
                      Used for stratification.
        df:           Optional full DataFrame (same row order as ``y``).
                      Required for group-aware splits.
        test_size:    Kept for API compatibility; 5-fold CV uses 20% test
                      per fold.
        val_size:     Fraction of samples reserved for validation.
        random_state: Random seed for reproducible splits.

    Returns:
        A list of ``(idx_train, idx_val, idx_test)`` tuples of integer index
        arrays.  ``idx_val`` may be ``None``.

    Student task:
        Replace or extend the skeleton below.  The only contract is that the
        function returns the list described above.
    """
    idx = np.arange(len(y))
    folds = []

    groups = _response_groups(df, len(y))
    if groups is None:
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=random_state)
        split_iter = cv.split(idx, y)
    else:
        cv = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=random_state)
        split_iter = cv.split(idx, y, groups)

    for fold_idx, (idx_train_val, idx_test) in enumerate(split_iter):
        idx_train, idx_val = _split_train_val(
            idx_train_val=idx_train_val,
            y=y,
            groups=groups,
            val_size=val_size,
            random_state=random_state + fold_idx,
        )
        folds.append(
            (
                np.asarray(idx_train, dtype=int),
                np.asarray(idx_val, dtype=int),
                np.asarray(idx_test, dtype=int),
            )
        )

    return folds


def _response_groups(df: pd.DataFrame | None, n_rows: int) -> np.ndarray | None:
    if df is None or "response" not in df.columns or len(df) != n_rows:
        return None
    return pd.factorize(df["response"].fillna("").astype(str), sort=False)[0]


def _split_train_val(
    idx_train_val: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray | None,
    val_size: float,
    random_state: int,
) -> tuple[np.ndarray, np.ndarray]:
    if groups is None:
        return train_test_split(
            idx_train_val,
            test_size=val_size,
            random_state=random_state,
            stratify=y[idx_train_val],
        )

    n_val_splits = max(2, int(round(1.0 / val_size)))
    n_groups = np.unique(groups[idx_train_val]).size
    min_class_count = int(np.bincount(y[idx_train_val].astype(int)).min())
    if n_groups < n_val_splits or min_class_count < n_val_splits:
        return train_test_split(
            idx_train_val,
            test_size=val_size,
            random_state=random_state,
            stratify=y[idx_train_val],
        )

    inner_cv = StratifiedGroupKFold(
        n_splits=n_val_splits,
        shuffle=True,
        random_state=random_state,
    )
    inner_train, inner_val = next(
        inner_cv.split(idx_train_val, y[idx_train_val], groups[idx_train_val])
    )
    return idx_train_val[inner_train], idx_train_val[inner_val]
