"""Three-way stratified split + train-only SMOTE.

Replaces the original notebook's 80/20 split (Cells 16/27), which fed
RandomizedSearchCV's CV-selected models straight into a test-set
comparison that then drove model selection — using the test set twice.
This module produces train/val/test instead: hyperparameter search is
scored against val (src/otda/train.py), and test is touched exactly
once, at the end, to report the final number.
"""
from __future__ import annotations

from dataclasses import dataclass

import pandas as pd
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split


@dataclass
class Split:
    X_train: pd.DataFrame
    X_val: pd.DataFrame
    X_test: pd.DataFrame
    y_train: pd.Series
    y_val: pd.Series
    y_test: pd.Series


def three_way_split(
    X: pd.DataFrame,
    y: pd.Series,
    val_size: float = 0.2,
    test_size: float = 0.2,
    random_state: int = 42,
) -> Split:
    """60/20/20 stratified split by default. test_size and val_size are
    both fractions of the *original* X/y (not of the train remainder)."""
    X_train_val, X_test, y_train_val, y_test = train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=random_state
    )
    # val_size was specified as a fraction of the full set; rescale it
    # to a fraction of the train_val remainder.
    val_fraction_of_remainder = val_size / (1 - test_size)
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_val,
        y_train_val,
        test_size=val_fraction_of_remainder,
        stratify=y_train_val,
        random_state=random_state,
    )
    return Split(
        X_train=X_train, X_val=X_val, X_test=X_test,
        y_train=y_train, y_val=y_val, y_test=y_test,
    )


def smote_resample(
    X_train: pd.DataFrame, y_train: pd.Series, random_state: int = 42
) -> tuple[pd.DataFrame, pd.Series]:
    """SMOTE on the training fold only. Never call this on val or test —
    both must retain the true class distribution for honest evaluation
    and for hyperparameter selection to mean anything."""
    smote = SMOTE(random_state=random_state)
    X_res, y_res = smote.fit_resample(X_train, y_train)
    return X_res, y_res
