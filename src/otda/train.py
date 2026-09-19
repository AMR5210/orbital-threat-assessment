"""Experiment orchestration: split -> SMOTE(train only) -> scale ->
hyperparameter search scored on validation -> single test evaluation.

This is the corrected replacement for the original notebook's model
selection (Cells 20/22/30 RandomizedSearchCV + Cells 18-29 fitting):
the original tuned hyperparameters via cross-validated PR-AUC on
SMOTE-resampled training data (inflated and acknowledged as such in
the notebook's own comments), then separately compared models by
eyeballing test-set metrics — using the test set to pick a winner, not
just to report one.

Here, hyperparameter candidates are scored directly against the held-
out validation set (true class distribution, no SMOTE, no CV) — a
harder, more honest proxy for generalization. Test is touched exactly
once, after the winning configuration is already fixed.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd
from sklearn.model_selection import ParameterSampler
from sklearn.preprocessing import StandardScaler

from otda.evaluate import Metrics, compute_metrics
from otda.models import NEEDS_SCALING, PARAM_GRIDS, TUNABLE_MODELS, make_model
from otda.split import Split, smote_resample


@dataclass
class ExperimentResult:
    model_name: str
    best_params: dict = field(default_factory=dict)
    val_metrics: Metrics | None = None
    test_metrics: Metrics | None = None
    # Fitted on train_resampled with the winning hyperparameters, in
    # that order — set by run_experiment's final refit only (never by
    # a search candidate), so this is exactly "the model that produced
    # test_metrics." scaler is None for tree-based models.
    fitted_model: object = None
    fitted_scaler: object = None


def _fit_predict(model, X_train, y_train, X_eval, scale: bool):
    scaler = None
    if scale:
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_eval = scaler.transform(X_eval)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_eval)
    y_prob = model.predict_proba(X_eval)[:, 1]
    return y_pred, y_prob, model, scaler


def _search_on_validation(
    model_name: str,
    X_train_res: pd.DataFrame,
    y_train_res: pd.Series,
    X_val: pd.DataFrame,
    y_val: pd.Series,
    n_iter: int,
    random_state: int,
) -> tuple[dict, Metrics]:
    """Sample n_iter candidate hyperparameter sets (sklearn's
    ParameterSampler — sampling only, no CV) and keep the one with the
    best validation PR-AUC. Falls back to the model's defaults (a
    single "candidate": {}) for models with no search space."""
    grid = PARAM_GRIDS.get(model_name)
    scale = NEEDS_SCALING[model_name]

    if grid is None:
        model = make_model(model_name)
        y_pred, y_prob, _, _ = _fit_predict(model, X_train_res, y_train_res, X_val, scale)
        return {}, compute_metrics(y_val, y_pred, y_prob)

    candidates = list(
        ParameterSampler(grid, n_iter=n_iter, random_state=random_state)
    )
    best_params, best_metrics = None, None
    for params in candidates:
        model = make_model(model_name, **params)
        y_pred, y_prob, _, _ = _fit_predict(model, X_train_res, y_train_res, X_val, scale)
        metrics = compute_metrics(y_val, y_pred, y_prob)
        if best_metrics is None or metrics.pr_auc > best_metrics.pr_auc:
            best_params, best_metrics = params, metrics

    return best_params, best_metrics


def run_experiment(
    split: Split,
    model_name: str,
    n_iter: int = 10,
    random_state: int = 42,
) -> ExperimentResult:
    """Full pipeline for one model on one already-split dataset:
    SMOTE the training fold, search hyperparameters against validation
    (tunable models only), refit the winning config, then evaluate once
    on test."""
    X_train_res, y_train_res = smote_resample(split.X_train, split.y_train, random_state)

    if model_name in TUNABLE_MODELS:
        best_params, val_metrics = _search_on_validation(
            model_name, X_train_res, y_train_res, split.X_val, split.y_val, n_iter, random_state
        )
    else:
        best_params = {}
        scale = NEEDS_SCALING[model_name]
        model = make_model(model_name)
        y_pred, y_prob, _, _ = _fit_predict(model, X_train_res, y_train_res, split.X_val, scale)
        val_metrics = compute_metrics(split.y_val, y_pred, y_prob)

    # Refit the winning configuration and touch test exactly once.
    final_model = make_model(model_name, **best_params)
    scale = NEEDS_SCALING[model_name]
    y_pred, y_prob, fitted_model, fitted_scaler = _fit_predict(
        final_model, X_train_res, y_train_res, split.X_test, scale
    )
    test_metrics = compute_metrics(split.y_test, y_pred, y_prob)

    return ExperimentResult(
        model_name=model_name,
        best_params=best_params,
        val_metrics=val_metrics,
        test_metrics=test_metrics,
        fitted_model=fitted_model,
        fitted_scaler=fitted_scaler,
    )
