"""Model factories and hyperparameter search spaces.

Ported from legacy/Project_codeFile.ipynb Cells 18-23 (model definitions)
and Cells 20/22/30 (RandomizedSearchCV param_dist dicts). class_weight
and default hyperparameters are unchanged from the original notebook.
"""
from __future__ import annotations

from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.tree import DecisionTreeClassifier

RANDOM_STATE = 42

# Models that need scaled input (StandardScaler fit on train, applied to
# val/test) — matches the notebook's Cell 17 comment: "Decision Tree and
# Random Forest use unscaled data — trees are scale-invariant."
NEEDS_SCALING = {"logistic_regression": True, "decision_tree": False,
                  "random_forest": False, "linear_svc": True}


def make_model(name: str, **overrides):
    if name == "logistic_regression":
        return LogisticRegression(
            class_weight="balanced", max_iter=1000, random_state=RANDOM_STATE, **overrides
        )
    if name == "decision_tree":
        params = {"max_depth": 4, "class_weight": "balanced", "random_state": RANDOM_STATE}
        params.update(overrides)
        return DecisionTreeClassifier(**params)
    if name == "random_forest":
        params = {
            "n_estimators": 100, "class_weight": "balanced",
            "random_state": RANDOM_STATE, "n_jobs": -1,
        }
        params.update(overrides)
        return RandomForestClassifier(**params)
    if name == "linear_svc":
        svc = LinearSVC(
            class_weight="balanced", max_iter=2000, random_state=RANDOM_STATE, **overrides
        )
        return CalibratedClassifierCV(svc, cv=3)
    raise ValueError(f"Unknown model: {name}")


# Hyperparameter search spaces, ported from the notebook's
# RandomizedSearchCV param_dist dicts (Cells 20, 22, 30). Only
# decision_tree and random_forest are tuned — matches the original,
# which never tuned logistic_regression or linear_svc beyond threshold
# selection (Cell 24).
PARAM_GRIDS = {
    "decision_tree": {
        "max_depth": [2, 3, 4, 5, 6, 8, 10, None],
        "min_samples_split": [2, 5, 10, 20, 50],
        "min_samples_leaf": [1, 2, 5, 10],
        "criterion": ["gini", "entropy"],
    },
    "random_forest": {
        "n_estimators": [50, 100, 150, 200],
        "max_depth": [4, 6, 8, 10, None],
        "max_features": ["sqrt", "log2", 0.3, 0.5],
        "min_samples_split": [2, 5, 10],
        "min_samples_leaf": [1, 2, 5],
    },
}

ALL_MODELS = ["logistic_regression", "decision_tree", "random_forest", "linear_svc"]
TUNABLE_MODELS = ["decision_tree", "random_forest"]
