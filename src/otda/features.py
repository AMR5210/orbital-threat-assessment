"""Feature-set and column-name constants shared across the codebase.

Deliberately dependency-free (no duckdb, not even pandas) so serving
code (otda.inference, otda.api) can import feature-list constants
without pulling in the DuckDB-querying machinery in otda.data — that
machinery is only needed for training/analysis, never for serving an
already-trained model. (The Docker serving image doesn't install
duckdb — see requirements-serving.txt — so otda.inference must not
import otda.data, which would pull duckdb in transitively and break
the container.)
"""
from __future__ import annotations

# Feature sets from the original two-experiment design (README /
# legacy/Project_codeFile.ipynb Cells 15 and 26). Column names updated
# to match the dbt marts (`neo` -> `is_neo`, `class` text -> `class_code`).
EXP1_FEATURES = ["e", "q", "H", "is_neo", "class_code"]
EXP2_FEATURES = [*EXP1_FEATURES, "moid"]

# Orbital element features for the unsupervised analysis (Stage 5) —
# deliberately distinct from EXP1/EXP2_FEATURES: no is_neo (categorical
# flag, not an orbital element), includes `a` and `i` (semi-major axis,
# inclination), which the PHA classifiers never used but which drive
# the *dynamical* class label (AMO/APO/MBA/TNO/...) this analysis
# compares clusters against.
CLUSTER_FEATURES = ["a", "e", "i", "moid", "H"]

TARGET_COL = "is_pha"
ID_COL = "spkid"
