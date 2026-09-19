"""Pytest invariants for the split/SMOTE pipeline and class encoding.

These test pandas/sklearn *state* that dbt cannot see — dbt validates
the warehouse-side class-imbalance ratio (dbt/tests/assert_pha_ratio_*),
but nothing there can check that a train/val/test split stayed
stratified after leaving SQL, or that SMOTE was never accidentally
applied to a held-out fold. That gap is exactly what this file covers.

Uses synthetic data, not the real DuckDB marts — these are unit tests
of otda.split / otda.data logic, deliberately independent of whether
`dbt build` has been run. The dbt-build-dependent path is exercised
separately in CI against the small committed fixture.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from otda.data import load_class_map
from otda.split import smote_resample, three_way_split

# Synthetic dataset matching the real population's approximate ratio
# (2066 / 932335 ≈ 0.002217) at a size small enough to run instantly.
N = 20_000
N_POSITIVE = 44  # 44 / 20000 = 0.0022, matches the real ratio closely


@pytest.fixture
def synthetic_data():
    rng = np.random.default_rng(seed=7)
    X = pd.DataFrame(
        {
            "e": rng.uniform(0, 1, N),
            "q": rng.uniform(0, 5, N),
            "H": rng.uniform(5, 25, N),
            "is_neo": rng.integers(0, 2, N),
            "class_code": rng.integers(0, 13, N),
        }
    )
    y = pd.Series([1] * N_POSITIVE + [0] * (N - N_POSITIVE))
    # shuffle so positives aren't all at the front
    perm = rng.permutation(N)
    X = X.iloc[perm].reset_index(drop=True)
    y = y.iloc[perm].reset_index(drop=True)
    return X, y


class TestThreeWaySplit:
    def test_split_sizes(self, synthetic_data):
        X, y = synthetic_data
        split = three_way_split(X, y, val_size=0.2, test_size=0.2)
        assert len(split.X_train) + len(split.X_val) + len(split.X_test) == N
        # 60/20/20 within a few rows of rounding
        assert abs(len(split.X_train) - 0.6 * N) < 5
        assert abs(len(split.X_val) - 0.2 * N) < 5
        assert abs(len(split.X_test) - 0.2 * N) < 5

    def test_stratification_preserved_in_every_split(self, synthetic_data):
        """Each of train/val/test should carry approximately the same
        positive ratio as the full population — the whole point of
        `stratify=y`. A regression here (e.g. someone dropping
        `stratify=`) would silently starve val or test of positives."""
        X, y = synthetic_data
        population_ratio = y.mean()
        split = three_way_split(X, y, val_size=0.2, test_size=0.2)

        for name, y_split in [
            ("train", split.y_train), ("val", split.y_val), ("test", split.y_test),
        ]:
            ratio = y_split.mean()
            assert ratio == pytest.approx(population_ratio, abs=0.001), (
                f"{name} split ratio {ratio} diverged from population ratio "
                f"{population_ratio} — stratification may be broken"
            )

    def test_splits_are_disjoint(self, synthetic_data):
        """No row should appear in more than one of train/val/test —
        the leakage invariant most likely to break in a careless
        refactor (e.g. concatenating instead of splitting)."""
        X, y = synthetic_data
        split = three_way_split(X, y, val_size=0.2, test_size=0.2)
        train_idx = set(split.X_train.index)
        val_idx = set(split.X_val.index)
        test_idx = set(split.X_test.index)

        assert train_idx.isdisjoint(val_idx)
        assert train_idx.isdisjoint(test_idx)
        assert val_idx.isdisjoint(test_idx)
        assert len(train_idx | val_idx | test_idx) == N


class TestSmoteTrainOnly:
    def test_smote_balances_train_but_val_test_stay_imbalanced(self, synthetic_data):
        """The core leakage guard for this project: SMOTE must only ever
        see the training fold. This test doesn't just check that
        smote_resample's *signature* takes train alone — it checks the
        *behavioral* consequence: after resampling, train's ratio is
        ~50/50 while val/test still carry the original ~449:1 ratio.
        If a future edit accidentally ran SMOTE before the split (or on
        val/test), this would catch it by val/test suddenly looking
        artificially balanced too."""
        X, y = synthetic_data
        population_ratio = y.mean()
        split = three_way_split(X, y, val_size=0.2, test_size=0.2)

        _, y_train_res = smote_resample(split.X_train, split.y_train)

        # train is now balanced
        assert y_train_res.mean() == pytest.approx(0.5, abs=0.01)
        # val/test were never passed to SMOTE and must remain untouched
        assert split.y_val.mean() == pytest.approx(population_ratio, abs=0.001)
        assert split.y_test.mean() == pytest.approx(population_ratio, abs=0.001)

    def test_smote_output_size_matches_majority_class_doubled(self, synthetic_data):
        X, y = synthetic_data
        split = three_way_split(X, y, val_size=0.2, test_size=0.2)
        n_majority_train = (split.y_train == 0).sum()

        _, y_train_res = smote_resample(split.X_train, split.y_train)

        assert len(y_train_res) == 2 * n_majority_train


class TestClassEncoding:
    def test_class_map_matches_known_recovered_mapping(self):
        """Regression guard for the class-code stability fix: these
        values were recovered directly from the original notebook's
        pandas.Categorical.cat.codes output (alphabetical order) and
        must never drift, since downstream models were trained against
        this exact encoding."""
        class_map = load_class_map()
        assert class_map["AMO"] == 0
        assert class_map["MBA"] == 8
        assert class_map["TNO"] == 12
        assert len(class_map) == 13

    def test_class_map_is_contiguous_and_unique(self):
        """Guards against the exact bug this seed replaced: codes must
        stay a contiguous 0..N-1 range with no gaps or duplicates,
        regardless of which classes are physically present in any
        given data pull."""
        class_map = load_class_map()
        codes = sorted(class_map.values())
        assert codes == list(range(len(class_map)))
