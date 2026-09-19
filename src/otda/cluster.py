"""Unsupervised analysis: do orbital elements alone recover the known
dynamical asteroid classes?

Secondary analysis, independent of the PHA classification task. Three
methods, two different inductive biases, compared against the same
ground truth (the `class` label — AMO/APO/MBA/TNO/... — which is a
human/IAU dynamical classification, not something derived from PHA
status):

  - KMeans: centroid-based, k fixed to 13 (the real number of classes)
    for a direct comparison against the known partition.
  - HDBSCAN: density-based, no fixed k, no single global eps (unlike
    plain DBSCAN) — better suited to a population where MBA alone is
    ~800K objects and HYA-scale classes have single digits.
  - k-NN similarity graph + Louvain community detection: a distinct
    inductive bias again (connectivity/density on a graph rather than
    distance-to-centroid), so agreement across all three is a stronger
    signal than agreement between any two similar methods would be.

Agreement is reported via Adjusted Rand Index and Normalized Mutual
Information (sklearn.metrics) — not a confusion-matrix eyeball, since
cluster label *numbers* are arbitrary and unaligned with class codes.
"""
from __future__ import annotations

from dataclasses import dataclass

import igraph as ig
import numpy as np
import pandas as pd
from hdbscan import HDBSCAN
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler

RANDOM_STATE = 42
N_CLASSES = 13  # matches dbt/seeds/asteroid_class_map.csv


@dataclass
class ClusterResult:
    method: str
    n_clusters_found: int
    ari: float
    nmi: float
    noise_fraction: float = 0.0  # HDBSCAN only; 0 for KMeans/Louvain


def scale_features(df: pd.DataFrame, features: list[str]) -> np.ndarray:
    return StandardScaler().fit_transform(df[features])


def run_kmeans(X_scaled: np.ndarray, class_code: pd.Series, k: int = N_CLASSES) -> ClusterResult:
    labels = KMeans(n_clusters=k, random_state=RANDOM_STATE, n_init=10).fit_predict(X_scaled)
    return ClusterResult(
        method="kmeans",
        n_clusters_found=k,
        ari=adjusted_rand_score(class_code, labels),
        nmi=normalized_mutual_info_score(class_code, labels),
    )


def run_hdbscan(
    X_scaled: np.ndarray, class_code: pd.Series, min_cluster_size: int = 30
) -> ClusterResult:
    labels = HDBSCAN(min_cluster_size=min_cluster_size).fit_predict(X_scaled)
    noise_fraction = float((labels == -1).mean())
    return ClusterResult(
        method="hdbscan",
        n_clusters_found=len(set(labels) - {-1}),
        ari=adjusted_rand_score(class_code, labels),
        nmi=normalized_mutual_info_score(class_code, labels),
        noise_fraction=noise_fraction,
    )


def run_knn_graph_louvain(
    X_scaled: np.ndarray, class_code: pd.Series, k: int = 10
) -> ClusterResult:
    """Builds a k-NN similarity graph (exact, via sklearn — the input
    here is the 100K-row sample, not the full 932K population, so exact
    NN is cheap and there's no need for faiss's approximate search) and
    runs Louvain community detection (igraph's community_multilevel).

    Note: unlike KMeans/HDBSCAN above, igraph's community_multilevel has
    no random_state parameter — its greedy merge order isn't seeded the
    same way sklearn's estimators are, so cluster count and ARI/NMI can
    shift slightly (single-digit percent) between runs. The qualitative
    finding (far more communities than real classes, low ARI) is stable
    across runs; the exact numbers are not bit-for-bit reproducible."""
    n = X_scaled.shape[0]
    nn = NearestNeighbors(n_neighbors=k + 1).fit(X_scaled)  # +1: includes self
    _, indices = nn.kneighbors(X_scaled)

    edges = [(i, j) for i in range(n) for j in indices[i] if j != i]
    g = ig.Graph(n=n, edges=edges)
    g.simplify()  # dedupe edges created from both directions of a mutual kNN pair

    communities = g.community_multilevel()
    labels = np.array(communities.membership)

    return ClusterResult(
        method="knn_graph_louvain",
        n_clusters_found=len(set(labels)),
        ari=adjusted_rand_score(class_code, labels),
        nmi=normalized_mutual_info_score(class_code, labels),
    )
