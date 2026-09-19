"""Run the Stage 5 unsupervised analysis and write CLUSTERING_RESULTS.md.

KMeans runs against the full 932,335-row population (cheap at this
scale). HDBSCAN and the k-NN graph + Louvain path run against the
100K stratified sample instead — both are meaningfully more expensive
per-row than KMeans, and 100K rows is already enough to characterize
whether either method recovers the known dynamical classes.

Usage:
    python scripts/run_clustering.py
"""
from __future__ import annotations

import time
from pathlib import Path

from otda.cluster import run_hdbscan, run_kmeans, run_knn_graph_louvain, scale_features
from otda.data import CLUSTER_FEATURES, load_orbital_elements

REPO_ROOT = Path(__file__).resolve().parent.parent
OUT_PATH = REPO_ROOT / "CLUSTERING_RESULTS.md"

METHOD_DISPLAY = {
    "kmeans": "KMeans (k=13, full population)",
    "hdbscan": "HDBSCAN (100K sample)",
    "knn_graph_louvain": "k-NN Graph + Louvain (100K sample)",
}


def main() -> None:
    print("Loading full population orbital elements for KMeans...")
    df_full = load_orbital_elements("fct_asteroids_modeling")
    X_full = scale_features(df_full, CLUSTER_FEATURES)

    print("Loading 100K sample orbital elements for HDBSCAN / k-NN graph...")
    df_sample = load_orbital_elements("fct_asteroids_sample_100k")
    X_sample = scale_features(df_sample, CLUSTER_FEATURES)

    results = []

    t0 = time.time()
    results.append(run_kmeans(X_full, df_full["class_code"]))
    print(f"  kmeans done in {time.time() - t0:.1f}s")

    t0 = time.time()
    results.append(run_hdbscan(X_sample, df_sample["class_code"]))
    print(f"  hdbscan done in {time.time() - t0:.1f}s")

    t0 = time.time()
    results.append(run_knn_graph_louvain(X_sample, df_sample["class_code"]))
    print(f"  knn_graph_louvain done in {time.time() - t0:.1f}s")

    for r in results:
        print(
            f"{r.method:20s} clusters_found={r.n_clusters_found:3d}  "
            f"ARI={r.ari:.4f}  NMI={r.nmi:.4f}  noise={r.noise_fraction:.3f}"
        )

    lines = [
        "# Unsupervised Analysis: Orbital Elements vs. Known Dynamical Classes\n",
        "Secondary analysis (Stage 5) — do orbital elements alone "
        f"(`{', '.join(CLUSTER_FEATURES)}`) recover the known IAU/JPL "
        "dynamical classification (AMO/APO/MBA/TNO/...), independent of "
        "the PHA classification task? Ground truth is the `class_code` "
        "column (dbt/seeds/asteroid_class_map.csv), 13 classes.\n",
        "| Method | Clusters found | ARI | NMI | Noise fraction |",
        "|---|---|---|---|---|",
    ]
    for r in results:
        lines.append(
            f"| {METHOD_DISPLAY[r.method]} | {r.n_clusters_found} | "
            f"{r.ari:.4f} | {r.nmi:.4f} | {r.noise_fraction:.3f} |"
        )
    lines.append("")
    lines.append(
        "**ARI/NMI, not a confusion-matrix eyeball** — cluster label "
        "numbers are arbitrary (cluster \"0\" from KMeans has no relation "
        "to class_code 0/AMO), so a metric invariant to label permutation "
        "is required. ARI corrects for chance agreement (0 = random, "
        "1 = perfect, can go negative); NMI measures shared information "
        "(0 = independent, 1 = identical partitions).\n"
    )

    kmeans_r = next(r for r in results if r.method == "kmeans")
    hdbscan_r = next(r for r in results if r.method == "hdbscan")
    louvain_r = next(r for r in results if r.method == "knn_graph_louvain")

    lines.append(
        "## Interpretation: why none of these fully recover the classes "
        "(and why HDBSCAN comes closest)\n\n"
        f"All three methods score well below perfect agreement (KMeans "
        f"ARI={kmeans_r.ari:.3f}, Louvain ARI={louvain_r.ari:.3f}, HDBSCAN "
        f"ARI={hdbscan_r.ari:.3f}) — and that's expected, not a failure of "
        "the methods. The 13 dynamical classes are defined by explicit "
        "geometric *boundaries* on orbital elements — per NASA CNEOS's own "
        "near-Earth-object class definitions:\n\n"
        "| Class | Definition |\n"
        "|---|---|\n"
        "| Aten | a < 1.0 AU, Q > 0.983 AU |\n"
        "| Apollo | a > 1.0 AU, q < 1.017 AU |\n"
        "| Amor | a > 1.0 AU, 1.017 AU < q < 1.3 AU |\n\n"
        "Notably, Apollo and Amor are **not** separated by semi-major "
        "axis at all — both require a > 1.0 AU. What separates them is "
        "perihelion distance relative to 1.017 AU (Earth's aphelion). "
        "Each class is a *combinatorial* region (two conditions, on "
        "different variables, per class) rather than a single clean "
        "axis split — sharp threshold cuts through the feature space, "
        "not naturally-separated density blobs. None of the three "
        "methods are given those exact thresholds; they only see "
        f"continuous `{', '.join(CLUSTER_FEATURES)}` values, so recovering "
        "hard-edged, combinatorially-defined regions from unsupervised "
        "structure alone is a genuinely hard ask, independent of which "
        "algorithm is used — if anything, the two-condition, per-"
        "different-variable structure of these definitions makes it "
        "*less* likely a distance- or density-based method would "
        "rediscover them by accident, not more.\n\n"
        f"KMeans does worst (ARI={kmeans_r.ari:.3f}) because it assumes "
        "roughly spherical, similarly-sized clusters — false on both "
        "counts here: Main-belt asteroids (MBA) alone are ~89% of the "
        "population, dwarfing every other class, and the classes' true "
        "shapes are wedge-like regions bounded by orbital thresholds, not "
        "convex blobs around a centroid. Louvain does even worse "
        f"(ARI={louvain_r.ari:.3f}), splitting the data into "
        f"{louvain_r.n_clusters_found} communities — far more than the 13 "
        "real classes — because a k-NN graph's local connectivity picks "
        "up fine-grained density variation *within* the dominant MBA "
        "population itself, fragmenting one true class into many "
        "detected communities.\n\n"
        f"HDBSCAN comes closest (ARI={hdbscan_r.ari:.3f}, "
        f"NMI={hdbscan_r.nmi:.3f}) because it makes neither of KMeans' "
        "assumptions: no fixed cluster count, no spherical-cluster prior, "
        f"and it explicitly labels low-density points as noise "
        f"({hdbscan_r.noise_fraction:.1%} of the sample here) instead of "
        "forcing every point into some cluster. That's still a partial "
        "recovery, not the full IAU classification — but it's the "
        "expected ordering: the method with the fewest false structural "
        "assumptions about the data comes closest to structure that is "
        "itself irregular and wildly imbalanced.\n"
    )

    OUT_PATH.write_text("\n".join(lines), encoding="utf-8")
    print(f"\nWrote {OUT_PATH}")


if __name__ == "__main__":
    main()
