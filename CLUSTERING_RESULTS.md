# Unsupervised Analysis: Orbital Elements vs. Known Dynamical Classes

Secondary analysis (Stage 5) — do orbital elements alone (`a, e, i, moid, H`) recover the known IAU/JPL dynamical classification (AMO/APO/MBA/TNO/...), independent of the PHA classification task? Ground truth is the `class_code` column (dbt/seeds/asteroid_class_map.csv), 13 classes.

| Method | Clusters found | ARI | NMI | Noise fraction |
|---|---|---|---|---|
| KMeans (k=13, full population) | 13 | 0.0381 | 0.2329 | 0.000 |
| HDBSCAN (100K sample) | 10 | 0.4610 | 0.3979 | 0.150 |
| k-NN Graph + Louvain (100K sample) | 33 | 0.0114 | 0.1837 | 0.000 |

**ARI/NMI, not a confusion-matrix eyeball** — cluster label numbers are arbitrary (cluster "0" from KMeans has no relation to class_code 0/AMO), so a metric invariant to label permutation is required. ARI corrects for chance agreement (0 = random, 1 = perfect, can go negative); NMI measures shared information (0 = independent, 1 = identical partitions).

## Interpretation: why none of these fully recover the classes (and why HDBSCAN comes closest)

All three methods score well below perfect agreement (KMeans ARI=0.038, Louvain ARI=0.011, HDBSCAN ARI=0.461) — and that's expected, not a failure of the methods. The 13 dynamical classes are defined by explicit geometric *boundaries* on orbital elements — per NASA CNEOS's own near-Earth-object class definitions:

| Class | Definition |
|---|---|
| Aten | a < 1.0 AU, Q > 0.983 AU |
| Apollo | a > 1.0 AU, q < 1.017 AU |
| Amor | a > 1.0 AU, 1.017 AU < q < 1.3 AU |

Notably, Apollo and Amor are **not** separated by semi-major axis at all — both require a > 1.0 AU. What separates them is perihelion distance relative to 1.017 AU (Earth's aphelion). Each class is a *combinatorial* region (two conditions, on different variables, per class) rather than a single clean axis split — sharp threshold cuts through the feature space, not naturally-separated density blobs. None of the three methods are given those exact thresholds; they only see continuous `a, e, i, moid, H` values, so recovering hard-edged, combinatorially-defined regions from unsupervised structure alone is a genuinely hard ask, independent of which algorithm is used — if anything, the two-condition, per-different-variable structure of these definitions makes it *less* likely a distance- or density-based method would rediscover them by accident, not more.

KMeans does worst (ARI=0.038) because it assumes roughly spherical, similarly-sized clusters — false on both counts here: Main-belt asteroids (MBA) alone are ~89% of the population, dwarfing every other class, and the classes' true shapes are wedge-like regions bounded by orbital thresholds, not convex blobs around a centroid. Louvain does even worse (ARI=0.011), splitting the data into 33 communities — far more than the 13 real classes — because a k-NN graph's local connectivity picks up fine-grained density variation *within* the dominant MBA population itself, fragmenting one true class into many detected communities.

HDBSCAN comes closest (ARI=0.461, NMI=0.398) because it makes neither of KMeans' assumptions: no fixed cluster count, no spherical-cluster prior, and it explicitly labels low-density points as noise (15.0% of the sample here) instead of forcing every point into some cluster. That's still a partial recovery, not the full IAU classification — but it's the expected ordering: the method with the fewest false structural assumptions about the data comes closest to structure that is itself irregular and wildly imbalanced.
