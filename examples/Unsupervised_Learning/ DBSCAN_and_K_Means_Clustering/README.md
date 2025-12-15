# DBSCAN and K-Means Clustering

## Overview
- DBSCAN: Density-Based Spatial Clustering of Applications with Noise. Discovers clusters of arbitrary shape by grouping points with sufficient local density and flags low-density points as noise.
- K-Means: Partitions data into `k` compact, spherical clusters by minimizing within-cluster variance via iterative center updates.

## Algorithm
- DBSCAN:
	- Core idea: For each point, find neighbors within radius `eps`. If neighbors count ≥ `min_samples`, it is a core point. Expand clusters by connecting density-reachable points.
	- Output: Cluster labels and noise points (label `-1`).
- K-Means:
	- Core idea: Initialize `k` centers, assign points to nearest center, update centers to mean of assigned points, and repeat until convergence.
	- Output: Cluster labels and final centers.

## Key Parameters
- DBSCAN:
	- `eps`: Neighborhood radius controlling density threshold.
	- `min_samples`: Minimum points to form a dense region.
- K-Means:
	- `k`: Number of clusters to discover.
	- `max_iters`, `tol`: Iteration cap and convergence tolerance.

## Complexity
- DBSCAN: Typically `O(n log n)` with spatial index; `O(n^2)` with naive distances.
- K-Means: `O(n * k * t)` where `t` is iterations; distance computations dominate.

## Strengths & Trade-offs
- DBSCAN:
	- Pros: Finds arbitrary-shaped clusters; identifies noise; no need to pre-set `k`.
	- Cons: Sensitive to `eps`/`min_samples`; struggles with varying densities.
- K-Means:
	- Pros: Simple, fast, scalable; good for compact, well-separated clusters.
	- Cons: Requires `k`; assumes spherical clusters; sensitive to initialization and scale.

## Data
- Source: Classic Iris dataset loaded via `sklearn.datasets.load_iris()`.
- Features: 4 numerical attributes per sample — `sepal length`, `sepal width`, `petal length`, `petal width` (all in centimeters).
- Samples: 150 total across three species (`setosa`, `versicolor`, `virginica`).
- Labels: Available in the dataset but not used by DBSCAN or K-Means during training; they can be used later for qualitative comparison.
- Preprocessing: Z-score standardization applied feature-wise: `X_std = (X - mean) / std`. This ensures both algorithms are scale-invariant and prevents any single feature from dominating distance calculations.
- Parameters used in notebook:
	- DBSCAN: `eps=0.6`, `min_samples=6` on standardized features.
	- K-Means: `k=3`, `max_iters=100`, `tol=1e-4`, with random seed set to `42` for reproducibility.
- Notes: Iris has overlap between `versicolor` and `virginica` in feature space. DBSCAN may merge these into one dense region depending on `eps`/`min_samples`, while K-Means partitions into `k=3` compact clusters regardless of overlap.
