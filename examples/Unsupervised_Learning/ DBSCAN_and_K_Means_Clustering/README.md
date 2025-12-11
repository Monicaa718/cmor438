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

## This Folder
- Notebook: `DBSCAN_and_K_Means_Clustering.ipynb` implements preprocessing and both algorithms from scratch, runs them on synthetic standardized data, and compares results with simple internal metrics.

## How to Run
1. Open the notebook: `examples/Unsupervised_Learning/ DBSCAN_and_K_Means_Clustering/DBSCAN_and_K_Means_Clustering.ipynb`.
2. Run cells top-to-bottom in a Python 3 environment. No external clustering packages are required; only `numpy` is used.

## References
- Ester et al. (1996). A Density-Based Algorithm for Discovering Clusters in Large Spatial Databases with Noise.
- MacQueen (1967). Some Methods for classification and Analysis of Multivariate Observations.
This directory contains example code and notes for the DBSCAN and K-means clustering algorithm
in unsupervised learning.

## Algorithm

_TODO: Describe the core idea of DBSCAN and K-menas Clustering, its objective, and key hyperparameters._

## Data

_TODO: Describe the input features, labels (if any), and how datasets are loaded or preprocessed for DBSCAN and K-means Clustering._
