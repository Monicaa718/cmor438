# Community Detection

This directory contains example code and notes for the Community Detection algorithm in unsupervised learning.

## What is Community Detection?

Community detection is a fundamental problem in network analysis that aims to identify groups of nodes (vertices) that are more densely connected internally than with the rest of the network. In graph theory and network science, a **community** (also called a module or cluster) is a group of nodes that have dense connections within the group and sparse connections between groups.

### Applications

Community detection has broad applications across many domains:

- **Social Networks**: Identifying friend groups, interest communities, or influential clusters
- **Biology**: Finding protein complexes, gene modules, or functional groups in metabolic networks
- **Web Analysis**: Discovering web communities, topic clusters, or information flow patterns
- **Neuroscience**: Identifying brain modules and functional connectivity patterns
- **Marketing**: Customer segmentation and targeted advertising
- **Infrastructure**: Detecting critical components in power grids or transportation networks
- **Fraud Detection**: Identifying suspicious clusters of transactions or coordinated activities

## Algorithms

This implementation includes two popular community detection algorithms:

### 1. Louvain Method

The **Louvain algorithm** is a hierarchical clustering method that optimizes modularity through iterative local moves. It works in two phases:

**Phase 1 (Local Optimization):**
- Start with each node in its own community
- For each node, calculate the modularity gain from moving it to neighboring communities
- Move the node to the community that maximizes modularity gain
- Repeat until no improvement is possible

**Phase 2 (Network Aggregation):**
- Build a new network where nodes represent communities from Phase 1
- Repeat Phase 1 on this aggregated network

**Key Concept - Modularity:**

Modularity measures the quality of a division of a network into communities:

$$Q = \frac{1}{2m} \sum_{ij} \left[A_{ij} - \frac{k_i k_j}{2m}\right] \delta(c_i, c_j)$$

where:
- $m$ is the total number of edges
- $A_{ij}$ is the adjacency matrix (1 if connected, 0 otherwise)
- $k_i$ and $k_j$ are the degrees of nodes $i$ and $j$
- $\delta(c_i, c_j)$ is 1 if nodes $i$ and $j$ are in the same community, 0 otherwise

**Modularity ranges from -1 to 1**, with higher values indicating stronger community structure.

**Hyperparameters:**
- `max_iterations`: Maximum number of optimization iterations (default: 100)
- `tolerance`: Convergence threshold for modularity improvement (default: 1e-6)

**Advantages:**
- Fast and efficient (works well on large networks)
- Produces high-quality communities with high modularity
- Deterministic results

**Limitations:**
- Resolution limit: may miss small communities in large networks
- Tends to produce more granular communities

### 2. Label Propagation

The **Label Propagation algorithm** is a simpler, faster method based on information diffusion:

**Algorithm Steps:**
1. Initialize each node with a unique label (community ID)
2. Iteratively update each node's label to the most frequent label among its neighbors
3. Continue until labels stabilize or max iterations reached

**Hyperparameters:**
- `max_iterations`: Maximum number of propagation iterations (default: 100)

**Advantages:**
- Very simple and intuitive
- Extremely fast (near-linear time complexity)
- No parameters to tune (besides stopping criteria)

**Limitations:**
- Non-deterministic: different runs may produce different results
- Can produce unbalanced communities
- May not converge to optimal modularity

## Data

### Zachary's Karate Club Dataset

This example uses the famous **Zachary's Karate Club** network, one of the most well-known datasets in network science literature.

**Background:**
Wayne W. Zachary studied the social network of a university karate club from 1970 to 1972. During his observation, a conflict arose between the club's administrator ("Mr. Hi") and the instructor ("Officer"), which led to the club splitting into two separate groups. This makes it an ideal dataset for testing community detection algorithms because we know the ground truth.

**Network Properties:**
- **Nodes**: 34 members of the karate club
- **Edges**: 78 undirected edges representing social interactions between members
- **Network Type**: Undirected, unweighted social network
- **Ground Truth**: 2 communities (the two groups after the split)
- **Density**: 0.139 (fairly sparse network)
- **Average Clustering Coefficient**: 0.571 (high local clustering)

**Data Format:**
The dataset is represented as an **adjacency matrix** where:
- Rows and columns represent nodes (club members)
- Entry $A_{ij}$ = 1 if there is an edge between nodes $i$ and $j$
- Entry $A_{ij}$ = 0 if there is no edge
- The matrix is symmetric since the network is undirected

**Why This Dataset?**
1. **Ground truth available**: We can validate algorithm performance
2. **Small and interpretable**: Easy to visualize and understand
3. **Real-world data**: Represents actual social dynamics
4. **Well-studied**: Allows comparison with published results
5. **Challenging**: Not trivially separable, providing a good test case

**Data Loading:**
The dataset is loaded using NetworkX's built-in function:
```python
import networkx as nx
G = nx.karate_club_graph()
```

Then converted to an adjacency matrix for our custom algorithms:
```python
adjacency_matrix = nx.to_numpy_array(G)
```
