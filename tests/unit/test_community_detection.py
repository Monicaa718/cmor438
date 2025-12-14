import numpy as np
import pytest

from rice_ml.unsupervised_learning.community_detection import (
    CommunityDetection,
    LabelPropagation,
    calculate_modularity
)


# ------------------------ Louvain/CommunityDetection ------------------------

def test_louvain_simple_two_cliques():
    """Test Louvain on two disconnected cliques (trivial case)."""
    # Two triangles: [0,1,2] and [3,4,5]
    adj = np.array([
        [0, 1, 1, 0, 0, 0],
        [1, 0, 1, 0, 0, 0],
        [1, 1, 0, 0, 0, 0],
        [0, 0, 0, 0, 1, 1],
        [0, 0, 0, 1, 0, 1],
        [0, 0, 0, 1, 1, 0]
    ], dtype=float)
    
    cd = CommunityDetection()
    cd.fit(adj, max_iterations=10)
    
    communities = cd.predict()
    n_communities = cd.get_n_communities()
    
    # Should detect 2-6 communities (may split triangles further)
    assert 2 <= n_communities <= 6
    
    # Key property: nodes in triangle [0,1,2] should not share community with [3,4,5]
    group1_communities = {communities[0], communities[1], communities[2]}
    group2_communities = {communities[3], communities[4], communities[5]}
    
    # The two triangles should be in separate groups
    assert len(group1_communities.intersection(group2_communities)) == 0
    
    # Modularity should be positive for separated communities
    mod = cd.get_modularity()
    assert mod > 0  # Positive modularity indicates some structure


def test_louvain_single_node():
    """Test Louvain with single isolated node."""
    adj = np.array([[0]], dtype=float)
    
    cd = CommunityDetection()
    cd.fit(adj)
    
    communities = cd.predict()
    assert len(communities) == 1
    assert communities[0] == 0


def test_louvain_empty_graph():
    """Test Louvain with nodes but no edges."""
    adj = np.zeros((4, 4), dtype=float)
    
    cd = CommunityDetection()
    cd.fit(adj)
    
    # Each node should be in its own community
    communities = cd.predict()
    n_communities = cd.get_n_communities()
    
    assert n_communities == 4
    assert len(set(communities.values())) == 4


def test_louvain_complete_graph():
    """Test Louvain on complete graph (all connected)."""
    n = 5
    adj = np.ones((n, n), dtype=float)
    np.fill_diagonal(adj, 0)
    
    cd = CommunityDetection()
    cd.fit(adj)
    
    communities = cd.predict()
    
    # Should have 1-2 communities (complete graph has no clear structure)
    unique_communities = set(communities.values())
    assert 1 <= len(unique_communities) <= 2


def test_louvain_convergence():
    """Test that modularity improves during iterations."""
    # Create a simple graph with some structure
    adj = np.array([
        [0, 1, 1, 1, 0, 0],
        [1, 0, 1, 0, 0, 0],
        [1, 1, 0, 0, 0, 0],
        [1, 0, 0, 0, 1, 1],
        [0, 0, 0, 1, 0, 1],
        [0, 0, 0, 1, 1, 0]
    ], dtype=float)
    
    cd = CommunityDetection()
    cd.fit(adj, max_iterations=50)
    
    # Check that modularity history is recorded
    assert len(cd.modularity_history) > 0
    
    # Final modularity should be >= initial modularity
    assert cd.modularity_history[-1] >= cd.modularity_history[0]


def test_louvain_get_communities_as_lists():
    """Test converting community dict to list of lists."""
    adj = np.array([
        [0, 1, 1, 0, 0, 0],
        [1, 0, 1, 0, 0, 0],
        [1, 1, 0, 0, 0, 0],
        [0, 0, 0, 0, 1, 1],
        [0, 0, 0, 1, 0, 1],
        [0, 0, 0, 1, 1, 0]
    ], dtype=float)
    
    cd = CommunityDetection()
    cd.fit(adj)
    
    communities_list = cd.get_communities_as_lists()
    
    # Should have communities
    assert len(communities_list) > 0
    
    # All nodes should be accounted for
    all_nodes = []
    for comm in communities_list:
        all_nodes.extend(comm)
    assert sorted(all_nodes) == list(range(6))


def test_louvain_weighted_graph():
    """Test Louvain with weighted edges."""
    # Two groups with stronger internal connections
    adj = np.array([
        [0, 5, 5, 1, 0, 0],
        [5, 0, 5, 0, 0, 0],
        [5, 5, 0, 1, 0, 0],
        [1, 0, 1, 0, 5, 5],
        [0, 0, 0, 5, 0, 5],
        [0, 0, 0, 5, 5, 0]
    ], dtype=float)
    
    cd = CommunityDetection()
    cd.fit(adj)
    
    communities = cd.predict()
    n_communities = cd.get_n_communities()
    
    # Should detect some community structure (not all separate)
    assert 2 <= n_communities <= 4
    
    # At least nodes with strong weights should show some clustering
    # Check that we don't have all nodes in separate communities
    assert n_communities < 6


# ------------------------ Label Propagation ------------------------

def test_label_propagation_simple_two_cliques():
    """Test Label Propagation on two disconnected cliques."""
    adj = np.array([
        [0, 1, 1, 0, 0, 0],
        [1, 0, 1, 0, 0, 0],
        [1, 1, 0, 0, 0, 0],
        [0, 0, 0, 0, 1, 1],
        [0, 0, 0, 1, 0, 1],
        [0, 0, 0, 1, 1, 0]
    ], dtype=float)
    
    lp = LabelPropagation(max_iterations=100)
    lp.fit(adj)
    
    labels = lp.predict()
    n_communities = lp.get_n_communities()
    
    # Should detect 2 communities
    assert n_communities == 2
    
    # Nodes 0,1,2 should be in same community
    assert labels[0] == labels[1] == labels[2]
    
    # Nodes 3,4,5 should be in same community
    assert labels[3] == labels[4] == labels[5]
    
    # But different from the first group
    assert labels[0] != labels[3]


def test_label_propagation_single_node():
    """Test Label Propagation with single node."""
    adj = np.array([[0]], dtype=float)
    
    lp = LabelPropagation()
    lp.fit(adj)
    
    labels = lp.predict()
    assert len(labels) == 1
    assert labels[0] == 0


def test_label_propagation_empty_graph():
    """Test Label Propagation with nodes but no edges."""
    adj = np.zeros((4, 4), dtype=float)
    
    lp = LabelPropagation()
    lp.fit(adj)
    
    # Each node keeps its initial label (no neighbors to propagate from)
    labels = lp.predict()
    n_communities = lp.get_n_communities()
    
    assert n_communities == 4
    assert len(set(labels.values())) == 4


def test_label_propagation_complete_graph():
    """Test Label Propagation on complete graph."""
    n = 5
    adj = np.ones((n, n), dtype=float)
    np.fill_diagonal(adj, 0)
    
    lp = LabelPropagation(max_iterations=50)
    lp.fit(adj)
    
    labels = lp.predict()
    
    # All nodes should converge to same label
    unique_labels = set(labels.values())
    assert len(unique_labels) == 1


def test_label_propagation_get_communities_as_lists():
    """Test converting labels to list of lists."""
    adj = np.array([
        [0, 1, 1, 0, 0, 0],
        [1, 0, 1, 0, 0, 0],
        [1, 1, 0, 0, 0, 0],
        [0, 0, 0, 0, 1, 1],
        [0, 0, 0, 1, 0, 1],
        [0, 0, 0, 1, 1, 0]
    ], dtype=float)
    
    lp = LabelPropagation()
    lp.fit(adj)
    
    communities_list = lp.get_communities_as_lists()
    
    # Should have communities
    assert len(communities_list) > 0
    
    # All nodes should be accounted for
    all_nodes = []
    for comm in communities_list:
        all_nodes.extend(comm)
    assert sorted(all_nodes) == list(range(6))


def test_label_propagation_max_iterations():
    """Test that Label Propagation respects max_iterations."""
    adj = np.array([
        [0, 1, 1, 1, 1],
        [1, 0, 1, 1, 1],
        [1, 1, 0, 1, 1],
        [1, 1, 1, 0, 1],
        [1, 1, 1, 1, 0]
    ], dtype=float)
    
    # Should converge before max_iterations
    lp = LabelPropagation(max_iterations=2)
    lp.fit(adj)
    
    labels = lp.predict()
    assert len(labels) == 5


def test_label_propagation_weighted():
    """Test Label Propagation with weighted edges."""
    # Create graph where weights influence label propagation
    adj = np.array([
        [0, 10, 10, 1, 0, 0],
        [10, 0, 10, 0, 0, 0],
        [10, 10, 0, 1, 0, 0],
        [1, 0, 1, 0, 10, 10],
        [0, 0, 0, 10, 0, 10],
        [0, 0, 0, 10, 10, 0]
    ], dtype=float)
    
    lp = LabelPropagation()
    lp.fit(adj)
    
    labels = lp.predict()
    
    # Stronger weights should group nodes together
    # Should get 2 communities
    assert lp.get_n_communities() == 2


# ------------------------ Modularity Calculation ------------------------

def test_calculate_modularity_perfect_separation():
    """Test modularity calculation for perfectly separated communities."""
    adj = np.array([
        [0, 1, 1, 0, 0, 0],
        [1, 0, 1, 0, 0, 0],
        [1, 1, 0, 0, 0, 0],
        [0, 0, 0, 0, 1, 1],
        [0, 0, 0, 1, 0, 1],
        [0, 0, 0, 1, 1, 0]
    ], dtype=float)
    
    communities = {0: 0, 1: 0, 2: 0, 3: 1, 4: 1, 5: 1}
    
    mod = calculate_modularity(adj, communities)
    
    # Perfect separation should have high modularity
    assert mod > 0.4


def test_calculate_modularity_no_communities():
    """Test modularity when all nodes in same community."""
    adj = np.array([
        [0, 1, 1, 0],
        [1, 0, 1, 1],
        [1, 1, 0, 1],
        [0, 1, 1, 0]
    ], dtype=float)
    
    communities = {0: 0, 1: 0, 2: 0, 3: 0}
    
    mod = calculate_modularity(adj, communities)
    
    # All in one community typically has low modularity
    assert -1 <= mod <= 1


def test_calculate_modularity_random_assignment():
    """Test modularity with random community assignment."""
    adj = np.array([
        [0, 1, 1, 1],
        [1, 0, 1, 1],
        [1, 1, 0, 1],
        [1, 1, 1, 0]
    ], dtype=float)
    
    # Random assignment
    communities = {0: 0, 1: 1, 2: 0, 3: 1}
    
    mod = calculate_modularity(adj, communities)
    
    # Should be in valid range
    assert -1 <= mod <= 1


def test_calculate_modularity_empty_graph():
    """Test modularity calculation on empty graph."""
    adj = np.zeros((4, 4), dtype=float)
    communities = {0: 0, 1: 0, 2: 1, 3: 1}
    
    mod = calculate_modularity(adj, communities)
    
    # Empty graph has modularity of 0
    assert mod == 0.0


def test_calculate_modularity_weighted_graph():
    """Test modularity with weighted edges."""
    adj = np.array([
        [0, 5, 5, 1],
        [5, 0, 5, 1],
        [5, 5, 0, 1],
        [1, 1, 1, 0]
    ], dtype=float)
    
    # Nodes 0,1,2 together, node 3 separate
    communities = {0: 0, 1: 0, 2: 0, 3: 1}
    
    mod = calculate_modularity(adj, communities)
    
    # Should be in valid range (may not be positive for this particular division)
    assert -1 <= mod <= 1


# ------------------------ Edge Cases ------------------------

def test_louvain_symmetric_adjacency():
    """Test that Louvain handles symmetric adjacency matrices."""
    adj = np.array([
        [0, 1, 0],
        [1, 0, 1],
        [0, 1, 0]
    ], dtype=float)
    
    # Make sure it's symmetric
    assert np.allclose(adj, adj.T)
    
    cd = CommunityDetection()
    cd.fit(adj)
    
    communities = cd.predict()
    assert len(communities) == 3


def test_label_propagation_deterministic_with_seed():
    """Test that results are reproducible when using numpy random seed."""
    adj = np.array([
        [0, 1, 1, 1, 0, 0],
        [1, 0, 1, 0, 0, 0],
        [1, 1, 0, 0, 0, 0],
        [1, 0, 0, 0, 1, 1],
        [0, 0, 0, 1, 0, 1],
        [0, 0, 0, 1, 1, 0]
    ], dtype=float)
    
    # Run twice with same seed
    np.random.seed(42)
    lp1 = LabelPropagation(max_iterations=50)
    lp1.fit(adj)
    labels1 = lp1.predict()
    
    np.random.seed(42)
    lp2 = LabelPropagation(max_iterations=50)
    lp2.fit(adj)
    labels2 = lp2.predict()
    
    # Should get same results
    assert labels1 == labels2


def test_large_graph_performance():
    """Test that algorithms can handle moderately large graphs."""
    # Create a larger graph with 50 nodes
    n = 50
    adj = np.zeros((n, n), dtype=float)
    
    # Create 5 communities of 10 nodes each
    for i in range(5):
        start = i * 10
        end = start + 10
        for j in range(start, end):
            for k in range(start, end):
                if j != k:
                    adj[j, k] = 1
    
    # Add some inter-community edges
    for i in range(0, n-10, 10):
        adj[i, i+10] = 1
        adj[i+10, i] = 1
    
    # Test Louvain
    cd = CommunityDetection()
    cd.fit(adj, max_iterations=20)
    assert cd.get_n_communities() > 0
    
    # Test Label Propagation
    lp = LabelPropagation(max_iterations=20)
    lp.fit(adj)
    assert lp.get_n_communities() > 0


def test_communities_cover_all_nodes():
    """Test that all nodes are assigned to some community."""
    adj = np.array([
        [0, 1, 0, 0],
        [1, 0, 1, 0],
        [0, 1, 0, 1],
        [0, 0, 1, 0]
    ], dtype=float)
    
    # Test Louvain
    cd = CommunityDetection()
    cd.fit(adj)
    communities = cd.predict()
    assert len(communities) == 4  # All 4 nodes assigned
    
    # Test Label Propagation
    lp = LabelPropagation()
    lp.fit(adj)
    labels = lp.predict()
    assert len(labels) == 4  # All 4 nodes assigned
