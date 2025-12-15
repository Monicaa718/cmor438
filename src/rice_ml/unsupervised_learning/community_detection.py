"""
Community Detection Algorithm Implementation

This module implements the Louvain algorithm for community detection in networks.
The Louvain method is a hierarchical clustering algorithm that optimizes modularity.

Author: Rice ML Library
Date: December 2025
"""

import numpy as np
from collections import defaultdict
from typing import Dict, List, Set, Tuple


class CommunityDetection:
    """
    Community Detection using the Louvain algorithm.
    
    The Louvain method is an iterative algorithm that:
    1. Assigns each node to its own community
    2. For each node, evaluates modularity gain by moving it to neighbor communities
    3. Moves nodes to communities that maximize modularity gain
    4. Aggregates the graph based on communities
    5. Repeats until modularity cannot be improved
    
    Attributes:
        adjacency_matrix (np.ndarray): The adjacency matrix of the graph
        communities (Dict[int, int]): Mapping from node to community ID
        modularity_history (List[float]): History of modularity values
    """
    
    def __init__(self):
        """Initialize the CommunityDetection object."""
        self.adjacency_matrix = None
        self.communities = {}
        self.modularity_history = []
        self.best_modularity = -np.inf
        self.best_communities = {}
        
    def fit(self, adjacency_matrix: np.ndarray, max_iterations: int = 100, 
            tolerance: float = 1e-6) -> 'CommunityDetection':
        """
        Detect communities in the graph using the Louvain algorithm.
        
        Parameters:
            adjacency_matrix (np.ndarray): Adjacency matrix of the graph (n x n)
            max_iterations (int): Maximum number of iterations for the algorithm
            tolerance (float): Convergence tolerance for modularity improvement
            
        Returns:
            self: The fitted CommunityDetection object
        """
        self.adjacency_matrix = adjacency_matrix.copy()
        n_nodes = len(adjacency_matrix)
        
        # Initialize: each node in its own community
        self.communities = {i: i for i in range(n_nodes)}
        
        # Calculate total edge weight
        self.m = np.sum(adjacency_matrix) / 2.0  # Total edge weight (undirected graph)
        
        if self.m == 0:
            return self
        
        # Calculate node degrees
        self.degrees = np.sum(adjacency_matrix, axis=1)
        
        # Iterative optimization
        current_modularity = self._calculate_modularity()
        self.modularity_history.append(current_modularity)
        
        for iteration in range(max_iterations):
            improved = False
            
            # Phase 1: Move nodes to optimize modularity
            for node in range(n_nodes):
                # Find neighbors
                neighbors = np.where(adjacency_matrix[node] > 0)[0]
                
                if len(neighbors) == 0:
                    continue
                
                # Get neighboring communities
                neighbor_communities = set(self.communities[neighbor] 
                                         for neighbor in neighbors)
                
                current_community = self.communities[node]
                best_community = current_community
                best_gain = 0.0
                
                # Try moving to each neighboring community
                for community in neighbor_communities:
                    if community == current_community:
                        continue
                    
                    # Calculate modularity gain
                    gain = self._modularity_gain(node, community)
                    
                    if gain > best_gain:
                        best_gain = gain
                        best_community = community
                
                # Move node if improvement found
                if best_gain > tolerance:
                    self.communities[node] = best_community
                    improved = True
            
            # Calculate new modularity
            new_modularity = self._calculate_modularity()
            self.modularity_history.append(new_modularity)
            
            # Check convergence
            if new_modularity - current_modularity < tolerance:
                break
            
            current_modularity = new_modularity
            
            if not improved:
                break
        
        # Renumber communities to be sequential
        self._renumber_communities()
        
        # Store best result
        self.best_modularity = self._calculate_modularity()
        self.best_communities = self.communities.copy()
        
        return self
    
    def _modularity_gain(self, node: int, target_community: int) -> float:
        """
        Calculate the modularity gain from moving a node to a target community.
        
        Parameters:
            node (int): Node to move
            target_community (int): Target community
            
        Returns:
            float: Modularity gain
        """
        # Sum of weights of links from node to nodes in target community
        k_i_in = 0.0
        for neighbor in range(len(self.adjacency_matrix)):
            if self.communities[neighbor] == target_community:
                k_i_in += self.adjacency_matrix[node][neighbor]
        
        # Sum of all weights of links to nodes in target community
        sigma_tot = sum(self.degrees[j] for j in range(len(self.adjacency_matrix))
                       if self.communities[j] == target_community)
        
        # Node degree
        k_i = self.degrees[node]
        
        # Modularity gain formula
        gain = (k_i_in / self.m) - (sigma_tot * k_i / (2 * self.m * self.m))
        
        return gain
    
    def _calculate_modularity(self) -> float:
        """
        Calculate the modularity of the current community assignment.
        
        Modularity Q = (1/2m) * Σ[A_ij - (k_i * k_j)/(2m)] * δ(c_i, c_j)
        where:
        - m is the total edge weight
        - A_ij is the adjacency matrix
        - k_i, k_j are node degrees
        - δ(c_i, c_j) is 1 if nodes i and j are in the same community, 0 otherwise
        
        Returns:
            float: The modularity value
        """
        if self.m == 0:
            return 0.0
        
        Q = 0.0
        n_nodes = len(self.adjacency_matrix)
        
        for i in range(n_nodes):
            for j in range(n_nodes):
                if self.communities[i] == self.communities[j]:
                    A_ij = self.adjacency_matrix[i][j]
                    expected = (self.degrees[i] * self.degrees[j]) / (2 * self.m)
                    Q += A_ij - expected
        
        Q = Q / (2 * self.m)
        return Q
    
    def _renumber_communities(self):
        """Renumber communities to be sequential starting from 0."""
        unique_communities = sorted(set(self.communities.values()))
        community_map = {old: new for new, old in enumerate(unique_communities)}
        self.communities = {node: community_map[comm] 
                          for node, comm in self.communities.items()}
    
    def predict(self) -> Dict[int, int]:
        """
        Get the community assignments.
        
        Returns:
            Dict[int, int]: Dictionary mapping node IDs to community IDs
        """
        return self.communities.copy()
    
    def get_communities_as_lists(self) -> List[List[int]]:
        """
        Get communities as a list of lists.
        
        Returns:
            List[List[int]]: List where each element is a list of nodes in that community
        """
        communities_dict = defaultdict(list)
        for node, community in self.communities.items():
            communities_dict[community].append(node)
        
        return [nodes for nodes in communities_dict.values()]
    
    def get_modularity(self) -> float:
        """
        Get the final modularity value.
        
        Returns:
            float: The modularity of the detected communities
        """
        return self._calculate_modularity()
    
    def get_n_communities(self) -> int:
        """
        Get the number of detected communities.
        
        Returns:
            int: Number of communities
        """
        return len(set(self.communities.values()))


class LabelPropagation:
    """
    Community Detection using Label Propagation algorithm.
    
    Label Propagation is a simple algorithm where:
    1. Each node is initialized with a unique label
    2. Iteratively, each node adopts the label most common among its neighbors
    3. Process continues until convergence
    
    Attributes:
        labels (Dict[int, int]): Mapping from node to community label
    """
    
    def __init__(self, max_iterations: int = 100):
        """
        Initialize the LabelPropagation object.
        
        Parameters:
            max_iterations (int): Maximum number of iterations
        """
        self.max_iterations = max_iterations
        self.labels = {}
        
    def fit(self, adjacency_matrix: np.ndarray) -> 'LabelPropagation':
        """
        Detect communities using label propagation.
        
        Parameters:
            adjacency_matrix (np.ndarray): Adjacency matrix of the graph
            
        Returns:
            self: The fitted LabelPropagation object
        """
        n_nodes = len(adjacency_matrix)
        
        # Initialize: each node gets its own label
        self.labels = {i: i for i in range(n_nodes)}
        
        # Iterative label propagation
        for iteration in range(self.max_iterations):
            changed = False
            
            # Random order to avoid bias
            nodes = list(range(n_nodes))
            np.random.shuffle(nodes)
            
            for node in nodes:
                # Find neighbors
                neighbors = np.where(adjacency_matrix[node] > 0)[0]
                
                if len(neighbors) == 0:
                    continue
                
                # Count labels among neighbors
                label_counts = defaultdict(float)
                for neighbor in neighbors:
                    weight = adjacency_matrix[node][neighbor]
                    label_counts[self.labels[neighbor]] += weight
                
                # Find most common label
                if label_counts:
                    new_label = max(label_counts.items(), key=lambda x: x[1])[0]
                    
                    if new_label != self.labels[node]:
                        self.labels[node] = new_label
                        changed = True
            
            # Check convergence
            if not changed:
                break
        
        # Renumber labels to be sequential
        self._renumber_labels()
        
        return self
    
    def _renumber_labels(self):
        """Renumber labels to be sequential starting from 0."""
        unique_labels = sorted(set(self.labels.values()))
        label_map = {old: new for new, old in enumerate(unique_labels)}
        self.labels = {node: label_map[label] 
                      for node, label in self.labels.items()}
    
    def predict(self) -> Dict[int, int]:
        """
        Get the community assignments.
        
        Returns:
            Dict[int, int]: Dictionary mapping node IDs to community IDs
        """
        return self.labels.copy()
    
    def get_communities_as_lists(self) -> List[List[int]]:
        """
        Get communities as a list of lists.
        
        Returns:
            List[List[int]]: List where each element is a list of nodes in that community
        """
        communities_dict = defaultdict(list)
        for node, label in self.labels.items():
            communities_dict[label].append(node)
        
        return [nodes for nodes in communities_dict.values()]
    
    def get_n_communities(self) -> int:
        """
        Get the number of detected communities.
        
        Returns:
            int: Number of communities
        """
        return len(set(self.labels.values()))


def calculate_modularity(adjacency_matrix: np.ndarray, 
                        communities: Dict[int, int]) -> float:
    """
    Calculate modularity for a given community assignment.
    
    Modularity measures the quality of a division of a network into communities.
    Q = (1/2m) * Σ[A_ij - (k_i * k_j)/(2m)] * δ(c_i, c_j)
    
    Parameters:
        adjacency_matrix (np.ndarray): Adjacency matrix of the graph
        communities (Dict[int, int]): Community assignments
        
    Returns:
        float: Modularity value (range: -1 to 1, higher is better)
    """
    m = np.sum(adjacency_matrix) / 2.0  # Total edge weight
    
    if m == 0:
        return 0.0
    
    degrees = np.sum(adjacency_matrix, axis=1)
    Q = 0.0
    n_nodes = len(adjacency_matrix)
    
    for i in range(n_nodes):
        for j in range(n_nodes):
            if communities[i] == communities[j]:
                A_ij = adjacency_matrix[i][j]
                expected = (degrees[i] * degrees[j]) / (2 * m)
                Q += A_ij - expected
    
    Q = Q / (2 * m)
    return Q
