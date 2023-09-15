import numpy as np
import sys


def generate_adjacency_matrix(X):
    # Adjacency Matrix - Free travels are considered as valid edges
    adj_mat = np.array(X >= 0, dtype=int)

    return adj_mat


def generate_random_graph(v: int, edge_prob: float) -> np.array:
    """
    Generate a random graph with some nodes with uni-directional edges, and some
    nodes with bidirectional edges

    Parameters
    ----------
    v: int
        Number of nodes in the graph
    edge_prob: float
        Probability of an edge being present between two nodes

    Returns
    -------
    np.array:
        A random graph matrix of size [v x v]

    """
    # A random matrix of size [v x v] with positive and negative integers,
    # based on directed edge probability
    lower = int(edge_prob * 10) - 10
    upper = int(edge_prob * 10)
    X = np.random.choice(list(range(lower, upper)), size=(v, v), p=[0.1] * 10)

    # Create positive and negative masks from the generated matrix
    pos_mask = np.array(X >= 0, dtype=int)
    neg_mask = np.array(X < 0, dtype=int) * -1

    # Fill all the negative edges with -1 and positive edges with a random
    # integer between 1 and 9
    lower = 1
    upper = 10
    X = np.random.choice(list(range(lower, upper)), size=(v, v), p=[1 / 9] * 9)
    X = (X * pos_mask) + neg_mask

    # Make diagonals (self-edge) have maximum cost
    np.fill_diagonal(X, upper)

    return X


def validate_graph(X: np.array, v: int) -> (bool, list, list):
    """
    Check if the graph is valid (traversable)

    Parameters
    ----------
    X: nd.array
        Graph matrix
    v: int
        Number of nodes in the graph

    Returns
    -------
    bool:
        True if the graph is valid (traversable), False otherwise
    list:
        List of valid starting nodes
    list:
        List of invalid destination nodes
    """
    # Create an adjacency matrix of graph matrix X
    adj_mat = generate_adjacency_matrix(X)

    # Remove self edges from the adjacency matrix for convenience
    np.fill_diagonal(adj_mat, 0)

    # Calculate total number of undirected edges in the graph
    e = (np.sum(np.array((adj_mat + adj_mat.T) > 0, dtype=int))) / 2.0
    if int(e) != e:
        print("Error: ", int(e), " != ", e)
        sys.exit(2)
    else:
        e = int(e)
    print("Number of edges: ", e)

    # Finding the total number of paths from vertex vᵢ to vⱼ,
    # ∀ i, j ∈ {1, 2, ... v}:
    # Compute the number of unique paths from vertex vᵢ to vⱼ with exactly
    # n hops, and sum over all n ∈ {1, 2, ..., |V|-1}
    paths_sum = np.copy(adj_mat)  # n=1
    for i in range(2, v):
        # No. of unique paths from vertex vᵢ to vⱼ with exactly n hops = (Adj)ⁿ
        paths_sum += np.linalg.matrix_power(adj_mat, i)

    # Check if at least one path exists between vertices vᵢ and vⱼ with n hops,
    # ∀ i, j ∈ {1, 2, ... v}, and ∀ n ∈ {1, 2, ... v-1}
    path_bool = np.array(np.abs(paths_sum), dtype=bool)
    np.fill_diagonal(path_bool, True)
    path_exists = np.logical_or(path_bool, path_bool.T)

    non_dest_nodes = []

    # If there does not exist a path between any vertex pair {vᵢ, vⱼ},
    # then the graph is not traversable
    if not np.all(path_exists):
        return False, [], []
    else:
        # Find the list of possible starting-nodes
        zero_cols = np.array(np.abs(np.sum(paths_sum, axis=0)) > 0, dtype=int)
        zero_rows = np.array(np.abs(np.sum(paths_sum, axis=1)) > 0, dtype=int)

        if np.sum(zero_cols) < v:
            # In the path_matrix, if there exists a column Cᵢ = [0, 0, ..., 0]ᵀ,
            # then vertex vᵢ has no incoming edges, so vertex vᵢ cannot be the
            # destination node
            ind = np.argwhere(zero_cols == 0).flatten()
            if ind.size != 1:
                print("Error in Zero_Cols: ", zero_cols)
                sys.exit(1)
            print(f"Node: {ind[0] + 1} cannot be the destination node")
            non_dest_nodes = [ind[0]]

        if np.sum(zero_rows) < v:
            # In the path_matrix, if there exists a row Rᵢ = [0, 0, ..., 0],
            # then vertex vᵢ has no outgoing edges, so vertex vⱼ cannot be a
            # starting-node
            ind = np.argwhere(zero_rows == 0).flatten()
            if ind.size != 1:
                print("Error in Zero_Rows: ", zero_rows)
                sys.exit(1)
            print(f"Node: {chr(65 + ind[0])} cannot be a starting node")
            valid_source_nodes = list(range(v))
            del valid_source_nodes[ind[0]]

            return True, valid_source_nodes, non_dest_nodes

    print("All nodes in the graph are valid staring-nodes.")
    return True, list(range(v)), non_dest_nodes