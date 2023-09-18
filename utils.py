import numpy as np
import torch
import matplotlib.pyplot as plt
import networkx as nx

from typing import Sequence

from graph_generator import generate_adjacency_matrix


def init_layer(layer, bias_const=0.0):
    torch.nn.init.kaiming_uniform_(layer.weight, nonlinearity='relu')
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer


def plot_graph_network(X: np.array, v: int, dest: int, t: int = 0):
    """
    Plot the graph network for visualization

    Parameters
    ----------
    X: np.array
        Graph matrix
    v: int
        Number of nodes in the graph
    dest: int
        Destination node
    t: int
        Time to wait before closing the plot window
    """
    # Check if the graph is too large to plot
    if v > 2:
        print("Graph too large to plot")
        return

    # Create an adjacency matrix of graph matrix X
    adj_mat = generate_adjacency_matrix(X)

    # ASCII offset for converting node numbers to alphabets
    ascii_offset = 65 if X.shape[0] <= 60 else 21 if X.shape[0] <= 100 else None

    # Create a list of directed edges associated with vertices vᵢ and vⱼ,
    # ∀ i, j ∈ {1, 2, ..., |V|}
    graph_edges = list()
    for i, row in enumerate(adj_mat):
        # Add a special character to the destination node for visualization
        from_special_char = '*' if i == dest else ''
        for j, val in enumerate(row):
            if val == 1:
                to_special_char = '*' if j == dest else ''
                if ascii_offset is None:
                    graph_edges.append((str(i) + from_special_char,
                                        str(j) + to_special_char))
                else:
                    graph_edges.append(
                        (chr(ascii_offset + i) + from_special_char,
                         chr(ascii_offset + j) + to_special_char))

    # Create a directed graph, with the associated edges
    G = nx.DiGraph()
    G.add_edges_from(graph_edges)

    # Plot the graph for visualization
    plt.figure(figsize=(8, 8))
    nx.draw(G, with_labels=True, node_size=1000, connectionstyle='arc3, rad = 0.5')
    plt.waitforbuttonpress(t)
    plt.close()
    return


def plot_all(axs, train_episodes: Sequence[int], train_loss: Sequence[float],
             train_reward: Sequence[float], eval_episodes: Sequence[int],
             eval_reward: Sequence[float], train_episode_len: Sequence[int],
             eval_episode_len: Sequence[int], baseline: float = None,
             text: str = None, show: bool = False, save: bool = False,
             path: str = None) -> None:
    """
       Method for plotting learning curves during policy training.

       Training loss, normalized episodic training and evaluation rewards, and
       training and evaluation episode lengths are plotted.

       Parameters
       ----------
       axs:
           A list of subplots.
       train_episodes : Sequence[int]
           A list of episodes pertaining to training rewards.
       train_loss : Sequence[float]
           A list of training-losses.
       train_reward : Sequence[float]
           A list of training rewards.
       eval_episodes: Sequence[int]
           A list of episodes pertaining to evaluation rewards.
       eval_reward : Sequence[float]
           A list of evaluation rewards.
       train_episode_len: Sequence[int]
           A list of episode lengths pertaining to training rewards.
       eval_episode_len: Sequence[int]
           A list of episode lengths pertaining to evaluation rewards.
       baseline: float
           Baseline reward for the evaluation reward.
       text: str
           Text to be displayed on the plot.
       show: bool
           Whether to show the plot or not.
       save: bool
           Whether to save the plot or not.
       path: str
           Path to save the plot.
    """

    # Training Loss plot
    axs[0].clear()
    axs[0].plot(train_episodes, train_loss, color='red', label='Train')
    axs[0].set(title='Training Loss')
    axs[0].set(ylabel='Loss')
    axs[0].set(xlabel='Episode')
    axs[0].legend(loc='upper right')

    # Normalized episodic reward of the policy during training and evaluation
    axs[1].clear()
    axs[1].plot(train_episodes, train_reward, color='red', label='Train')
    axs[1].plot(eval_episodes, eval_reward, color='blue', label='Eval')
    if baseline is not None:
        axs[1].plot(eval_episodes, [baseline] * len(eval_episodes),
                    color='black', label='DP-Baseline')
    axs[1].set(title='Normalized Episode Reward')
    axs[1].set(ylabel='Normalized Reward')
    axs[1].set(xlabel='Episode')
    axs[1].legend(loc='lower right')

    # Episode lengths
    axs[2].clear()
    axs[2].plot(train_episodes, train_episode_len, label='Train Episode Len.',
                color='red')
    axs[2].plot(eval_episodes, eval_episode_len, color='blue', label='Eval Episode Len.')
    axs[2].set(title="Episode Length")
    axs[2].set(ylabel='Steps')
    axs[2].set(xlabel='Episode')
    axs[2].legend(loc='upper right')

    # Add text to the plot
    if text is not None:
        x_min = axs[0].get_xlim()[0]
        y_max = axs[0].get_ylim()[1]
        axs[0].text(x_min * 1.0, y_max * 1.2, text, fontsize=14, color='Black')

    # Save the plot and data to file
    if save:
        plt.savefig(path + "plots/learning_curves.png")

        with open(path + 'plots/train_episodes.npy', 'wb') as f:
            np.save(f, np.array(train_episodes))

        with open(path + 'plots/train_loss.npy', 'wb') as f:
            np.save(f, np.array(train_loss))

        with open(path + 'plots/train_reward.npy', 'wb') as f:
            np.save(f, np.array(train_reward))

        with open(path + 'plots/eval_episodes.npy', 'wb') as f:
            np.save(f, np.array(eval_episodes))

        with open(path + 'plots/eval_reward.npy', 'wb') as f:
            np.save(f, np.array(eval_reward))

        with open(path + 'plots/train_episode_len.npy', 'wb') as f:
            np.save(f, np.array(train_episode_len))

        with open(path + 'plots/eval_episode_len.npy', 'wb') as f:
            np.save(f, np.array(eval_episode_len))

    if show:
        plt.show(block=False)
        plt.pause(0.01)


def bellman_ford(graph: np.array, trans_prob: np.array, num_nodes: int,
                 destination_node: int, starting_nodes: Sequence[int],
                 lambda_: float = 1.0, verbose: bool = False) -> (
        Sequence[float], dict):
    """
    The Bellman-Ford algorithm for finding the shortest path from each valid
    starting node to the destination node

    Parameters
    ----------
    graph: np.array
        Graph matrix
    trans_prob: np.array
        Transition probability matrix
    num_nodes:
        Number of nodes in the graph
    destination_node: int
        The destination node
    starting_nodes: Sequence[int]
        List of valid starting nodes
    lambda_: float
        The lambda value for the weighted sum of expected distance and expected
        time
    verbose: bool
        Whether to print the shortest path or not

    Returns
    -------
    Sequence[float]:
        A list of distances from each valid starting node to the destination
        node
    dict:
        A dictionary of nodes with the shortest path from the source node
    """
    # Remove self edges from the adjacency matrix for convenience
    np.fill_diagonal(graph, 0)

    # Fill all the negative edges with 0
    graph = np.where(graph < 0, 0, graph)

    # Calculate total number of directed edges in the graph
    num_edges = np.count_nonzero(graph)
    print(f"Number of edges: {num_edges}")

    # Get all pairs of nodes that are connected by an edge
    edge_pairs = np.argwhere(graph > 0).tolist()

    # Convert numpy array to list of tuples
    edge_pairs = [tuple(edge_pair) for edge_pair in edge_pairs]

    # Create a dictionary of nodes for storing the shortest path to the
    # destination node
    shortest_paths = dict()

    # Create a list of distances from each valid starting node to the
    # destination node
    dist_list = list()

    # ASCII offset for converting node numbers to alphabets for printing
    ascii_offset = 65 if num_nodes <= 60 else 21 if num_nodes <= 100 else None

    # Find the shortest path starting from each valid starting node to all
    # other nodes
    for starting_node in starting_nodes:
        # Create a dictionary of nodes
        nodes = dict()

        # Create a dictionary of nodes with the shortest path from the
        # source node
        paths = dict()

        # Initialize all nodes with infinity distance and empty path
        for node in range(num_nodes):
            nodes[node] = 0 if node == starting_node else np.inf
            paths[node] = [starting_node] if node == starting_node else []

        # Relax all edges |V| - 1 times
        for k in range(num_nodes - 1):
            # Relax each edge
            for edge_pair in edge_pairs:
                # Get the nodes connected by the edge
                u, v = edge_pair[0], edge_pair[1]

                # Get the distance of the nodes
                du, dv = nodes[u], nodes[v]

                # The distance of the destination node is just the weight of
                # the edge
                dist = graph[u, v]

                # The expected time to traverse the edge from the source
                # node (u) to the destination node (v) is the edge weight
                # divided by the transition probability from u to v
                expt_time = graph[u, v] * (1 / trans_prob[u, v])

                # The distance of the destination node is the minimum of the
                # current distance and the weighted sum of the distance and
                # the time taken to traverse the edge from the source node to
                # the destination node
                new_dist = \
                    du + (lambda_ * expt_time) + ((1 - lambda_) * dist)

                # If the new distance is less than the current distance,
                # update the distance and the path
                if new_dist < dv:
                    nodes[v] = new_dist
                    paths[v] = paths[u] + [v]

        if verbose:
            # Print the shortest path from each valid starting node to the
            # destination node, along with the number of hops and the total
            # distance
            from_node = str(starting_node) if ascii_offset is None else (
                chr(ascii_offset + starting_node))
            to_node = str(destination_node) if ascii_offset is None else (
                chr(ascii_offset + destination_node))
            path_string = str([' -> '.join([
                str(n) if ascii_offset is None else chr(ascii_offset + n) for n
                in paths[destination_node]])])

            print(f"Shortest path from {from_node} to {to_node}: " +
                  f"{path_string} -- Length: " +
                  f"{np.round(nodes[destination_node], 4)} " +
                  f"({len(paths[destination_node]) - 1} hops)")

        # Store the distance and the shortest path from the source node
        shortest_paths[starting_node] = paths[destination_node]
        dist_list.append(nodes[destination_node])

    # Return the list of distances and the dictionary of shortest paths
    return dist_list, shortest_paths