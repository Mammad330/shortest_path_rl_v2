import numpy as np
import random
import time

from graph_generator import generate_random_graph, validate_graph
from utils import plot_graph_network, bellman_ford

def main():
    # Set random generator seed
    np.random.seed(SEED)
    random.seed(SEED)

    # ASCII offset for converting node numbers to alphabets for printing
    ascii_offset = 65 if NUM_NODES <= 60 else 21 if NUM_NODES <= 100 else None

    # Generate a random directed cyclical graph with n nodes
    graph = generate_random_graph(NUM_NODES, EDGE_PROB)

    # Check if the generated graph is valid (traversable)
    valid_graph, valid_source_nodes, non_dest_nodes = \
        validate_graph(graph, NUM_NODES)

    # Regenerate graph, if not valid or destination-node is not reachable
    while not valid_graph or DESTINATION_NODE in non_dest_nodes:
        # Regenerate graph, if not valid
        graph = generate_random_graph(NUM_NODES, EDGE_PROB)

        # Check if the generated graph is valid (traversable)
        valid_graph, valid_source_nodes, non_dest_nodes = validate_graph(
            graph, NUM_NODES)

    # Generate a random transition probability matrix for the graph, by
    # sampling from a uniform distribution
    trans_prob = np.random.uniform(TRANS_PROB_LOW, TRANS_PROB_HIGH,
                                   (NUM_NODES, NUM_NODES))

    # Check if the destination-node is in the list of possible starting nodes,
    # and remove if it is
    if DESTINATION_NODE in valid_source_nodes:
        valid_source_nodes.remove(DESTINATION_NODE)

    # Deep copy the graph matrix for later use
    graph_copy = np.copy(graph)

    start_time = time.time()

    # Using Bellman-Ford algorithm, find the shortest path from each valid
    # starting node to all other nodes
    distances, shortest_paths = bellman_ford(
        graph, trans_prob, NUM_NODES, DESTINATION_NODE, valid_source_nodes,
        LAMBDA)

    # Print the shortest path from each valid starting node to the destination
    # node, along with the number of hops and the total distance
    for i, starting_node in enumerate(valid_source_nodes):
        from_node = str(starting_node) if ascii_offset is None else (
            chr(ascii_offset + starting_node))
        to_node = str(DESTINATION_NODE) if ascii_offset is None else (
            chr(ascii_offset + DESTINATION_NODE))
        path_string = str([' -> '.join([chr(ascii_offset + n) for n in
                                        shortest_paths[starting_node]])])

        print(f"Shortest path from {from_node} to {to_node}: {path_string} " +
              f"-- Total Distance: {np.round(distances[i], 4)} " +
              f"({len(shortest_paths[starting_node]) - 1} hops)")

    # Calculate the expected average reward for reaching the destination node
    # from each valid starting node
    expecter_avg_reward = np.mean((NUM_NODES * 10) - np.array(distances))

    path_lengths = [len(v) - 1 for k, v in shortest_paths.items()]

    end_time = time.time()
    print("\nTime taken for Bellman-Ford: %.2f(s)" % (end_time - start_time))
    print(f"Expected Avg. Reward: {np.round(expecter_avg_reward, 4)}")
    print(f"Mean Distance to Destination: {np.round(np.mean(distances), 4)}")
    print("Median Distance to Destination: " +
          f"{np.round(np.median(distances), 4)}")
    print(f"Mean Number of Hops: {np.round(np.mean(path_lengths), 4)}")
    print(f"Median Number of Hops: {np.round(np.median(path_lengths), 4)}")

    # Plot the graph for visualization
    plot_graph_network(X=graph_copy, v=NUM_NODES, dest=DESTINATION_NODE)


if __name__ == '__main__':
    # Hyperparameters for graph generation
    SEED = 0
    NUM_NODES = 25
    DESTINATION_NODE = 20
    EDGE_PROB = 0.3
    TRANS_PROB_LOW = 0.1
    TRANS_PROB_HIGH = 0.2
    LAMBDA = 0.5

    main()