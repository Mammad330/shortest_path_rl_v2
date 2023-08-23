import numpy as np
import matplotlib.pyplot as plt
import networkx as nx

from typing import Sequence

from graph_generator import generate_adjacency_matrix


def plot_graph_network(X, v: int, dest: int, t: int = 0):
    # Create an adjacency matrix of graph matrix X
    adj_mat = generate_adjacency_matrix(X)

    # Create a list of directed edges associated with vertices vᵢ and vⱼ,
    # ∀ i, j ∈ {1, 2, ..., |V|}
    graph_edges = list()
    for i, row in enumerate(adj_mat):
        from_special_char = '*' if i == dest else ''
        for j, val in enumerate(row):
            if val == 1:
                to_special_char = '*' if j == dest else ''
                # graph_edges.append((str(i + 1), str(j + 1)))
                graph_edges.append((chr(65 + i) + from_special_char,
                                    chr(65 + j) + to_special_char))

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
             eval_episode_len: Sequence[int], text: str = None, show: bool = False,
             save: bool = False, path: str = None, ) -> None:
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
    axs[1].plot(eval_episodes, eval_reward, color='blue', label='Evaluation')
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

    if text is not None:
        x_min = axs[0].get_xlim()[0]
        y_max = axs[0].get_ylim()[1]
        axs[0].text(x_min * 1.0, y_max * 1.2, text, fontsize=14, color='Black')

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