import gymnasium as gym
import numpy as np
import sys

from typing import Sequence, Optional, Union, Mapping

from graph_generator import generate_random_graph, validate_graph
from utils import plot_graph_network


class GraphEnv(gym.Env):
    """
        Custom Gym GraphEnv Class.

        A custom gym Env class representing a Bidirectional Graph for
        shortest path problem.

        Attributes
        ----------
        num_node : int
            The number of nodes/vertices in the graph

        destination_node : int
            The destintion node
    """

    def __init__(self, num_nodes: int, destination_node: int, bi_prob: float):

        if destination_node < 0 or destination_node >= num_nodes:
            print("Error: destination_node should be 0 >= destination_node < num_node")
            sys.exit()

        self.num_nodes = num_nodes
        self.episode_length = num_nodes

        self.destination_node = destination_node

        # Generate a random directed graph with n nodes, represented as matrix X
        self.graph = generate_random_graph(num_nodes, bi_prob)

        # Check if the generated graph is valid (traversable)
        valid_graph, valid_source_nodes = validate_graph(self.graph, num_nodes)

        while not valid_graph:
            # Regenerate graph, if not valid
            self.graph = generate_random_graph(num_nodes, bi_prob)

            # Check if the generated graph is valid (traversable)
            valid_graph, valid_source_nodes = validate_graph(self.graph, num_nodes)

        self.possible_starting_nodes = valid_source_nodes

        if self.destination_node in self.possible_starting_nodes:
            # Remove destination-node from the list of possile staring-nodes
            self.possible_starting_nodes.remove(self.destination_node)

        # Plot the graph for visualization
        plot_graph_network(X=self.graph, v=num_nodes, dest=destination_node)

        obs_low = np.array([-1] * num_nodes) / 10
        obs_high = np.array([10] * num_nodes) / 10

        # Observation space defined as a 1d vector, where -1 represents no edge, and a
        # positive value represents the weights to a connected node
        self.observation_space = \
            gym.spaces.Box(shape=(num_nodes,), low=obs_low, high=obs_high)

        # An action represents another node to transition to
        self.action_space = gym.spaces.Discrete(num_nodes)

    def reset(self, starting_node: int = None, seed: Optional[int] = None
              ) -> Sequence[Union[np.array, Mapping[str, Union[np.array, str]]]]:
        """
           Wrapper method for performing reset of the simulation environment.

           This method sets a starting node, and resets other environment variables.

           Parameters
           ----------
           seed : int
               An integer value to initialize the random initial distribution ρ₀.

           Returns
           -------
           np.array
               An array of observations
            Mapping[str, Union[np.array, str]]
               A dict object with info pertaining to the environment at
               the current time-step - action_mask, and current_node
        """

        super().reset()

        if seed is not None:
            pass

        if starting_node is None:
            # Reset starting node of an episode to any random node in the graph
            # excluding the destination node
            self.current_node = np.random.choice(self.possible_starting_nodes)
        else:  # User defined starting-node
            if starting_node not in self.possible_starting_nodes:
                print(f"Error: In env.reset(), starting_node: {starting_node} should" +
                      " be in the list of possible_starting_nodes: " +
                      f"{self.possible_starting_nodes}")
                sys.exit()

            self.current_node = starting_node

        # Reset the episode step counter
        self.episode_step = 0

        # Get observation as the current state of the graph
        observation = self.graph[self.current_node] / 10

        # Create action-mask to indicate valid actions from the current state
        info = {'action_mask': np.array(observation >= 0, dtype=int),
                'current_node': self.current_node}

        return observation, info

    def step(self, action) -> Sequence[Union[np.array, float, bool,
                                       Mapping[str, Union[np.array, str]]]]:
        """
           Wrapper method for performing one step through the simulation environment.

           Parameters
           ----------
           action : int
               The agent's action for the current time-step.

           Returns
           -------
           np.array
               An array of observations
           float
               The scalar reward value
           bool
               True if the episode has terminated
           bool
               True if the end-of-episode has been reached
           Mapping[str, Union[np.array, str]]
               A dict object with info pertaining to the environment at
               the current time-step.
        """

        # Get the distance/weight of the edge between the current node and the
        # destination-node
        distance = self.graph[self.current_node, action]

        if distance < 0:
            # Trying to travers through a non-existing edge
            terminated = True
            reward = -self.num_nodes * 10
        else:
            # Transitions to the destination-node
            self.current_node = action
            terminated = False
            reward = -distance

        # Get the new observation
        observation = self.graph[self.current_node] / 10

        # Increment the episode by one step
        self.episode_step += 1

        terminated = False
        truncated = False

        # Check if the end-of-episode has been reached
        if self.episode_step >= self.episode_length:
            if self.current_node == self.destination_node:
                # Reached destintion node on the last step
                reward += self.num_nodes * 10
                terminated = True
            else:
                # Reached end of episode without reaching destination-node
                reward += -self.num_nodes * 10
                truncated = True
        elif self.current_node == self.destination_node:
            # Reached destintion-node before end-of-episode
            terminated = True
            reward += self.num_nodes * 10

        # Create action-mask to indicate valid actions from the current state
        info = {'action_mask': np.array(observation >= 0, dtype=int),
                'current_node': self.current_node}

        return observation, reward, terminated, truncated, info