import gymnasium as gym
import numpy as np
import sys

from typing import Sequence, Optional, Union, Mapping

from graph_generator import generate_random_graph, validate_graph
from utils import plot_graph_network


class GraphEnv(gym.Env):
    """
        Custom Gym GraphEnv Class.

        A custom gym Env class representing a Bidirectional Graph for the
        shortest path problem.

        Attributes
        ----------
        num_nodes : int
            The number of nodes/vertices in the graph
        destination_node : int
            The destination node
        edge_prob : float
            The probability of an edge from a node to another node in
            each direction
        trans_prob_low : float
            The lower bound of the transition probability
        trans_prob_high : float
            The upper bound of the transition probability
    """

    def __init__(self, num_nodes: int, destination_node: int, edge_prob: float,
                 trans_prob_low: float, trans_prob_high: float,
                 lambda_: float = 1.0):

        if destination_node < 0 or destination_node >= num_nodes:
            print("Error: destination_node should be 0 >= destination_node " +
                  "< num_node")
            sys.exit()

        # The number of nodes/vertices in the graph
        self.num_nodes = num_nodes

        # The maximum number of steps in an episode. This is set to the number
        # of nodes in the graph, as the shortest path between any two nodes
        # cannot be more than the number of nodes in the graph
        self.episode_length = num_nodes

        # The fixed destination node for all episodes
        self.destination_node = destination_node

        # The lambda value for the reward function, to control the trade-off
        # between distance travelled and travel time
        self.lambda_ = lambda_

        # Variables to keep track of the current state of the environment
        self.current_node = None
        self.episode_step = 0

        # Generate a random directed cyclical graph with n nodes
        self.graph = generate_random_graph(num_nodes, edge_prob)

        # Check if the generated graph is valid (traversable)
        valid_graph, valid_source_nodes, non_dest_nodes = \
            validate_graph(self.graph, num_nodes)

        # Regenerate graph, if not valid or destination-node is not reachable
        while not valid_graph or self.destination_node in non_dest_nodes:
            # Regenerate graph, if not valid
            self.graph = generate_random_graph(num_nodes, edge_prob)

            # Check if the generated graph is valid (traversable)
            valid_graph, valid_source_nodes, non_dest_nodes = validate_graph(
                self.graph, num_nodes)

        # List of possible starting-nodes, from which to start an episode from
        self.possible_starting_nodes = valid_source_nodes

        # Check if the destination-node is in the list of possible starting
        # nodes, and remove if it is
        if self.destination_node in self.possible_starting_nodes:
            self.possible_starting_nodes.remove(self.destination_node)

        # Generate a random transition probability matrix for the graph, by
        # sampling from a uniform distribution
        self.trans_prob = np.random.uniform(trans_prob_low, trans_prob_high,
                                            (num_nodes, num_nodes))

        # Plot the graph for visualization
        plot_graph_network(X=self.graph, v=num_nodes, dest=destination_node)

        # Define the observation space range
        obs_low = np.array([-1] * num_nodes) / 10
        obs_high = np.array([10] * num_nodes) / 10

        # Observation space defined as a 1d vector, where -1 represents no edge,
        # and a positive value represents the weights to a connected node.
        # Self edges are represented with the maximum weight of 10, while edges
        # to other nodes are in the range [1, 9]
        self.observation_space = \
            gym.spaces.Box(shape=(num_nodes,), low=obs_low, high=obs_high)

        # An action represents a node (including self) to transition to
        self.action_space = gym.spaces.Discrete(num_nodes)

    def reset(self, starting_node: int = None) -> Sequence[
            Union[np.array, Mapping[str, Union[np.array, str]]]]:
        """
           Wrapper method for performing reset of the simulation environment.

           This method sets a starting node, and resets other environment
           variables.

           Parameters
           ----------
           starting_node : int
               Optional starting node for the episode.

           Returns
           -------
           np.array
               An array of observations
            Mapping[str, Union[np.array, str]]
               A dict object with info pertaining to the environment at the
               current time-step: action_mask and the current_node
        """

        super().reset()

        if starting_node is None:
            # Reset starting node of an episode by randomly choosing form the
            # possible starting nodes list
            self.current_node = np.random.choice(self.possible_starting_nodes)
        else:  # User defined starting-node
            # Check if the user defined starting-node is in the list of
            # possible starting nodes
            if starting_node not in self.possible_starting_nodes:
                print(f"Error: In env.reset(), starting_node: {starting_node}" +
                      " should be in the list of possible_starting_nodes: " +
                      f"{self.possible_starting_nodes}")
                sys.exit()

            self.current_node = starting_node

        # Reset the episode step counter
        self.episode_step = 0

        # Get observation as the current state of the graph.
        # The observation is the weight of outgoing edges between the current
        # node and all other nodes in the graph, and normalized to the
        # range: [-1, 1]
        observation = self.graph[self.current_node] / 10

        # Create action-mask to indicate valid actions from the current state
        # i.e, valid nodes that the agent can transition to from the current
        # node
        info = {'action_mask': np.array(observation >= 0, dtype=int),
                'current_node': self.current_node}

        return observation, info

    def step(self, action, eval: bool = False) -> Sequence[
       Union[np.array, float, bool, Mapping[str, Union[np.array, str]]]]:
        """
           Wrapper method for performing one step through the simulation
           environment.

           Parameters
           ----------
           action : int
               The agent's action for the current time-step.
            eval : bool
                Flag to indicate if the environment is being used for evaluation
                or training.

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
               A dict object with info pertaining to the environment at the
               current time-step: action_mask and the current_node
        """

        # Get the distance/weight of the edge between the current node and the
        # node the agent intends to transition to
        distance = self.graph[self.current_node, action]

        # Travel time is the same as the distance, since we are considering
        # speed of travel as one unit of distance travelled per unit of time.
        travel_time = self.graph[self.current_node, action]

        if distance < 0:
            # The agent is trying to travers through a non-existing edge.
            # Terminate the episode and penalize the agent
            terminated = True
            reward = -self.num_nodes * 10
            print("EPISODE TERMINATED: Trying to travers through a " +
                  f"non-existing edge: {action}.")
            print("List of possible transition nodes: " +
                  f"{np.where(self.graph[self.current_node] > 0)[0].tolist()}")
        else:
            if eval:
                # The expected travel time is the travel time multiplied by the
                # inverse of the transition probability.
                # Since we are considering speed or travel as one unit of
                # distance travelled per unit of time, the travel time is the
                # same as the distance.
                expected_travel_time = distance * \
                    (1 / self.trans_prob[self.current_node, action])

                # The reward is modelled as a combination of expected travel
                # time and the expected distance travelled, based on the
                # transition probability
                reward = -((self.lambda_ * expected_travel_time) +
                           ((1 - self.lambda_) * distance))

                # During policy evaluation, transition to the next-node without
                # considering transition probability, but the reward is
                # calculated as the expected reward based on the transition
                # probability
                self.current_node = action
            else:
                # Transition to the next-node based on the transition
                # probability
                if np.random.rand(1) <= self.trans_prob[
                   self.current_node, action]:
                    # Transition to the next-node, if the transition probability
                    # is met
                    self.current_node = action
                else:
                    # Stay in the current node, if the transition probability
                    # is not met. Hence, the distance travelled is 0
                    distance = 0

                # The reward is modelled as a combination of travel time and
                # distance travelled. Even when the agent stays in the current
                # node (due to transition probability), the travel time is
                # considered, to emulate the additional time taken to traverse
                # through an edge on disruption
                reward = -((self.lambda_ * travel_time) +
                           ((1 - self.lambda_) * distance))

            terminated = False

        # Get the new observation as the current state of the graph.
        # The observation is the weight of outgoing edges between the current
        # node and all other nodes in the graph, and normalized to the
        # range: [-1, 1]
        observation = self.graph[self.current_node] / 10

        # Increment the episode by one step
        self.episode_step += 1
        truncated = False

        # Check if the end-of-episode has been reached
        if self.episode_step >= self.episode_length:
            if self.current_node == self.destination_node:
                # Reached destination node on the last step
                # Reward the agent for reaching the destination-node
                reward += self.num_nodes * 10
                terminated = True
            else:
                # Reached end of episode without reaching destination-node
                # Penalize the agent for not reaching the destination-node in
                # the episode
                reward += -self.num_nodes * 10
                truncated = True
        elif self.current_node == self.destination_node:
            # Reached destination-node before end-of-episode
            # Terminal the episode and reward the agent for reaching the
            # destination-node
            terminated = True
            reward += self.num_nodes * 10

        # Create action-mask to indicate valid actions from the current state
        # i.e, valid nodes that the agent can transition to from the current
        # node
        info = {'action_mask': np.array(observation >= 0, dtype=int),
                'current_node': self.current_node}

        return observation, reward, terminated, truncated, info