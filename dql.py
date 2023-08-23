import numpy as np
import gymnasium as gym
import time
import os
import matplotlib.pyplot as plt

from typing import Sequence, Union

import torch
import torch.nn as nn
import torch.optim as optim

from utils import plot_all


class ReplayMemory():
    """
        A class for storing and sampling transitions for Experience Replay.

        This class creates a memory buffer of a predetermined size for storing and
        sampling batch-sized transitions {sₜ, aₜ, rₜ, sₜ₊₁} during training.
    """

    def __init__(self, mem_size: int, state_size: int, num_actions: int, device):
        """
           The init() method creates and initializes memory buffers for storing the
           transitions. It also initializes counters for indexing the array for
           roll-over transition insertion and sampling.

           Parameters
           ----------
           mem_size : int
               The maximum size of the replay memory buffer.
           state_size : int
               The feature size of the observations.
        """

        self.mem_size = mem_size
        self.device = device
        self.mem_count = 0
        self.current_index = 0

        self.states = np.zeros((mem_size, state_size), dtype='f8')
        self.actions = np.zeros((mem_size,), dtype='i4')
        self.rewards = np.zeros((mem_size,), dtype='f8')
        self.terminals = np.zeros((mem_size,), dtype='?')
        self.action_masks = np.zeros((mem_size, num_actions), dtype='i4')

    def add(self, state: np.array, action: int, reward: float, terminal: bool,
            action_masks: np.array) -> None:
        """
           Method for inserting a transition {sₜ, aₜ, rₜ, sₜ₊₁} to the
           replay memory buffer.

           Parameters
           ----------
           state : np.array
               An array of obsertations from the environment.
           action : int
               The action taken by the agent.
           reward : float
               The observed reward.
           terminal : bool
                A boolean indicating if state sₜ is a terminal state or not.
        """

        self.states[self.current_index % self.mem_size] = state
        self.actions[self.current_index % self.mem_size] = action
        self.rewards[self.current_index % self.mem_size] = reward
        self.terminals[self.current_index % self.mem_size] = terminal
        self.action_masks[self.current_index % self.mem_size] = action_masks

        self.current_index = (self.current_index + 1) % self.mem_size
        self.mem_count = max(self.mem_count, self.current_index)

    def sample_batch(self, batch_size: int = 32) -> Sequence[np.array]:
        """
           Method for randomly sampling transitions {s, a, r, s'} of batch_size from
           the replay memory buffer.

           Parameters
           ----------
           batch_size : int
               Number of transitions to be sampled.

           Returns
           -------
           Sequence[np.array]
               A list of arrays each containing the sampled states, actions, rewards,
               terminal booleans, and the respective next-states (s').
        """

        while True:
            sampled_idx = \
                np.random.choice(self.mem_count, size=batch_size, replace=False)

            if (sampled_idx == self.current_index - 1).any():
                # Resample if any sampled transition is the most recently recorded
                # transition
                continue
            break

        return (
            torch.tensor(self.states[sampled_idx], dtype=torch.float32).to(self.device),
            torch.tensor(self.actions[sampled_idx], dtype=torch.int64).to(self.device),
            torch.tensor(self.rewards[sampled_idx], dtype=torch.float32).to(self.device),
            torch.tensor(self.terminals[sampled_idx], dtype=torch.int).to(self.device),
            torch.tensor(self.states[(sampled_idx + 1) % self.mem_count],
                         dtype=torch.float32).to(self.device),  # s'
            torch.tensor(self.action_masks[sampled_idx],
                         dtype=torch.int64).to(self.device))


class DQN(nn.Module):

    def __init__(self, stateDim: int, actionDim: int, hl1_size: int, hl2_size: int):
        super().__init__()

        # Hidden Layer 1
        self.fc1 = nn.Sequential(
            self.init_layer(nn.Linear(in_features=stateDim, out_features=hl1_size)),
            nn.ReLU(True))

        # Hidden Layer 2
        self.fc2 = nn.Sequential(
            self.init_layer(nn.Linear(in_features=hl1_size, out_features=hl2_size)),
            nn.ReLU(True))

        # Output Layer
        # No activation function because we are estimating Q-values
        self.fcOutput = nn.Linear(in_features=hl2_size, out_features=actionDim)

    def init_layer(self, layer, bias_const=0.0):
        torch.nn.init.kaiming_uniform_(layer.weight, nonlinearity='relu')
        torch.nn.init.constant_(layer.bias, bias_const)
        return layer

    def forward(self, x):

        """Forward Pass"""

        out = self.fc1(x)
        out = self.fc2(out)
        out = self.fcOutput(out)

        return out


class DoubleDQL():
    """
        Double Deep Q-Learning Class.

        This class contains the DDQL implementation.
    """

    def __init__(self, train_env: gym.Env, gamma: float = 0.99, epsilon: float = 1.0,
                 lr: float = 0.001, replay_mem_size: int = 100_000, loss_fn=None,
                 hl1_size: int = 64, hl2_size: int = 64, device: str = "cpu"):
        """
           Parameters
           ----------
           train_env : GraphEnv
               GraphEnv env for training the policy.
           loss_fn : Loss
               Loss function object for training the main-dqn.
           gamma : float
               Discount parameter ɣ.
           epsilon : float
               Exploration parameter Ɛ.
        """

        self.train_env = train_env
        self.device = device

        self.state_size = train_env.observation_space.shape[0]
        self.num_actions = train_env.action_space.n

        self.main_dqn = DQN(
            self.state_size, self.num_actions, hl1_size, hl2_size).to(self.device)
        self.target_dqn = DQN(
            self.state_size, self.num_actions, hl1_size, hl2_size).to(self.device)
        self.update_target_dqn()

        self.replay_memory = ReplayMemory(
            mem_size=replay_mem_size, state_size=self.state_size,
            num_actions=self.num_actions, device=device)

        self.optimizer = optim.Adam(self.main_dqn.parameters(), lr=lr)
        self.loss_fn = loss_fn

        self.gamma = gamma
        self.epsilon = epsilon

    def update_target_dqn(self) -> None:
        """
           Method for updating target-dqn parameters to that of the main-dqn.
        """

        self.target_dqn.load_state_dict(self.main_dqn.state_dict())

    def load_main_dqn(self, model_path: str = 'models/best_policy.pth') -> None:
        """
           Method to load main-dqn from saved model file.
        """

        self.main_dqn.load_state_dict(torch.load(model_path))
        self.main_dqn.eval()

    def epsilon_sample(self, q_values: np.array, epsilon: float, action_mask: np.array
                       ) -> int:
        """
           Method for sampling from the discrete action space based on Ɛ-greedy
           expolration strategy.

           Parameters
           ----------
           q_values : np.array
               An array of q-values.
           epsilon : float
               The probability of sampling an action from a uniform distribution.

           Returns
           -------
           int
               The index of the sampled action.
        """
        if np.random.rand() < epsilon:
            return np.random.choice(
                np.argwhere(np.array(action_mask) > 0).reshape(-1).tolist())
        else:
            a_mask_tensor = torch.tensor(action_mask)
            return torch.argmax(q_values + torch.log(a_mask_tensor)).cpu().numpy()

    def evaluate(self) -> Sequence[Union[int, float]]:
        """
           Method for evaluating policy during training.

           This method evaluates the current policy with each possible starting node.

           Returns
           -------
           float
               Mean normalized episodic reward among all evaluated episodes.
            int
               Mean episode length among all evaluated episodes.

        """

        # Initialize lists to store normalized eposide rewards and episode lengths
        episode_rewards = list()
        episode_lengths = list()

        # Evaluate the current policy starting with each possible/valid node per episode
        for node in self.train_env.possible_starting_nodes:
            observation, info = self.train_env.reset(starting_node=node)

            terminated = truncated = False
            episode_reward = 0
            episode_length = 0

            # The state -> action -> reward, next-state loop
            while not(terminated or truncated):
                state = observation
                q_values = self.main_dqn(
                    torch.tensor(state, dtype=torch.float32).to(self.device))
                action = self.epsilon_sample(
                    q_values, epsilon=0.0, action_mask=info['action_mask'])
                observation, reward, terminated, truncated, info = \
                    self.train_env.step(action)
                episode_reward += reward
                episode_length += 1

            # Store the normalized reward and the length of the current episode
            episode_rewards.append(episode_reward)
            episode_lengths.append(episode_length)

        return np.mean(episode_rewards), np.mean(episode_lengths)

    def final_evaluation(self) -> Sequence[float]:
        """
           Method for evaluating the policy at the end of training.

           Returns
           -------
           Sequence[float]
               A list of normalized episode rewards of the evaluation.
           Sequence[int]
               A list of episode lengths of the evaluation.
        """

        episode_rewards = list()
        episode_lengths = list()
        for e, node in enumerate(self.train_env.possible_starting_nodes):
            observation, info = self.train_env.reset(starting_node=node)

            terminated = truncated = False
            episode_reward = 0
            episode_length = 0

            # The state -> action -> reward, next-state loop
            while not(terminated or truncated):
                state = observation
                q_values = self.main_dqn(
                    torch.tensor(state, dtype=torch.float32).to(self.device))
                action = self.epsilon_sample(
                    q_values, epsilon=0.0, action_mask=info['action_mask'])
                observation, reward, terminated, truncated, info = \
                    self.train_env.step(action)
                episode_reward += reward
                episode_length += 1

            episode_rewards.append(episode_reward)
            episode_lengths.append(episode_length)
            print(f"Episode-{e+1}:  Reward: {episode_reward} - Length: {episode_length}")

        # Final policy evaluation statistics
        print("\nFinal Policy Evaluation Statistics:")
        print(f"Total number of episodes: {len(episode_rewards)}")
        print(f"Mean Normalized Episodic Reward: {np.mean(episode_rewards)}")
        print(f"Median Normalized Episodic Reward: {np.median(episode_rewards)}")

        return episode_rewards, episode_lengths

    def train(self, training_steps, init_training_period, main_update_period,
              target_update_period, batch_size=32, epsilon_decay: float = 0.99999,
              epsilon_min: float = 0.01, evaluation_freq: int = 50, verbose: bool = True,
              episode_len: int = 200, show_plot: bool = False, path: str = None):
        """
           Method for training the policy based with DDQL algorithm.

           Parameters
           ----------
           training_steps : int
               Maximum number of time steps to train the policy for.
           init_training_period : int
               Number of time steps of recording transitions, before initiating
               policy training.
           main_update_period : int
               Number of time steps between consecutive main-dqn batch updates.
           target_update_period : int
               Number of time steps between consecutive target-dqn updates to main-dqn.
           batch_size : int
               Batch size for main-dqn training.
           evaluation_freq : int
               Number of time steps between policy evaluations during training.
        """

        start_time = time.time()

        # Create a matlibplot canvas for plotting learning curves
        fig, axs = plt.subplots(3, figsize=(10, 11), sharey=False, sharex=True)

        # Initialize lists for storing learning curve data
        t_list = list()
        train_loss = list()
        train_reward = list()
        train_episode_len = list()
        train_episodes = list()
        eval_reward = list()
        eval_episodes = list()
        eval_episode_len = list()

        episode_loss = 0
        episode_reward = 0
        episode_duration = 0
        episode_count = 0
        best_eval_reward = -np.inf
        saved_model_txt = None

        self.main_dqn.train()
        self.target_dqn.train()

        # The state -> action -> reward, next-state loop for policy training
        observation, info = self.train_env.reset(seed=episode_count)
        for t in range(training_steps):
            state = observation

            with torch.no_grad():
                # Get Q(sₜ,a|θ) ∀ a ∈ A from the main-dqn
                q_values = self.main_dqn(
                    torch.tensor(state, dtype=torch.float32).to(self.device))

                # From Q(sₜ,a|θ) sample aₜ based on Ɛ-greedy
                action = self.epsilon_sample(q_values, epsilon=self.epsilon,
                                             action_mask=info['action_mask'])

            # Step through the enviroment with action aₜ, receiving reward rₜ, and
            # observing the new state sₜ₊₁
            observation, reward, terminated, truncated, info = \
                self.train_env.step(action)

            done = terminated or truncated

            # Save the transition {sₜ, aₜ, rₜ, sₜ₊₁} to the Replay Memory
            self.replay_memory.add(state, action, reward, done, info['action_mask'])

            # Normalized reward
            episode_reward += reward

            # Episode length for plotting
            episode_duration += 1

            if t > init_training_period:
                # Decay exploration parameter Ɛ over time to a minimum of EPSILON_MIN
                # Ɛₜ = (Ɛ-decay)ᵗ
                if self.epsilon > epsilon_min:
                    self.epsilon *= epsilon_decay

                # Main-DQN batch update
                if t % main_update_period == 0:
                    # From Replay Memory Buffer, uniformly sample a batch of transitions
                    states, actions, rewards, terminals, state_primes, action_masks = \
                        self.replay_memory.sample_batch(batch_size=batch_size)

                    with torch.no_grad():
                        # Best next action estimate of the main-dqn, for the sampled batch
                        # aⱼ = argmaxₐ Q(sₜ₊₁,a|θ), a ∈ A
                        best_action = torch.argmax(
                            self.main_dqn(state_primes) + torch.log(action_masks),
                            axis=-1, keepdims=True)

                        target_all_q = self.target_dqn(state_primes)

                        # Target q value for the sampled batch:
                        # yⱼ = rⱼ, if sⱼ' is a terminal-state
                        # yⱼ = rⱼ + ɣ Q(sⱼ',aⱼ|θ⁻), otherwise.
                        target_q = rewards + (self.gamma * torch.gather(
                            input=target_all_q, dim=1, index=best_action).reshape(-1) *
                            (1 - terminals))

                    # Predicted q value of the main-dqn, for the sampled batch
                    # Q(sⱼ,aⱼ|θ)
                    pred_q = self.main_dqn(states)
                    pred_q_a = torch.gather(
                        input=pred_q, dim=1, index=actions.reshape(-1, 1)).reshape(-1)

                    # Calculate loss:
                    # L(θ) = 𝔼[(Q(s,a|θ) - y)²]
                    loss = self.loss_fn(pred_q_a, target_q)

                    # Calculate the gradient of the loss w.r.t main-dqn parameters θ
                    self.optimizer.zero_grad()
                    loss.backward()

                    # Update main-dqn parameters θ:
                    self.optimizer.step()

                    # For plotting
                    episode_loss += loss.detach().numpy()

                if t % target_update_period == 0:
                    # Reset Target-DQN to Main-DQN
                    self.update_target_dqn()

            if done and (episode_count + 1) % np.abs(evaluation_freq) == 0:
                # Evaluate the current policy
                self.main_dqn.eval()
                mean_eval_rewards, mean_eval_episodes_length = \
                    self.evaluate()
                eval_reward.append(mean_eval_rewards)
                eval_episode_len.append(mean_eval_episodes_length)
                eval_episodes.append(episode_count + 1)

                if verbose:
                    print(f"{t+1} - Train: {np.round(train_reward[-1], 4)} - " +
                          f"Evaluation: {np.round(eval_reward[-1], 4)}")

                # Save a snapshot of the best policy (main-dqn) based on the
                # evaluation results
                if (mean_eval_rewards > best_eval_reward):
                    torch.save(self.main_dqn.state_dict(),
                               path + 'models/best_policy.pth')
                    best_eval_reward = mean_eval_rewards
                    saved_model_txt = \
                        f"Best Model Saved @ Episode {episode_count + 1} with " + \
                        f"eval reward: {np.round(best_eval_reward, 4)}"

                # Plot loss, rewards, and transition percentage
                plot_all(axs, train_episodes, train_loss, train_reward, eval_episodes,
                         eval_reward, train_episode_len=train_episode_len, path=path,
                         eval_episode_len=eval_episode_len, show=show_plot, save=True,
                         text=saved_model_txt)

                self.main_dqn.train()

            # Reset the environment and store normalized episode reward on completion or
            # termination of the current episode
            if done:
                episode_count += 1
                observation, info = self.train_env.reset(seed=episode_count)

                # For plotting
                train_episodes.append(episode_count)
                train_loss.append(episode_loss/episode_duration)
                train_reward.append(episode_reward)
                train_episode_len.append(episode_duration)
                episode_loss = 0
                episode_reward = 0
                episode_duration = 0

        end_time = time.time()
        print("\nTraining Time: %.2f(s)" % (end_time - start_time))
        input("Completed training.\nPress Enter to start the final evaluation")

        self.main_dqn.load_state_dict(torch.load('models/best_policy.pth'))

        self.main_dqn.eval()
        _ = self.final_evaluation()