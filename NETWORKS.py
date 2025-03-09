import torch as th
import torch.nn.functional as F
import numpy as np
from stable_baselines3 import DQN
from UTILS import compute_kl_divergence
from stable_baselines3.dqn.policies import CnnPolicy, DQNPolicy, MlpPolicy, MultiInputPolicy, QNetwork
import torch.nn as nn
import torch
import gym
from typing import Union, Optional, Dict, Tuple

############################################################
#Defining the NatureCNN architecture
#NOTE: This is the same CNN architecture as used in the Q_network
# in the original DQN function.
############################################################

class NatureCNN(nn.Module):
    def __init__(self, input_channels, features_dim=512):
        super().__init__()
        # Define the convolutional layers
        self.cnn = nn.Sequential(
            nn.Conv2d(input_channels, 32, kernel_size=8, stride=4),  # Use input_channels directly
            nn.ReLU(),  # Activation function
            nn.Conv2d(32, 64, kernel_size=4, stride=2),  # Second conv layer
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),  # Third conv layer
            nn.ReLU(),
            nn.Flatten(start_dim=1, end_dim=-1)  # Flatten the output for the linear layer
        )
        # Define the linear layer
        self.linear = nn.Sequential(
            nn.Linear(9216, features_dim),  # 3136 is from CNN output
            nn.ReLU()
        )

        # Ensure model is on CUDA if available
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.to(self.device)

    def forward(self, x):
        # Ensure input is float and on correct device
        if x.dtype != torch.float32:
            x = x.to(dtype=torch.float32)
        x = x.to(self.device)
        
        # Debug print
        
        x = self.cnn(x)
        x = self.linear(x)
        return x


############################################################
# Defining the SharedPolicy architecture using the same architecture
# as the Qnetwork in the DQN function. 
############################################################
class SharedPolicy(nn.Module):
    def __init__(self, input_channels, action_space):
        super().__init__()
        # Use NatureCNN as the feature extractor
        self.features_extractor = NatureCNN(input_channels)
        # Define the Q-network with output size action_space
        self.q_net = nn.Sequential(
            nn.Linear(512, action_space)  # 512 is features_dim from NatureCNN
        )

        # Ensure model is on CUDA if available
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.to(self.device)

    def forward(self, x):
        # Ensure input is on correct device
        x = x.to(self.device)
        x = self.features_extractor(x)
        x = self.q_net(x)
        return x

    def set_training_mode(self, mode: bool) -> None:
        """
        Put the policy in either training or evaluation mode.
        
        :param mode: if true, set to training mode
        """
        self.train(mode)

############################################################
# We define a custom DQN class that inherits from the DQN class
# This allows for easier training and implementation of the 
# KL divergence loss term.
############################################################

class CustomDQN(DQN):
    def __init__(self, *args, shared_policy=None, kl_weight=0.1, env_moves=None, mapping=None, max_actions=None, env_number=None, **kwargs):
        """
        Initialize CustomDQN with an additional KL divergence weight term.

        Args:
            shared_policy (nn.Module): The shared policy to compute KL divergence.
            kl_weight (float): Weight for the KL divergence term.
            env_moves (list): List of moves for the specific environment.
            mapping (dict): Mapping of moves to indices.
            max_actions (int): Size of the common action space.
            env_number (int): The index or number of the environment.
        """
        super().__init__(*args, **kwargs)
        self.shared_policy = shared_policy
        self.kl_weight = kl_weight
        self.env_moves = env_moves
        self.mapping = mapping
        self.max_actions = max_actions
        self.env_number = env_number
        
        #Printing networks for debugging purposes
        print(f"CustomDQN initialized for env {env_number} with {len(env_moves)} actions: {env_moves}")
        print(f"Q-net: {self.q_net}\n")
        print(f"Q-net target: {self.q_net_target}\n")

    ############################################################
    #Mostly the same train function as the original DQN, However,
    #we added a KL divergence loss term to the loss function.
    ############################################################
    
    
    def train(self, gradient_steps: int, batch_size: int = 100) -> None:
        # Switch to train mode (this affects batch norm / dropout)
        self.policy.set_training_mode(True)
        # Update learning rate according to schedule
        self._update_learning_rate(self.policy.optimizer)

        losses = []

        for _ in range(gradient_steps):
            # Sample replay buffer
            replay_data = self.replay_buffer.sample(batch_size, env=self._vec_normalize_env)

            with th.no_grad():
                # Process next observations through target Q-network
                next_q_values = self.q_net_target(replay_data.next_observations)
                next_q_values, _ = next_q_values.max(dim=1)
                next_q_values = next_q_values.reshape(-1, 1)
                # 1-step TD target
                target_q_values = replay_data.rewards + (1 - replay_data.dones) * self.gamma * next_q_values

            # Ensure the input is of type float32 and move to the correct device
            replay_data_observations = replay_data.observations.to(th.float32).to(self.device)

            # Process current observations through the Q-network
            current_q_values = self.q_net(replay_data_observations)

            # Retrieve the q-values for the actions from the replay buffer
            current_q_values = th.gather(current_q_values, dim=1, index=replay_data.actions.long())

            # Compute Huber loss (less sensitive to outliers)
            loss = F.smooth_l1_loss(current_q_values, target_q_values)

            # Get logits (action probabilities) from both the DQN model and shared policy
            dqn_logits = self.q_net(replay_data_observations)  # Logits from DQN's policy
            shared_logits = self.shared_policy(replay_data_observations)  # Logits from the external shared policy

            ############################################################
            #Compute the KL divergence loss (NEW)
            ############################################################
            
            kl_divergence_loss = compute_kl_divergence(dqn_logits, shared_logits, self.env_moves, self.mapping, self.env_number)

            # Combine the losses (standard Q-learning loss + KL divergence loss)
            total_loss = loss + self.kl_weight * kl_divergence_loss
            losses.append(total_loss.item())

            # Optimize the policy
            self.policy.optimizer.zero_grad()
            total_loss.backward()
            # Clip gradient norm
            th.nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
            self.policy.optimizer.step()

        # Increase update counter
        self._n_updates += gradient_steps

        self.logger.record("train/n_updates", self._n_updates, exclude="tensorboard")
        self.logger.record("train/loss", np.mean(losses))