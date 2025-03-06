import torch
from stable_baselines3.common.callbacks import BaseCallback
import torch.nn.functional as F


def map_tensor_to_actions(input_tensor, moves, mapping, max_actions):
    """
    Maps the input tensor to a new tensor of size max_actions based on the moves and mapping.

    Args:
        input_tensor (torch.Tensor): The input tensor with values corresponding to the moves.
        moves (list): A list of move names corresponding to the input tensor.
        mapping (dict): A dictionary mapping move names to their respective indices.
        max_actions (int): The size of the output tensor.

    Returns:
        torch.Tensor: A new tensor of size max_actions with values mapped according to the mapping.
    """
    # Initialize a new tensor of zeros with size max_actions
    output_tensor = torch.zeros(max_actions, dtype=input_tensor.dtype)

    # Map the input tensor values to the output tensor based on the mapping
    for i, move in enumerate(moves):
        if move in mapping:
            output_tensor[mapping[move]] = input_tensor[i]

    return output_tensor


class RewardTrackingCallback(BaseCallback):
    def __init__(self):
        super(RewardTrackingCallback, self).__init__()
        self.episode_rewards = []
        self.current_episode_reward = 0
        self.current_episode_length = 0

    def _on_step(self) -> bool:
        # Check if the environment is done
        done = self.locals['dones'][0]
        reward = self.locals['rewards'][0]
        
        # Accumulate reward and episode length
        self.current_episode_reward += reward
        self.current_episode_length += 1
        
        # If the episode is done, print and log it
        if done:
            self.episode_rewards.append(self.current_episode_reward)
            self.current_episode_reward = 0  # Reset for the next episode
            self.current_episode_length = 0  # Reset for the next episode
            
        return True
    
def compute_kl_divergence(task_policy_q, shared_policy_q, env_moves, mapping, env_number):
    # Debug prints
    print(f"Task policy Q shape: {task_policy_q.shape}")
    print(f"Shared policy Q shape: {shared_policy_q.shape}")
    
    # Get the actual indices for this environment's moves
    env_indices = [mapping[move] for move in env_moves]
    print(f"Environment {env_number} indices: {env_indices}\n")
    
    # Use the environment-specific indices
    task_policy_q_common = task_policy_q  # Already has correct shape
    shared_policy_q_common = shared_policy_q[:, env_indices]  # Select only relevant actions
    
    # Compute KL divergence
    task_probs = F.softmax(task_policy_q_common, dim=1)
    shared_probs = F.softmax(shared_policy_q_common, dim=1)
    
    kl_div = F.kl_div(
        F.log_softmax(task_policy_q_common, dim=1),
        shared_probs,
        reduction='batchmean'
    )
    
    return kl_div
