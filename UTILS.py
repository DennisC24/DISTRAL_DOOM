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
            print(f"Episode Number {len(self.episode_rewards)} done! Reward: {self.current_episode_reward} | Length: {self.current_episode_length}")
            self.current_episode_reward = 0  # Reset for the next episode
            self.current_episode_length = 0  # Reset for the next episode
            
        return True
    
def compute_kl_divergence(task_policy_q, shared_policy_q, env_moves, mapping, env_number, temperature=1.0, epsilon=1e-10):
    """
    Computes the KL divergence between task-specific and shared policy logits for common actions.

    Args:
        task_policy_q (torch.Tensor): Logits from the task-specific policy (batch_size, num_common_actions).
        shared_policy_q (torch.Tensor): Logits from the shared policy (batch_size, num_total_actions).
        env_moves (list): List of moves for the specific environment.
        mapping (dict): Mapping of moves to indices.
        env_number (int): The index or number of the environment.
        temperature (float): Temperature for scaling logits.
        epsilon (float): Small value to avoid log(0).

    Returns:
        torch.Tensor: The mean KL divergence over the batch for common actions.
    """
    # Identify common action indices in the shared policy
    common_indices = [mapping[move] for move in env_moves if move in mapping]


    # Filter logits for common actions
    task_policy_q_common = task_policy_q[:, :len(common_indices)]
    shared_policy_q_common = shared_policy_q[:, common_indices]

    # Apply temperature scaling and convert logits to probabilities
    task_probs = F.softmax(task_policy_q_common / temperature, dim=-1)
    shared_probs = F.softmax(shared_policy_q_common / temperature, dim=-1)

    
    # Add epsilon to avoid log(0)
    task_probs = task_probs + epsilon
    shared_probs = shared_probs + epsilon
    
    # Compute KL divergence
    kl_div = F.kl_div(task_probs.log(), shared_probs, reduction='batchmean')


    return kl_div
