import torch
import argparse
import pandas as pd
from NETWORKS import CustomDQN, SharedPolicy
from ENVIRONMENTS import create_vizdoom_envs
from UTILS import RewardTrackingCallback, compute_kl_divergence
from gymnasium import spaces

def set_seeds(seed=42):
    """Set seeds for reproducibility"""
    import numpy as np
    import torch
    import random
    
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def parse_args():
    parser = argparse.ArgumentParser(description="Run DISTRAL algorithm with custom KL weight.")
    parser.add_argument('--kl_weight', type=float, default=0.125, help='KL divergence weight for the CustomDQN policy')
    parser.add_argument('--learning_rate', type=float, default=1e-4, help='Learning rate for the DQN algorithm')
    parser.add_argument('--num_timesteps', type=int, default=1000, help='Number of timesteps to run in each learning iteration')
    parser.add_argument('--run_name', type=str, default='run1', help='Name for the current run')
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    set_seeds(42)
    # Setting device to cuda initially
    device = "cuda"
    torch.device(device)

    ########################################################
    # Creating the environments
    ########################################################
    config_paths = [
        "./VIZDOOM/scenarios/defend_the_line.cfg",
        "./VIZDOOM/scenarios/defend_the_center.cfg",
        "./VIZDOOM/scenarios/deadly_corridor.cfg",
        "./VIZDOOM/scenarios/my_way_home.cfg",
        "./VIZDOOM/scenarios/take_cover.cfg",
        "./VIZDOOM/scenarios/health_gathering.cfg"
    ]
    envs = create_vizdoom_envs(config_paths)
    for env in envs:
        env.reset(seed=42) 
    ########################################################
    # Define the Mapping and Environment Moves
    ########################################################
    Mapping = {
        "SHOOT": 0,
        "MOVE_LEFT": 1,
        "MOVE_RIGHT": 2,
        "MOVE_FORWARD": 3,
        "MOVE_BACKWARD": 4,
        "TURN_LEFT": 5,
        "TURN_RIGHT": 6,
    }
    ENV_MOVES = [["SHOOT", "TURN_LEFT", "TURN_RIGHT"],
                 ["SHOOT", "TURN_LEFT", "TURN_RIGHT"],
                 ["SHOOT", "MOVE_LEFT", "MOVE_RIGHT", "MOVE_FORWARD", "MOVE_BACKWARD", "TURN_LEFT", "TURN_RIGHT"],
                 ["TURN_LEFT", "TURN_RIGHT", "MOVE_FORWARD"],
                 ["MOVE_RIGHT","MOVE_LEFT"],
                 ["TURN_LEFT", "TURN_RIGHT", "MOVE_FORWARD"]]
    max_actions = max([len(moves) for moves in ENV_MOVES])

    ########################################################
    # Defining the RewardTrackingCallback
    ########################################################
    reward_callbacks = [RewardTrackingCallback() for _ in envs]
    losses = []

    ########################################################
    # Creating the shared policy
    ########################################################
    input_channels = 1  # We have greyscale input
    shared_policy = SharedPolicy(input_channels, max_actions).to(device)
    shared_optimizer = torch.optim.Adam(shared_policy.parameters(), lr=args.learning_rate)

    ########################################################
    # Creating the task-specific policies
    ########################################################
    
    task_policies = []
    for i, env in enumerate(envs):
        # Create action space based on number of actions for this environment
        action_dim = len(ENV_MOVES[i])
        env.action_dim = action_dim
        # Modify environment's action space to match our desired size
        env.action_space = spaces.Discrete(action_dim)
        
        task_policy = CustomDQN(
            "CnnPolicy",
            env,  # Now env has the correct action space
            verbose=0,
            learning_rate=args.learning_rate,
            shared_policy=shared_policy,
            kl_weight=args.kl_weight,
            env_moves=ENV_MOVES[i],
            mapping=Mapping,
            max_actions=action_dim,
            env_number= i 
        )
        task_policies.append(task_policy)

    
    
    
    
    ########################################################
    # Main Training Loop
    ########################################################
    batch_size = 100
    run_name = args.run_name
    num_play = args.num_timesteps
    num_updates = 100
    all_episode_rewards = [[] for _ in envs]
    initial_memory = torch.cuda.memory_allocated()
    for step in range(num_updates):
        for i, (task_policy, reward_callback) in enumerate(zip(task_policies, reward_callbacks)):
            task_policy.learn(total_timesteps=num_play, reset_num_timesteps=False, callback=reward_callback)
            all_episode_rewards[i] = reward_callback.episode_rewards

        # Shared loss calculation
        replay_data = [policy.replay_buffer.sample(batch_size, env=policy._vec_normalize_env) for policy in task_policies]
        batch_obs = [torch.tensor(data.observations, dtype=torch.float32).to(device) for data in replay_data]
        task_policy_q = []
        shared_policy_q = []

        for i in range(len(task_policies)):
            current_policy = task_policies[i]
            current_q_net = current_policy.q_net
            current_obs = batch_obs[i]
            task_logits = current_q_net(current_obs)
            shared_logits = shared_policy(current_obs)
            task_policy_q.append(task_logits)
            shared_policy_q.append(shared_logits)

        shared_loss = sum(
            compute_kl_divergence(tp_q, sp_q, ENV_MOVES[i], Mapping, i)
            for i, (tp_q, sp_q) in enumerate(zip(task_policy_q, shared_policy_q))
        ) / len(task_policy_q)

        shared_optimizer.zero_grad()
        shared_loss.backward()
        shared_optimizer.step()
        losses.append(shared_loss.item())

        # Save losses to CSV
        loss_df = pd.DataFrame({"Step": range(len(losses)), "SharedLoss": losses})
        loss_df.to_csv(f"./Losses/shared_losses-{run_name}.csv", index=False)

        # Save rewards to separate CSV files for each environment
        for i, rewards in enumerate(all_episode_rewards):
            reward_df = pd.DataFrame({"Episode": range(len(rewards)), "Reward": rewards})
            reward_df.to_csv(f"./Rewards/rewards_env{i+1}-{run_name}.csv", index=False)

        if step % 100 == 0:
            current_memory = torch.cuda.memory_allocated()
            print(f"Memory change: {(current_memory - initial_memory)/1e9:.2f}GB")

    print("Training complete.")
