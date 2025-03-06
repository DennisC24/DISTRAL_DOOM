import optuna
from optuna.pruners import MedianPruner
import torch
import numpy as np
from NETWORKS import SharedPolicy, CustomDQN
from ENVIRONMENTS import create_vizdoom_envs
from UTILS import RewardTrackingCallback, compute_kl_divergence

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

def train(config_paths, mapping, env_moves, kl_weight):
    """
    Train DISTRAL agents.
    
    Args:
        config_paths (list): List of paths to VizDoom configuration files
        mapping (dict): Mapping of action names to indices
        env_moves (list): List of available moves for each environment
        kl_weight (float): Weight for KL divergence term
    """
    import gc
    
    # Set seeds
    set_seeds(42)
    
    # Initialize environments
    envs = create_vizdoom_envs(config_paths)
    for env in envs:
        env.reset(seed=42)
    
    # Initialize shared policy
    max_actions = max([len(moves) for moves in env_moves])
    shared_policy = SharedPolicy(input_channels=1, max_actions=max_actions).to("cuda")
    
    initial_memory = torch.cuda.memory_allocated()
    
    # Initialize models and callbacks
    task_policies = []
    reward_callbacks = []
    losses = []
    all_episode_rewards = [[] for _ in envs]
    
    # Setup shared optimizer
    learning_rate = 1e-4
    shared_optimizer = torch.optim.Adam(shared_policy.parameters(), lr=learning_rate)
    
    # Create policies for each environment
    for i, env in enumerate(envs):
        action_dim = len(env_moves[i])
        task_policy = CustomDQN(
            "CnnPolicy",
            env,
            verbose=0,
            learning_rate=learning_rate,
            shared_policy=shared_policy,
            kl_weight=kl_weight,
            env_moves=env_moves[i],
            mapping=mapping,
            max_actions=action_dim,
            env_number=i
        )
        task_policy.q_net = task_policy.q_net.to("cuda")
        task_policy.q_net_target = task_policy.q_net_target.to("cuda")
        task_policies.append(task_policy)
        reward_callbacks.append(RewardTrackingCallback())

    # Main Training Loop
    batch_size = 100
    num_updates = 1000
    num_play = 1
    
    for step in range(num_updates):
        torch.cuda.empty_cache()
        gc.collect()
        
        for i, (task_policy, reward_callback) in enumerate(zip(task_policies, reward_callbacks)):
            task_policy.learn(total_timesteps=num_play, reset_num_timesteps=False, callback=reward_callback)
            all_episode_rewards[i] = reward_callback.episode_rewards
            print("Env completed")
        replay_data = [policy.replay_buffer.sample(batch_size, env=policy._vec_normalize_env) for policy in task_policies]
        batch_obs = [torch.tensor(data.observations, dtype=torch.float32).to("cuda") for data in replay_data]
        task_policy_q = []
        shared_policy_q = []

        for i in range(len(task_policies)):
            with torch.cuda.amp.autocast():
                current_policy = task_policies[i]
                current_q_net = current_policy.q_net
                current_obs = batch_obs[i]
                task_logits = current_q_net(current_obs)
                shared_logits = shared_policy(current_obs)
                task_policy_q.append(task_logits)
                shared_policy_q.append(shared_logits)

        shared_loss = sum(
            compute_kl_divergence(tp_q, sp_q, env_moves[i], mapping, i)
            for i, (tp_q, sp_q) in enumerate(zip(task_policy_q, shared_policy_q))
        ) / len(task_policy_q)

        shared_optimizer.zero_grad()
        shared_loss.backward()
        shared_optimizer.step()
        losses.append(shared_loss.item())

        del replay_data, batch_obs, task_policy_q, shared_policy_q
        torch.cuda.empty_cache()

        if step % 10 == 0:
            current_memory = torch.cuda.memory_allocated()
            print(f"Step {step}: Shared Policy Loss = {shared_loss.item()}")
            print(f"Memory change: {(current_memory - initial_memory)/1e9:.2f}GB")
            print(f"Current memory: {current_memory/1e9:.2f}GB")

    return {
        'episode_rewards': all_episode_rewards,
        'losses': losses,
        'task_policies': task_policies,
        'shared_policy': shared_policy
    }

def objective(trial):
    """Optuna objective function to minimize."""
    # Define the configuration
    config_paths = [
        "./VIZDOOM/scenarios/defend_the_line.cfg",
        "./VIZDOOM/scenarios/defend_the_center.cfg",
        "./VIZDOOM/scenarios/deadly_corridor.cfg",
        "./VIZDOOM/scenarios/my_way_home.cfg",
        "./VIZDOOM/scenarios/take_cover.cfg",
        "./VIZDOOM/scenarios/health_gathering.cfg"
    ]
    
    mapping = {
        "SHOOT": 0,
        "MOVE_LEFT": 1,
        "MOVE_RIGHT": 2,
        "MOVE_FORWARD": 3,
        "MOVE_BACKWARD": 4,
        "TURN_LEFT": 5,
        "TURN_RIGHT": 6,
    }
    
    env_moves = [
        ["SHOOT", "TURN_LEFT", "TURN_RIGHT"],
        ["SHOOT", "TURN_LEFT", "TURN_RIGHT"],
        ["SHOOT", "MOVE_LEFT", "MOVE_RIGHT", "MOVE_FORWARD", "MOVE_BACKWARD", "TURN_LEFT", "TURN_RIGHT"],
        ["TURN_LEFT", "TURN_RIGHT", "MOVE_FORWARD"],
        ["MOVE_RIGHT","MOVE_LEFT"],
        ["TURN_LEFT", "TURN_RIGHT", "MOVE_FORWARD"]
    ]
    
    # Sample KL weight between 0 and 10
    kl_weight = trial.suggest_float('kl_weight', 0.00001, 10.0, log=True)
    
    # Train using the training function
    results = train(config_paths, mapping, env_moves, kl_weight)
    
    # Calculate mean reward across all environments
    mean_rewards = []
    for env_rewards in results['episode_rewards']:
        mean_reward = np.mean(env_rewards[-100:]) if env_rewards else 0
        mean_rewards.append(mean_reward)
    
    final_mean_reward = np.mean(mean_rewards)
    
    # Report intermediate values every 100 steps
    for step in range(0, len(results['losses']), 100):
        intermediate_value = np.mean([
            np.mean(env_rewards[max(0, step-100):step]) if len(env_rewards) > step else 0
            for env_rewards in results['episode_rewards']
        ])
        trial.report(intermediate_value, step)
        
        if trial.should_prune():
            raise optuna.TrialPruned()
    
    return final_mean_reward

if __name__ == "__main__":
    # Create study
    study_name = "distral_kl_tuning"
    storage_name = f"sqlite:///{study_name}.db"
    
    study = optuna.create_study(
        study_name=study_name,
        storage=storage_name,
        direction="maximize",
        pruner=MedianPruner(
            n_startup_trials=5,
            n_warmup_steps=100,
            interval_steps=100
        ),
        load_if_exists=True
    )
    
    # Optimize
    study.optimize(objective, n_trials=50)
    
    # Print results
    print("Best trial:")
    print("  Value: ", study.best_trial.value)
    print("  Params: ")
    for key, value in study.best_trial.params.items():
        print(f"    {key}: {value}")