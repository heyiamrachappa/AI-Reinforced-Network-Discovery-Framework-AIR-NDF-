import os
import sys
import numpy as np
import torch
import json
import matplotlib.pyplot as plt
from typing import Dict, Any, List

# Add workspace to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from src.environment.net_env import NetworkDiscoveryEnv
from src.utils.generator import load_network_from_json
from src.agents.dqn import DQNAgent
from src.agents.ppo import PPOAgent
from src.agents.a2c import A2CAgent

def train_agent(agent_type: str, 
                network_scale: str = "medium", 
                episodes: int = 250, 
                checkpoint_dir: str = "results/checkpoints", 
                log_dir: str = "results/logs",
                progress_callback=None) -> List[float]:
    """
    Orchestrates the training loop for DQN, PPO, or A2C agents on a specific network.
    Saves checkpoints and loss/reward histories.
    """
    os.makedirs(checkpoint_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)
    
    # Load env
    dataset_path = f"src/datasets/network_{network_scale}.json"
    if os.path.exists(dataset_path):
        G = load_network_from_json(dataset_path)
    else:
        from src.utils.generator import NetworkGenerator
        gen = NetworkGenerator(seed=42)
        G = gen.generate(network_scale)
        
    env = NetworkDiscoveryEnv(graph=G)
    
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n
    
    # Instantiate agent
    if agent_type.upper() == "DQN":
        agent = DQNAgent(state_dim=state_dim, action_dim=action_dim, lr=1e-3, target_update_freq=100)
    elif agent_type.upper() == "PPO":
        agent = PPOAgent(state_dim=state_dim, action_dim=action_dim, lr=3e-4, ppo_epochs=5, batch_size=32)
    elif agent_type.upper() == "A2C":
        agent = A2CAgent(state_dim=state_dim, action_dim=action_dim, lr=5e-4)
    else:
        raise ValueError(f"Unknown agent type: {agent_type}")
        
    print(f"Starting training for {agent_type} on {network_scale} network ({episodes} episodes)...")
    
    reward_history = []
    loss_history = []
    coverage_history = []
    detection_history = []
    
    for ep in range(episodes):
        obs, info = env.reset()
        done = False
        truncated = False
        ep_reward = 0.0
        losses = []
        
        while not (done or truncated):
            if agent_type.upper() == "DQN":
                action = agent.select_action(obs, evaluation=False)
                next_obs, reward, done, truncated, info = env.step(action)
                agent.memory.push(obs, action, reward, next_obs, done or truncated)
                update_info = agent.update()
                if "loss" in update_info:
                    losses.append(update_info["loss"])
                obs = next_obs
                
            elif agent_type.upper() == "PPO":
                action, log_prob, val = agent.get_action_log_prob_and_value(obs)
                next_obs, reward, done, truncated, info = env.step(action)
                agent.store_transition(obs, action, log_prob, reward, val, done or truncated)
                obs = next_obs
                
            elif agent_type.upper() == "A2C":
                action = agent.select_action(obs, evaluation=False)
                next_obs, reward, done, truncated, info = env.step(action)
                agent.store_transition(obs, action, reward, next_obs, done or truncated)
                obs = next_obs

            ep_reward += reward

        # Ep end updates
        if agent_type.upper() == "DQN":
            agent.decay_epsilon()
        elif agent_type.upper() == "PPO":
            # Estimate next state value
            with torch.no_grad():
                if done or truncated:
                    next_val = 0.0
                else:
                    next_val = agent.ac.get_value(torch.FloatTensor(obs).unsqueeze(0).to(agent.device)).item()
            update_info = agent.update(next_val, done or truncated)
            if "actor_loss" in update_info:
                losses.append(update_info["actor_loss"] + update_info["critic_loss"])
        elif agent_type.upper() == "A2C":
            update_info = agent.update()
            if "loss" in update_info:
                losses.append(update_info["loss"])
                
        # Log episode metrics
        reward_history.append(ep_reward)
        coverage_history.append(len(env.explored_nodes) / len(env.G))
        detection_history.append(1.0 if info["detected"] else 0.0)
        
        avg_loss = np.mean(losses) if losses else 0.0
        loss_history.append(float(avg_loss))
        
        if progress_callback:
            progress_callback(ep + 1, episodes, ep_reward, len(env.explored_nodes) / len(env.G), avg_loss)
            
        if (ep + 1) % 50 == 0:
            avg_rew_last_50 = np.mean(reward_history[-50:])
            avg_cov_last_50 = np.mean(coverage_history[-50:])
            print(f"Episode {ep+1}/{episodes} | Avg Reward (last 50): {avg_rew_last_50:.2f} | Avg Coverage (last 50): {avg_cov_last_50*100:.1f}% | Loss: {avg_loss:.4f}")
            
    # Save checkpoint
    chk_path = os.path.join(checkpoint_dir, f"{agent_type.lower()}_agent.pt")
    agent.save(chk_path)
    print(f"Checkpoint saved to {chk_path}")
    
    # Save log history
    log_data = {
        "agent": agent_type,
        "network_scale": network_scale,
        "rewards": reward_history,
        "losses": loss_history,
        "coverages": coverage_history,
        "detections": detection_history
    }
    log_path = os.path.join(log_dir, f"{agent_type.lower()}_history.json")
    with open(log_path, 'w') as f:
        json.dump(log_data, f)
    print(f"Logs saved to {log_path}")
    
    # Generate simple training curve plot
    plot_training_curves(agent_type, reward_history, coverage_history, log_dir)
    
    return reward_history

def plot_training_curves(agent_type: str, rewards: List[float], coverages: List[float], output_dir: str):
    """Plots and saves the training metrics over episodes."""
    plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
    
    fig, ax1 = plt.subplots(figsize=(8, 5))
    
    color = '#1f77b4'
    ax1.set_xlabel('Episodes', fontweight='bold')
    ax1.set_ylabel('Cumulative Episode Reward', color=color, fontweight='bold')
    # Use rolling average for smoother visualization
    rolling_rewards = np.convolve(rewards, np.ones(10)/10, mode='valid')
    ax1.plot(rewards, color=color, alpha=0.3, label='Raw Reward')
    ax1.plot(range(9, len(rewards)), rolling_rewards, color=color, linewidth=2, label='10-ep Rolling Reward')
    ax1.tick_params(axis='y', labelcolor=color)
    
    ax2 = ax1.twinx()  
    color = '#2ca02c'
    ax2.set_ylabel('Discovery Coverage Ratio', color=color, fontweight='bold')
    rolling_coverages = np.convolve(coverages, np.ones(10)/10, mode='valid')
    ax2.plot(coverages, color=color, alpha=0.3, label='Raw Coverage')
    ax2.plot(range(9, len(coverages)), rolling_coverages, color=color, linewidth=2, linestyle='--', label='10-ep Rolling Coverage')
    ax2.tick_params(axis='y', labelcolor=color)
    
    plt.title(f"{agent_type} Training Convergence Curves", fontsize=14, fontweight='bold', pad=15)
    fig.tight_layout()  
    plot_path = os.path.join(output_dir, f"training_{agent_type.lower()}.png")
    plt.savefig(plot_path, dpi=300)
    plt.close()
    print(f"Saved training curve plot to {plot_path}")

if __name__ == "__main__":
    # If run directly, run a quick training of DQN on small network
    train_agent("DQN", network_scale="small", episodes=50)
