import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import torch
from typing import Dict, Any, List, Type
import json

from src.environment.net_env import NetworkDiscoveryEnv
from src.utils.generator import load_network_from_json
from src.evaluation.baselines import RandomSearchAgent, BFSAgent, DFSAgent, GreedySearchAgent
from src.agents.dqn import DQNAgent
from src.agents.ppo import PPOAgent
from src.agents.a2c import A2CAgent

def evaluate_agent(agent: Any, 
                   env: NetworkDiscoveryEnv, 
                   num_episodes: int = 20, 
                   is_rl: bool = True) -> Dict[str, float]:
    """Runs evaluation episodes for an agent and returns average metrics."""
    covs = []
    costs = []
    steps = []
    detected_count = 0
    rewards = []
    info_gain = []

    for _ in range(num_episodes):
        obs, info = env.reset()
        done = False
        truncated = False
        episode_reward = 0.0
        
        while not (done or truncated):
            if is_rl:
                action = agent.select_action(obs, evaluation=True)
            else:
                action = agent.select_action(env)
                
            obs, reward, done, truncated, info = env.step(action)
            episode_reward += reward

        # Record metrics
        total_nodes = len(env.G)
        covs.append(len(env.explored_nodes) / total_nodes)
        
        # Risk estimation: scan penalty + base steps
        steps.append(info["step"])
        if info["detected"]:
            detected_count += 1
            
        rewards.append(episode_reward)
        
        # Info gain: cumulative vulnerability scores of explored nodes
        vuln_sum = sum(env.G.nodes[n].get("vulnerability_score", 0.0) for n in env.explored_nodes)
        info_gain.append(vuln_sum)

    return {
        "avg_coverage": float(np.mean(covs)),
        "avg_steps": float(np.mean(steps)),
        "avg_reward": float(np.mean(rewards)),
        "detection_rate": float(detected_count / num_episodes),
        "avg_info_gain": float(np.mean(info_gain))
    }

def run_benchmarks(network_scale: str = "medium", 
                   num_episodes: int = 20, 
                   checkpoint_dir: str = "results/checkpoints", 
                   output_dir: str = "results") -> pd.DataFrame:
    """Runs benchmarks for all baseline and RL agents, saving outputs."""
    os.makedirs(output_dir, exist_ok=True)
    
    # Load environment
    dataset_path = f"src/datasets/network_{network_scale}.json"
    if os.path.exists(dataset_path):
        G = load_network_from_json(dataset_path)
    else:
        # Fallback to direct generation if json doesn't exist
        from src.utils.generator import NetworkGenerator
        gen = NetworkGenerator(seed=42)
        G = gen.generate(network_scale)
        
    env = NetworkDiscoveryEnv(graph=G)
    
    # Initialize agents
    agents = {
        "Random": (RandomSearchAgent(), False),
        "BFS": (BFSAgent(), False),
        "DFS": (DFSAgent(), False),
        "Greedy": (GreedySearchAgent(), False)
    }
    
    # Try to load RL agents from checkpoints
    rl_agent_types = [
        ("DQN", DQNAgent, "dqn_agent.pt"),
        ("PPO", PPOAgent, "ppo_agent.pt"),
        ("A2C", A2CAgent, "a2c_agent.pt")
    ]
    
    for name, agent_cls, checkpoint_file in rl_agent_types:
        chk_path = os.path.join(checkpoint_dir, checkpoint_file)
        if os.path.exists(chk_path):
            try:
                # Instantiate agent with default dims
                agent = agent_cls(state_dim=24, action_dim=5, device="cpu")
                agent.load(chk_path)
                agents[name] = (agent, True)
                print(f"Loaded trained RL agent: {name} from {chk_path}")
            except Exception as e:
                print(f"Failed to load RL agent {name}: {e}. Running default agent instead.")
                # We still evaluate default agent for architecture verification
                agent = agent_cls(state_dim=24, action_dim=5, device="cpu")
                agents[name] = (agent, True)
        else:
            print(f"No checkpoint found for {name} at {chk_path}. Evaluating untrained agent.")
            agent = agent_cls(state_dim=24, action_dim=5, device="cpu")
            agents[name] = (agent, True)

    results_data = []
    
    for name, (agent, is_rl) in agents.items():
        print(f"Benchmarking agent: {name} on {network_scale} network...")
        metrics = evaluate_agent(agent, env, num_episodes=num_episodes, is_rl=is_rl)
        metrics["agent"] = name
        results_data.append(metrics)
        
    df = pd.DataFrame(results_data)
    df = df[["agent", "avg_coverage", "avg_steps", "avg_reward", "detection_rate", "avg_info_gain"]]
    
    # Save table
    csv_path = os.path.join(output_dir, f"benchmark_{network_scale}.csv")
    df.to_csv(csv_path, index=False)
    print(f"Saved benchmark results table to {csv_path}")
    
    # Generate Plots
    generate_comparison_plots(df, network_scale, output_dir)
    
    return df

def generate_comparison_plots(df: pd.DataFrame, scale: str, output_dir: str):
    """Generates academic-grade comparative plots."""
    plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
    
    # Plot colors (sleek dark/modern theme palette)
    colors = ['#5A6F72', '#2F4F4F', '#708090', '#4682B4', '#1F77B4', '#FF7F0E', '#2CA02C']
    
    # 1. Coverage vs Detection Rate (Multi-Objective Tradeoff)
    fig, ax = plt.subplots(figsize=(8, 6))
    for idx, row in df.iterrows():
        ax.scatter(row['detection_rate'], row['avg_coverage'], s=250, label=row['agent'], alpha=0.9, edgecolors='black', linewidth=1.5)
        ax.annotate(row['agent'], (row['detection_rate']+0.01, row['avg_coverage']+0.01), fontsize=12, fontweight='bold')
        
    ax.set_title(f"Risk-Discovery Trade-off Profile ({scale.capitalize()} Network)", fontsize=14, fontweight='bold', pad=15)
    ax.set_xlabel("Detection Probability (Threat Vector Risk)", fontsize=12)
    ax.set_ylabel("Discovery Coverage (Topology Mapped %)", fontsize=12)
    ax.set_xlim(-0.05, 1.05)
    ax.set_ylim(-0.05, 1.05)
    ax.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"tradeoff_{scale}.png"), dpi=300)
    plt.close()
    
    # 2. Grouped metrics bar chart (Coverage, Steps, Rewards)
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    # Subplot 1: Discovery Coverage
    axes[0].bar(df['agent'], df['avg_coverage'] * 100.0, color=colors[:len(df)], edgecolor='black', alpha=0.8)
    axes[0].set_title("Average Discovery Coverage (%)", fontsize=12, fontweight='bold')
    axes[0].set_ylabel("Coverage (%)")
    axes[0].set_ylim(0, 105)
    
    # Subplot 2: Average Steps (Efficiency)
    axes[1].bar(df['agent'], df['avg_steps'], color=colors[:len(df)], edgecolor='black', alpha=0.8)
    axes[1].set_title("Average Steps (lower is better)", fontsize=12, fontweight='bold')
    axes[1].set_ylabel("Steps taken")
    
    # Subplot 3: Accumulated Reward
    axes[2].bar(df['agent'], df['avg_reward'], color=colors[:len(df)], edgecolor='black', alpha=0.8)
    axes[2].set_title("Accumulated Reward (policy efficiency)", fontsize=12, fontweight='bold')
    axes[2].set_ylabel("Cumulative Reward")
    
    for ax in axes:
        ax.tick_params(axis='x', rotation=30)
        ax.grid(True, linestyle=':', alpha=0.6)
        
    plt.suptitle(f"AIR-NDF Agent Benchmark Profiles ({scale.capitalize()} Network)", fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"comparison_{scale}.png"), dpi=300)
    plt.close()
    
    print(f"Generated and saved comparative plots in {output_dir}")

if __name__ == "__main__":
    # If run directly, evaluate on small network for validation
    run_benchmarks("small", num_episodes=5)
