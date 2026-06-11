import argparse
import os
import sys

# Ensure project root is in python path
sys.path.append(os.path.abspath(os.path.dirname(__file__)))

from src.datasets.generate_datasets import generate_all
from src.utils.trainer import train_agent
from src.evaluation.benchmark import run_benchmarks

def main():
    parser = argparse.ArgumentParser(description="AIR-NDF: AI-Reinforced Adaptive Network Discovery Framework Experiment Automation")
    parser.add_argument("--generate", action="store_true", help="Generate all synthetic network datasets")
    parser.add_argument("--train", action="store_true", help="Train DQN, PPO, and A2C agents")
    parser.add_argument("--train-agent", type=str, choices=["DQN", "PPO", "A2C"], help="Train a specific agent")
    parser.add_argument("--episodes", type=int, default=300, help="Number of training episodes")
    parser.add_argument("--scale", type=str, default="medium", choices=["small", "medium", "large", "cloud-hybrid"], help="Network topology scale")
    parser.add_argument("--benchmark", action="store_true", help="Run comparative benchmark evaluation")
    parser.add_argument("--all", action="store_true", help="Run entire pipeline: generate -> train all -> benchmark")
    
    args = parser.parse_args()
    
    if len(sys.argv) == 1:
        parser.print_help()
        return

    # 1. Dataset Generation
    if args.generate or args.all:
        print("\n=== STAGE 1: Generating Synthetic Networks ===")
        generate_all()
        
    # 2. Agent Training
    if args.all:
        print("\n=== STAGE 2: Training RL Agents (DQN, PPO, A2C) ===")
        train_agent("DQN", network_scale=args.scale, episodes=args.episodes)
        train_agent("PPO", network_scale=args.scale, episodes=args.episodes)
        train_agent("A2C", network_scale=args.scale, episodes=args.episodes)
    elif args.train:
        print("\n=== STAGE 2: Training All RL Agents ===")
        train_agent("DQN", network_scale=args.scale, episodes=args.episodes)
        train_agent("PPO", network_scale=args.scale, episodes=args.episodes)
        train_agent("A2C", network_scale=args.scale, episodes=args.episodes)
    elif args.train_agent:
        print(f"\n=== STAGE 2: Training {args.train_agent} Agent ===")
        train_agent(args.train_agent, network_scale=args.scale, episodes=args.episodes)
        
    # 3. Benchmarking
    if args.benchmark or args.all:
        print("\n=== STAGE 3: Running Benchmark Evaluation ===")
        run_benchmarks(network_scale=args.scale, num_episodes=50)
        
    print("\nAIR-NDF Pipeline Completed Successfully!")

if __name__ == "__main__":
    main()
