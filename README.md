# AI-Reinforced Adaptive Network Discovery Framework (AIR-NDF)

AIR-NDF is a complete, publication-grade research framework that utilizes Reinforcement Learning (RL) to optimize network discovery strategies. By framing network discovery as a Markov Decision Process (MDP), the system trains RL agents to explore simulated enterprise networks, maximizing topological information gain while minimizing exploration overhead and defensive detection risk.

---

## 🏗️ Project Architecture

```text
├── docs/                     # Tutorials and documentations
├── src/                      # Source Code
│   ├── environment/          # Custom Gymnasium network discovery environment
│   ├── agents/               # RL Agents (DQN, PPO, A2C)
│   ├── models/               # PyTorch deep network architectures
│   ├── evaluation/           # Baseline search algorithms (BFS/DFS/Greedy) and benchmarks
│   ├── datasets/             # Pre-generated synthetic network topologies
│   ├── utils/                # Network generator, trainers, and loaders
│   ├── static/               # Premium HTML dashboard interface
│   └── main.py               # FastAPI server application
├── research/                 # Publication materials (IEEE conference paper.md)
├── tests/                    # PyTest unit testing suites
├── results/                  # Checkpoints, charts, logs, and CSV reports
├── run_experiments.py        # CLI orchestrator script
├── requirements.txt          # Python dependencies
└── README.md                 # Project Documentation
```

---

## 🚀 Getting Started

### 1. Prerequisites
Ensure you have **Python 3.12+** installed on your system.

### 2. Set Up Virtual Environment & Install Dependencies
Create a clean python virtual environment and install dependencies:
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 3. Pre-generate Synthetic Datasets
Generate network topologies of different sizes (Small, Medium, Large, and Cloud-Hybrid):
```bash
python src/datasets/generate_datasets.py
```

---

## 📊 Run Experiments (CLI)

The CLI tool `run_experiments.py` orchestrates the complete workflow:

```bash
# Get help
python run_experiments.py --help

# Generate networks, train all RL agents for 150 episodes, and run evaluations
python run_experiments.py --all --episodes 150 --scale small

# Train a specific agent on a medium enterprise scale network
python run_experiments.py --train-agent PPO --scale medium --episodes 300

# Evaluate trained agents against traditional search baselines (Random, BFS, DFS, Greedy)
python run_experiments.py --benchmark --scale small
```

---

## 🖥️ Launch Interactive Web Dashboard

To run the interactive cyber-themed dashboard (powered by FastAPI, Vis.js, and Chart.js):

```bash
python src/main.py
```
After launching the server, open your browser and navigate to:
👉 **[http://localhost:8000](http://localhost:8000)**

From the dashboard, you can:
*   **Visualize Topologies**: Render and inspect 2D/3D graphs of small, medium, large, and cloud-hybrid networks.
*   **Train Agents**: Trigger background training for DQN, PPO, or A2C, and monitor rewards/loss curves in real time.
*   **Run Benchmarks**: Trigger comparative evaluations to see risk-discovery tradeoffs and performance bar plots.
*   **Study Theory**: Read the mathematical MDP state/action formulations.

---

## 🧪 Run Automated Tests

Run PyTest unit and integration tests to verify Gymnasium transitions and PyTorch networks:
```bash
pytest tests/
```
All unit tests should complete successfully, confirming environment validation and agent functionality.
# AI-Reinforced-Network-Discovery-Framework-AIR-NDF-
