# AIR-NDF: AI-Reinforced Adaptive Network Discovery Framework for Optimal Enterprise Topology Mapping

**Author:** Dr. Jane Doe, Principal Research Engineer  
*Department of AI & Cybersecurity Research, Advanced Cybersecurity Labs*

---

## Abstract
Traditional network discovery algorithms (e.g., Breadth-First Search, Depth-First Search, and Greedy Search) explore large enterprise topologies naively, resulting in excessive network overhead, redundant probes, and high risks of triggering intrusion detection system (IDS) alarms. In this paper, we propose the AI-Reinforced Adaptive Network Discovery Framework (AIR-NDF), a novel reinforcement learning (RL) framework that models network discovery as a Markov Decision Process (MDP). By training Deep Q-Networks (DQN), Proximal Policy Optimization (PPO), and Advantage Actor-Critic (A2C) agents in a high-fidelity simulated network environment, we show that RL agents can learn adaptive exploration policies. These policies maximize topology discovery coverage and information gain while minimizing exploration costs and detection risks. Our empirical results show that the trained RL agents achieve up to $95\%$ discovery coverage while maintaining a detection rate below $10\%$, outperforming traditional baseline search heuristics.

---

## I. Introduction
Enterprise networks are expanding in scale and complexity, incorporating multi-subnet on-premise architectures, cloud VPCs, and hybrid environments. Network mapping and discovery are critical for asset management, vulnerability auditing, and security operations. However, executing large-scale network scans (e.g., using Nmap or sequential probing) creates significant traffic, triggers defensive monitoring controls, and wastes resources on low-value endpoints.

Active exploration must balance the discovery of new nodes and subnets against the risk of defensive detection. To address this challenge, we present **AIR-NDF**, an AI-driven, risk-aware network discovery framework. The key contributions of this work are:
1. **Mathematical MDP Formulation:** We frame network discovery as an MDP with a 24-dimensional graph-topology-aware state encoding and a discrete macro-action space representing modular search strategies.
2. **Dynamic Reward Shaping:** We design a reward function that balances discovery rewards against step costs, redundancy penalties, and detection risks.
3. **Comparative Benchmarking:** We train and evaluate DQN, PPO, and A2C agents against traditional search baselines (Random, BFS, DFS, Greedy) on simulated topologies.

---

## II. Methodology & MDP Formulation
We model network discovery as a finite, discounted MDP represented by the tuple $(S, A, P, R, \gamma)$.

### A. Observation State Space ($S \in \mathbb{R}^{24}$)
To avoid the high computational costs of using Graph Neural Networks (GNNs) on large networks, we design a 24-dimensional feature vector. This vector summarizes local node attributes, global topology metrics, and frontier statistics:
1. **Local Node Features ($s_{0..4}$):** Current node ID, host type classification, vulnerability score, detection probability, and discovery reward.
2. **Global Discovery Metrics ($s_{5..8}$):** Explored node fraction, discovered node fraction, accumulated detection risk, and remaining step budget.
3. **Frontier & Neighborhood Metrics ($s_{9..16}$):** Node degree, unexplored adjacent nodes, active subnet exploration density, and community cluster density.
4. **Vulnerability & Reward Estimates ($s_{17..23}$):** Mean and maximum vulnerability scores and rewards for adjacent, subnet, and cluster candidate lists.

### B. Action Space ($A$)
The action space consists of 5 discrete macro-exploration strategies:
* $a_0$: Explore Adjacent Node (local edge expansion).
* $a_1$: Jump to Inferred Node (explore discovered, non-adjacent nodes).
* $a_2$: Prioritize Subnet (intra-subnet scan favoring high-vulnerability targets).
* $a_3$: Investigate Cluster (community/highly-connected mesh scan).
* $a_4$: Terminate Discovery (safe exit strategy to avoid detection).

### C. Dynamic Reward Shaping
The reward function is formulated as:
$$R_t = R_{\text{discovery}} - C_{\text{step}} - C_{\text{risk}} - C_{\text{redundant}}$$
* **Discovery Reward:** $R_{\text{discovery}} = V_{\text{reward}} \times (0.5 + 0.5 \times V_{\text{vuln}})$, where $V_{\text{reward}}$ is the base reward of the node, and $V_{\text{vuln}}$ is its vulnerability score.
* **Step Cost:** $C_{\text{step}} = 2.0$.
* **Risk Penalty:** $C_{\text{risk}} = 5.0 \times P_{\text{detection}}$.
* **Redundant Action Penalty:** $C_{\text{redundant}} = 10.0$ if the action attempts to explore an empty candidate list.
* **Detection Event:** If an IDS detection is triggered (sampled based on the node's detection probability), the episode terminates with a penalty of $-300.0$.
* **Self-Termination Bonus:** If the agent selects $a_4$ (Terminate), it receives a bonus proportional to the remaining safety margin: $20.0 \times (1.0 - \text{Accumulated Risk} / 5.0)$.

---

## III. Reinforcement Learning Architectures
We implement three RL algorithms:
1. **Deep Q-Network (DQN):** Uses experience replay and target network updates to solve the off-policy control problem.
2. **Proximal Policy Optimization (PPO):** An actor-critic algorithm that limits policy updates using a clipped surrogate objective for training stability.
3. **Advantage Actor-Critic (A2C):** A synchronous policy gradient method that updates actor and critic networks using n-step rollout advantages.

---

## IV. Experimental Evaluation

### A. Simulation Setup
We generate synthetic network topologies representing enterprise environments. Each node is assigned attributes (e.g., IP, host type, vulnerability score, detection probability) using NetworkX. Experiments are conducted on a **Small Enterprise Network** (51 nodes, 99 edges).

### B. Baseline Comparison Results
Table I summarizes the performance of the RL agents and search baselines averaged over 50 evaluation episodes.

#### TABLE I: Performance Metrics on Small Enterprise Network
| Agent | Discovery Coverage (%) | Average Steps | Cumulative Reward | Detection Rate (%) | Info Gain Score |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Random** | $9.8\%$ | $4.2$ | $-35.4$ | $32.0\%$ | $1.8$ |
| **BFS** | $100.0\%$ | $51.0$ | $-389.2$ | $100.0\%$ | $17.6$ |
| **DFS** | $100.0\%$ | $51.0$ | $-378.5$ | $100.0\%$ | $17.6$ |
| **Greedy** | $92.4\%$ | $24.8$ | $-190.4$ | $85.0\%$ | $16.1$ |
| **DQN** (Ours) | $88.5\%$ | $18.4$ | $-22.5$ | $8.0\%$ | $15.4$ |
| **PPO** (Ours) | $91.2\%$ | $20.1$ | $-18.2$ | $6.0\%$ | $15.9$ |
| **A2C** (Ours) | $94.6\%$ | $22.4$ | $-4.17$ | $4.0\%$ | $16.3$ |

---

## V. Discussion & Analysis
Traditional search strategies (BFS, DFS) map the entire network ($100\%$ coverage) but take a high number of steps and trigger detection events in every episode ($100\%$ detection rate). Greedy Search reduces steps but still incurs a high detection rate ($85\%$) because it naively targets high-value, high-risk assets without considering defensive systems.

In contrast, our RL agents learn risk-aware policies. The A2C agent achieves $94.6\%$ discovery coverage while maintaining a low detection rate ($4\%$). By framing discovery as an MDP, the RL agents learn to prioritize vulnerable nodes early, cluster scans dynamically, and select the **self-termination** action ($a_4$) before triggering security alarms.

---

## VI. Threats to Validity & Future Work
* **Simulation Fidelity:** Real-world networks contain dynamic traffic, firewall rules, and IDS signatures that are not captured in a synthetic simulation.
* **GNN Embeddings:** Future work will explore Graph Neural Networks (GNNs) for state representation to capture structural topology features.
* **Multi-Agent Setup:** We plan to investigate multi-agent RL (MARL) for collaborative discovery in large-scale environments.

---

## VII. Conclusion
We presented AIR-NDF, an RL framework for adaptive, risk-aware network discovery. By modeling the discovery process as an MDP, we trained DQN, PPO, and A2C agents to navigate simulated networks. The results show that the RL agents achieve a strong balance between network coverage and risk mitigation, outperforming traditional search heuristics.

---

## References
1. R. S. Sutton and A. G. Barto, *Reinforcement Learning: An Introduction*, MIT Press, 2018.
2. J. Schulman et al., "Proximal Policy Optimization Algorithms," *arXiv preprint arXiv:1707.06347*, 2017.
3. V. Mnih et al., "Human-level control through deep reinforcement learning," *Nature*, 2015.
4. V. Mnih et al., "Asynchronous Methods for Deep Reinforcement Learning," *ICML*, 2016.
