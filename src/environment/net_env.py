import gymnasium as gym
from gymnasium import spaces
import numpy as np
import networkx as nx
import random
from typing import Dict, List, Tuple, Any, Set
import os

class NetworkDiscoveryEnv(gym.Env):
    """
    Custom Gymnasium Environment for simulating network discovery.
    State representation: A 24-dimensional feature vector.
    Action representation: Discrete(5) macro-actions.
    """
    metadata = {"render_modes": ["human"]}

    def __init__(self, 
                 graph: nx.Graph = None, 
                 max_steps: int = 100, 
                 step_cost: float = 2.0, 
                 detection_penalty: float = 300.0,
                 entry_point_type: str = "web_server"):
        super(NetworkDiscoveryEnv, self).__init__()
        
        self.G = graph
        self.max_steps = max_steps
        self.step_cost = step_cost
        self.detection_penalty = detection_penalty
        self.entry_point_type = entry_point_type
        
        # 5 Macro-actions:
        # 0: Explore Adjacent Node
        # 1: Jump to Inferred Node
        # 2: Prioritize Subnet
        # 3: Investigate Cluster
        # 4: Terminate Exploration
        self.action_space = spaces.Discrete(5)
        
        # 24-dimensional float state vector
        self.observation_space = spaces.Box(
            low=-np.inf, 
            high=np.inf, 
            shape=(24,), 
            dtype=np.float32
        )
        
        # Initialize internal state vars
        self.current_node = None
        self.discovered_nodes: Set[int] = set()
        self.explored_nodes: Set[int] = set()
        self.current_step = 0
        self.accumulated_risk = 0.0
        self.cumulative_reward = 0.0
        self.detected = False
        self.communities: List[Set[int]] = []

    def set_graph(self, graph: nx.Graph):
        """Allows dynamic updates of the target network graph."""
        self.G = graph

    def _compute_communities(self):
        """Computes network communities/clusters to support cluster actions."""
        try:
            # Try to use NetworkX community detection
            import networkx.algorithms.community as nx_comm
            comms = list(nx_comm.louvain_communities(self.G, seed=42))
            self.communities = comms
        except Exception:
            # Fallback to connected components if louvain is not available or errors out
            self.communities = list(nx.connected_components(self.G))

    def reset(self, seed: int = None, options: Dict[str, Any] = None) -> Tuple[np.ndarray, Dict[str, Any]]:
        super().reset(seed=seed)
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)
            
        if self.G is None:
            # If no graph is set, generate a default small enterprise network
            from src.utils.generator import NetworkGenerator
            gen = NetworkGenerator(seed=seed if seed is not None else 42)
            self.G = gen.generate("small")
            
        self._compute_communities()
        
        # Reset simulation state variables
        self.current_step = 0
        self.accumulated_risk = 0.0
        self.cumulative_reward = 0.0
        self.detected = False
        self.explored_nodes = set()
        self.discovered_nodes = set()
        
        # Select entry point: node of entry_point_type if possible, otherwise random node
        entry_nodes = [n for n, attr in self.G.nodes(data=True) if attr.get("host_type") == self.entry_point_type]
        if not entry_nodes:
            entry_nodes = list(self.G.nodes())
            
        self.current_node = random.choice(entry_nodes)
        
        # Mark entry node as discovered and explored
        self.explored_nodes.add(self.current_node)
        self.discovered_nodes.add(self.current_node)
        
        # Discovered adjacent nodes from entry point
        for neighbor in self.G.neighbors(self.current_node):
            self.discovered_nodes.add(neighbor)
            
        obs = self._get_observation()
        info = self._get_info()
        
        return obs, info

    def _get_observation(self) -> np.ndarray:
        """Constructs the 24-dimensional standardized state representation."""
        num_nodes = len(self.G)
        curr_attr = self.G.nodes[self.current_node]
        
        # Node Type Mapping
        type_mapping = {
            "router": 0.1, "firewall": 0.2, "workstation": 0.3, "iot_device": 0.4,
            "cloud_vm": 0.5, "cloud_storage": 0.6, "web_server": 0.7, 
            "database_server": 0.8, "active_directory": 0.9
        }
        curr_type_val = type_mapping.get(curr_attr.get("host_type", "workstation"), 0.5)
        
        # Topology metrics
        deg = self.G.degree(self.current_node)
        unexplored_neighbors = [n for n in self.G.neighbors(self.current_node) if n not in self.explored_nodes]
        
        # Subnet calculations
        curr_subnet = curr_attr.get("subnet", "DMZ")
        subnet_nodes = [n for n, attr in self.G.nodes(data=True) if attr.get("subnet") == curr_subnet]
        discovered_subnet = [n for n in subnet_nodes if n in self.discovered_nodes]
        unexplored_subnet = [n for n in subnet_nodes if n not in self.explored_nodes]
        
        # Cluster/Community calculations
        curr_cluster = next((comm for comm in self.communities if self.current_node in comm), set())
        unexplored_cluster = [n for n in curr_cluster if n not in self.explored_nodes]
        
        # Unexplored overall
        unexplored_total = [n for n in self.G.nodes() if n not in self.explored_nodes]
        
        # Mean vulnerabilities
        mean_vuln_adj = np.mean([self.G.nodes[n].get("vulnerability_score", 0.0) for n in unexplored_neighbors]) if unexplored_neighbors else 0.0
        mean_vuln_sub = np.mean([self.G.nodes[n].get("vulnerability_score", 0.0) for n in unexplored_subnet]) if unexplored_subnet else 0.0
        mean_vuln_clust = np.mean([self.G.nodes[n].get("vulnerability_score", 0.0) for n in unexplored_cluster]) if unexplored_cluster else 0.0
        
        # Max rewards
        max_rew_sub = max([self.G.nodes[n].get("discovery_reward", 0.0) for n in unexplored_subnet]) if unexplored_subnet else 0.0
        max_rew_clust = max([self.G.nodes[n].get("discovery_reward", 0.0) for n in unexplored_cluster]) if unexplored_cluster else 0.0
        
        # Pack state vector
        state = np.zeros(24, dtype=np.float32)
        state[0] = self.current_node / num_nodes
        state[1] = curr_type_val
        state[2] = curr_attr.get("vulnerability_score", 0.0)
        state[3] = curr_attr.get("detection_probability", 0.0)
        state[4] = curr_attr.get("discovery_reward", 0.0) / 300.0
        state[5] = len(self.discovered_nodes) / num_nodes
        state[6] = len(self.explored_nodes) / num_nodes
        state[7] = min(self.accumulated_risk, 10.0) / 10.0  # Normalized
        state[8] = self.current_step / self.max_steps
        state[9] = deg / 50.0  # Capped at 50 neighbors
        state[10] = len(unexplored_neighbors) / 50.0
        state[11] = len(discovered_subnet) / max(len(subnet_nodes), 1)
        state[12] = len(unexplored_subnet) / num_nodes
        state[13] = len([n for n in unexplored_subnet if self.G.nodes[n].get("vulnerability_score", 0.0) > 0.5]) / num_nodes
        state[14] = len(curr_cluster) / num_nodes
        state[15] = len(unexplored_cluster) / num_nodes
        state[16] = len(unexplored_total) / num_nodes
        state[17] = mean_vuln_adj
        state[18] = mean_vuln_sub
        state[19] = mean_vuln_clust
        state[20] = max_rew_sub / 300.0
        state[21] = max_rew_clust / 300.0
        state[22] = self.cumulative_reward / 2000.0  # Scaled expectation
        state[23] = max(0.0, self.max_steps - self.current_step) / self.max_steps
        
        return state

    def _get_info(self) -> Dict[str, Any]:
        return {
            "current_node": self.current_node,
            "discovered_count": len(self.discovered_nodes),
            "explored_count": len(self.explored_nodes),
            "accumulated_risk": self.accumulated_risk,
            "detected": self.detected,
            "step": self.current_step
        }

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        self.current_step += 1
        
        # 1. Check for Termination Action
        if action == 4:
            # Agent self-terminates. Grant a reward bonus for clean exit.
            safe_exit_bonus = 20.0 * (1.0 - (self.accumulated_risk / 5.0))
            reward = safe_exit_bonus - self.step_cost
            self.cumulative_reward += reward
            obs = self._get_observation()
            info = self._get_info()
            return obs, reward, True, False, info
            
        target_node = None
        redundant_action = False
        
        # 2. Select target based on Action Class
        curr_attr = self.G.nodes[self.current_node]
        
        if action == 0:  # Explore Adjacent Node
            unexplored_neighbors = [n for n in self.G.neighbors(self.current_node) if n not in self.explored_nodes]
            if unexplored_neighbors:
                # Prioritize based on highest vulnerability
                target_node = max(unexplored_neighbors, key=lambda n: self.G.nodes[n].get("vulnerability_score", 0.0))
            else:
                redundant_action = True
                
        elif action == 1:  # Jump to Inferred Node
            # Focus on discovered but unexplored nodes in the network that are not adjacent
            adjacents = set(self.G.neighbors(self.current_node))
            inferred = [n for n in self.discovered_nodes if n not in self.explored_nodes and n != self.current_node and n not in adjacents]
            if inferred:
                target_node = max(inferred, key=lambda n: self.G.nodes[n].get("vulnerability_score", 0.0))
            else:
                # If no inferred available, fall back to any discovered and unexplored node
                candidates = [n for n in self.discovered_nodes if n not in self.explored_nodes and n != self.current_node]
                if candidates:
                    target_node = max(candidates, key=lambda n: self.G.nodes[n].get("vulnerability_score", 0.0))
                else:
                    redundant_action = True
                    
        elif action == 2:  # Prioritize Subnet
            curr_subnet = curr_attr.get("subnet", "DMZ")
            subnet_nodes = [n for n, attr in self.G.nodes(data=True) if attr.get("subnet") == curr_subnet]
            # Must be discovered and unexplored
            candidates = [n for n in subnet_nodes if n in self.discovered_nodes and n not in self.explored_nodes]
            if candidates:
                target_node = max(candidates, key=lambda n: self.G.nodes[n].get("vulnerability_score", 0.0))
            else:
                redundant_action = True
                
        elif action == 3:  # Investigate Cluster
            curr_cluster = next((comm for comm in self.communities if self.current_node in comm), set())
            candidates = [n for n in curr_cluster if n in self.discovered_nodes and n not in self.explored_nodes]
            if candidates:
                target_node = max(candidates, key=lambda n: self.G.nodes[n].get("vulnerability_score", 0.0))
            else:
                redundant_action = True

        # 3. Calculate Reward & Transition State
        reward = 0.0
        
        if redundant_action or target_node is None:
            # Redundant action penalty
            reward = -10.0 - self.step_cost
        else:
            # Action succeeds: probe target_node
            node_attr = self.G.nodes[target_node]
            vuln_score = node_attr.get("vulnerability_score", 0.0)
            det_prob = node_attr.get("detection_probability", 0.0)
            disc_reward = node_attr.get("discovery_reward", 0.0)
            
            # Simulated vulnerability exploration test:
            # We succeed if we exploit/discover the node based on its vulnerability score.
            # However, for simplicity of topology mapping, let's say the scan always succeeds in mapping it,
            # but the reward gained is scaled by the vulnerability score (representing ease of access).
            discovery_gain = disc_reward * (0.5 + 0.5 * vuln_score)
            
            # Risk check: roll for detection based on target's detection probability
            # With dynamic reward shaping, we also penalize step cost based on node risk
            risk_penalty = -5.0 * det_prob
            
            # Simulated Detection Event check
            roll = random.random()
            if roll < det_prob:
                # Agent is detected!
                self.detected = True
                reward = -self.detection_penalty
                self.cumulative_reward += reward
                self.explored_nodes.add(target_node)
                self.discovered_nodes.add(target_node)
                # Terminate episode immediately
                obs = self._get_observation()
                info = self._get_info()
                return obs, reward, True, False, info
            
            # Standard successful scan
            self.explored_nodes.add(target_node)
            self.discovered_nodes.add(target_node)
            self.current_node = target_node
            
            # Discovered its neighbors
            for neighbor in self.G.neighbors(target_node):
                self.discovered_nodes.add(neighbor)
                
            self.accumulated_risk += det_prob
            reward = discovery_gain - self.step_cost + risk_penalty

        # 4. Check for Episode End Conditions (max steps, or all nodes explored)
        self.cumulative_reward += reward
        
        terminated = False
        truncated = False
        
        if self.current_step >= self.max_steps:
            truncated = True
            
        if len(self.explored_nodes) >= len(self.G):
            # All nodes in the network mapped
            terminated = True
            # Completion bonus
            completion_bonus = 100.0
            reward += completion_bonus
            self.cumulative_reward += completion_bonus
            
        obs = self._get_observation()
        info = self._get_info()
        
        return obs, reward, terminated, truncated, info

def validate_env():
    """Simple script to check environment setup."""
    env = NetworkDiscoveryEnv()
    obs, info = env.reset()
    print("Environment validation successful!")
    print("Obs shape:", obs.shape)
    print("Initial Info:", info)
    
    # Try a few steps
    for a in range(5):
        obs, rew, term, trunc, info = env.step(a)
        print(f"Action {a}: reward={rew:.2f}, term={term}, trunc={trunc}, current_node={info['current_node']}")
