import random
import networkx as nx
from typing import Dict, Any, List, Set
from src.environment.net_env import NetworkDiscoveryEnv

class BaselineAgent:
    """
    Base class for baseline non-RL discovery algorithms.
    """
    def __init__(self, action_dim: int = 5):
        self.action_dim = action_dim

    def select_action(self, env: NetworkDiscoveryEnv) -> int:
        raise NotImplementedError


class RandomSearchAgent(BaselineAgent):
    """
    Random exploration agent. Selects random exploration macro-actions.
    Terminates randomly if risk gets too high.
    """
    def select_action(self, env: NetworkDiscoveryEnv) -> int:
        # If risk is extremely high, terminate with some probability
        if env.accumulated_risk > 3.0 and random.random() < 0.3:
            return 4  # Terminate
            
        # Randomly choose between Explore Adjacent, Jump, Subnet, or Cluster
        return random.randint(0, 3)


class BFSAgent(BaselineAgent):
    """
    Breadth-First Search discovery agent.
    Focuses on local exploration first (Adjacent), then expands to the local subnet,
    then cluster, and finally jumps when local options are exhausted.
    """
    def select_action(self, env: NetworkDiscoveryEnv) -> int:
        # Terminate if all nodes explored
        if len(env.explored_nodes) >= len(env.G):
            return 4
            
        # 1. Check for unexplored adjacent nodes (local BFS)
        unexplored_neighbors = [n for n in env.G.neighbors(env.current_node) if n not in env.explored_nodes]
        if unexplored_neighbors:
            return 0  # Explore Adjacent
            
        # 2. Check for unexplored subnet nodes
        curr_attr = env.G.nodes[env.current_node]
        curr_subnet = curr_attr.get("subnet", "")
        subnet_nodes = [n for n, attr in env.G.nodes(data=True) if attr.get("subnet") == curr_subnet]
        unexplored_subnet = [n for n in subnet_nodes if n in env.discovered_nodes and n not in env.explored_nodes]
        if unexplored_subnet:
            return 2  # Prioritize Subnet
            
        # 3. Check for unexplored cluster nodes
        curr_cluster = next((comm for comm in env.communities if env.current_node in comm), set())
        unexplored_cluster = [n for n in curr_cluster if n in env.discovered_nodes and n not in env.explored_nodes]
        if unexplored_cluster:
            return 3  # Investigate Cluster
            
        # 4. Fallback: Jump to inferred node
        inferred = [n for n in env.discovered_nodes if n not in env.explored_nodes]
        if inferred:
            return 1  # Jump
            
        return 4  # Terminate if no actions available


class DFSAgent(BaselineAgent):
    """
    Depth-First Search discovery agent.
    Dives deep by preferring to continue along the adjacent path.
    If it hits a dead end (no adjacent unexplored), it jumps to the most
    recently discovered unexplored node (using Jump or Subnet).
    """
    def select_action(self, env: NetworkDiscoveryEnv) -> int:
        if len(env.explored_nodes) >= len(env.G):
            return 4
            
        # DFS prefers going adjacent (Action 0) as much as possible
        unexplored_neighbors = [n for n in env.G.neighbors(env.current_node) if n not in env.explored_nodes]
        if unexplored_neighbors:
            return 0  # Keep going deeper
            
        # Dead end: Jump to back-track/move to a different subnet
        # We try cluster, subnet, then jump
        curr_attr = env.G.nodes[env.current_node]
        curr_subnet = curr_attr.get("subnet", "")
        subnet_nodes = [n for n, attr in env.G.nodes(data=True) if attr.get("subnet") == curr_subnet]
        unexplored_subnet = [n for n in subnet_nodes if n in env.discovered_nodes and n not in env.explored_nodes]
        if unexplored_subnet:
            return 2
            
        inferred = [n for n in env.discovered_nodes if n not in env.explored_nodes]
        if inferred:
            return 1
            
        return 4


class GreedySearchAgent(BaselineAgent):
    """
    Greedy Search agent.
    Inspects the frontier of discovered-but-unexplored nodes and selects
    the action that targets the node with the highest vulnerability / reward profile,
    while terminating early if detection risk is too high.
    """
    def select_action(self, env: NetworkDiscoveryEnv) -> int:
        if len(env.explored_nodes) >= len(env.G):
            return 4
            
        # Risk management: if risk is high, terminate
        if env.accumulated_risk > 4.5:
            return 4
            
        # Find all unexplored candidate nodes and identify the best one
        unexplored_candidates = [n for n in env.discovered_nodes if n not in env.explored_nodes]
        if not unexplored_candidates:
            return 4
            
        # Target node with highest vulnerability score
        best_node = max(unexplored_candidates, key=lambda n: env.G.nodes[n].get("vulnerability_score", 0.0))
        
        # Determine how to reach this best node
        # 1. Is it adjacent?
        if best_node in env.G.neighbors(env.current_node):
            return 0
            
        # 2. Is it in the same subnet?
        curr_subnet = env.G.nodes[env.current_node].get("subnet", "")
        target_subnet = env.G.nodes[best_node].get("subnet", "")
        if curr_subnet == target_subnet:
            return 2
            
        # 3. Is it in the same cluster?
        curr_cluster = next((comm for comm in env.communities if env.current_node in comm), set())
        if best_node in curr_cluster:
            return 3
            
        # 4. Jump
        return 1
