import pytest
import gymnasium as gym
import numpy as np
import networkx as nx
import os
import sys

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.environment.net_env import NetworkDiscoveryEnv
from src.utils.generator import NetworkGenerator

def test_environment_initialization():
    env = NetworkDiscoveryEnv(max_steps=50)
    assert env.max_steps == 50
    assert env.action_space.n == 5
    assert env.observation_space.shape == (24,)

def test_environment_reset():
    # Setup test graph
    gen = NetworkGenerator(seed=123)
    G = gen.generate("small")
    env = NetworkDiscoveryEnv(graph=G)
    
    obs, info = env.reset(seed=42)
    
    assert obs.shape == (24,)
    assert info["current_node"] in G.nodes()
    assert info["explored_count"] == 1
    assert info["discovered_count"] >= 1
    assert env.current_step == 0
    assert not env.detected

def test_environment_steps():
    gen = NetworkGenerator(seed=123)
    G = gen.generate("small")
    env = NetworkDiscoveryEnv(graph=G, max_steps=10)
    obs, info = env.reset(seed=42)
    
    # Try all actions and verify bounds
    for action in range(5):
        obs, reward, terminated, truncated, info = env.step(action)
        assert obs.shape == (24,)
        assert isinstance(reward, float)
        assert isinstance(terminated, bool)
        assert isinstance(truncated, bool)
        
        # Reset if ended
        if terminated or truncated:
            obs, info = env.reset()

def test_safe_termination_action():
    env = NetworkDiscoveryEnv()
    obs, info = env.reset()
    
    # Action 4 is self-termination
    obs, reward, terminated, truncated, info = env.step(4)
    
    assert terminated
    assert not truncated
    assert reward > -env.step_cost  # Should receive exit bonus
