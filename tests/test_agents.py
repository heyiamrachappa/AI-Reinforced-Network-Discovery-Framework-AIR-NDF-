import pytest
import numpy as np
import torch
import os
import sys

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.agents.dqn import DQNAgent
from src.agents.ppo import PPOAgent
from src.agents.a2c import A2CAgent

def test_dqn_agent():
    agent = DQNAgent(state_dim=24, action_dim=5, lr=1e-3, target_update_freq=5)
    state = np.random.randn(24).astype(np.float32)
    
    # Action selection
    action = agent.select_action(state, evaluation=True)
    assert 0 <= action < 5
    
    # Buffer insertion and update
    for _ in range(70):  # Push enough samples to satisfy batch_size (64)
        agent.memory.push(state, 1, 10.0, state, False)
        
    assert len(agent.memory) == 70
    update_info = agent.update()
    assert "loss" in update_info
    assert isinstance(update_info["loss"], float)
    
    # Save & Load
    checkpoint_path = "test_dqn_checkpoint.pt"
    agent.save(checkpoint_path)
    assert os.path.exists(checkpoint_path)
    
    new_agent = DQNAgent(state_dim=24, action_dim=5)
    new_agent.load(checkpoint_path)
    
    os.remove(checkpoint_path)

def test_ppo_agent():
    agent = PPOAgent(state_dim=24, action_dim=5, ppo_epochs=2, batch_size=10)
    state = np.random.randn(24).astype(np.float32)
    
    # Selection and storage
    action, log_prob, value = agent.get_action_log_prob_and_value(state)
    assert 0 <= action < 5
    
    # Run a few steps to fill trace
    for _ in range(15):
        agent.store_transition(state, action, log_prob, 5.0, value, False)
        
    # Run update
    update_info = agent.update(next_value=0.0, next_done=True)
    assert "actor_loss" in update_info
    assert "critic_loss" in update_info
    
    # Save & Load
    checkpoint_path = "test_ppo_checkpoint.pt"
    agent.save(checkpoint_path)
    assert os.path.exists(checkpoint_path)
    
    new_agent = PPOAgent(state_dim=24, action_dim=5)
    new_agent.load(checkpoint_path)
    os.remove(checkpoint_path)

def test_a2c_agent():
    agent = A2CAgent(state_dim=24, action_dim=5)
    state = np.random.randn(24).astype(np.float32)
    
    action = agent.select_action(state)
    assert 0 <= action < 5
    
    for _ in range(5):
        agent.store_transition(state, action, 2.0, state, False)
        
    update_info = agent.update()
    assert "loss" in update_info
    
    # Save & Load
    checkpoint_path = "test_a2c_checkpoint.pt"
    agent.save(checkpoint_path)
    assert os.path.exists(checkpoint_path)
    
    new_agent = A2CAgent(state_dim=24, action_dim=5)
    new_agent.load(checkpoint_path)
    os.remove(checkpoint_path)
