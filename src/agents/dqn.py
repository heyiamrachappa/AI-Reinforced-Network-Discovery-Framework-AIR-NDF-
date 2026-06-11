import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
from collections import deque
from typing import Dict, Any, List, Tuple
from src.agents.base import BaseAgent
from src.models.networks import QNetwork

class ReplayBuffer:
    """Experience replay buffer for off-policy DQN training."""
    def __init__(self, capacity: int = 10000):
        self.buffer = deque(maxlen=capacity)

    def push(self, state: np.ndarray, action: int, reward: float, next_state: np.ndarray, done: bool):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        state, action, reward, next_state, done = zip(*random.sample(self.buffer, batch_size))
        return (np.array(state, dtype=np.float32), 
                np.array(action, dtype=np.int64), 
                np.array(reward, dtype=np.float32), 
                np.array(next_state, dtype=np.float32), 
                np.array(done, dtype=np.float32))

    def __len__(self) -> int:
        return len(self.buffer)


class DQNAgent(BaseAgent):
    """
    DQN Agent implementing experiences replay, double-Q concepts,
    target updates, and epsilon-greedy exploration.
    """
    def __init__(self, 
                 state_dim: int = 24, 
                 action_dim: int = 5, 
                 lr: float = 1e-3, 
                 gamma: float = 0.99, 
                 epsilon_start: float = 1.0, 
                 epsilon_end: float = 0.05, 
                 epsilon_decay: float = 0.995, 
                 buffer_size: int = 10000, 
                 batch_size: int = 64, 
                 target_update_freq: int = 200, 
                 device: str = "cpu"):
        super(DQNAgent, self).__init__(state_dim, action_dim, device)
        
        self.gamma = gamma
        self.batch_size = batch_size
        self.target_update_freq = target_update_freq
        
        # Epsilon-greedy schedule
        self.epsilon = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        
        # Networks
        self.q_network = QNetwork(state_dim, action_dim).to(self.device)
        self.target_network = QNetwork(state_dim, action_dim).to(self.device)
        self.target_network.load_state_dict(self.q_network.state_dict())
        self.target_network.eval()
        
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=lr)
        self.memory = ReplayBuffer(buffer_size)
        self.steps_done = 0

    def select_action(self, state: np.ndarray, evaluation: bool = False) -> int:
        # Epsilon-greedy exploration
        if not evaluation and random.random() < self.epsilon:
            return random.randint(0, self.action_dim - 1)
        
        state_t = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        with torch.no_grad():
            q_values = self.q_network(state_t)
            return int(q_values.argmax(dim=-1).item())

    def update(self) -> Dict[str, float]:
        """Runs a optimization gradient step on a batch sampled from ReplayBuffer."""
        if len(self.memory) < self.batch_size:
            return {"loss": 0.0}

        states, actions, rewards, next_states, dones = self.memory.sample(self.batch_size)
        
        # Convert to PyTorch tensors
        states_t = torch.FloatTensor(states).to(self.device)
        actions_t = torch.LongTensor(actions).unsqueeze(1).to(self.device)
        rewards_t = torch.FloatTensor(rewards).unsqueeze(1).to(self.device)
        next_states_t = torch.FloatTensor(next_states).to(self.device)
        dones_t = torch.FloatTensor(dones).unsqueeze(1).to(self.device)
        
        # Current Q-values
        curr_q = self.q_network(states_t).gather(1, actions_t)
        
        # Target Q-values (Double-DQN approach)
        with torch.no_grad():
            # Get best action for next state from current Q network
            next_actions = self.q_network(next_states_t).argmax(dim=-1, keepdim=True)
            # Evaluate that action using the target network
            next_q = self.target_network(next_states_t).gather(1, next_actions)
            target_q = rewards_t + (self.gamma * next_q * (1 - dones_t))
            
        # Compute loss
        loss = nn.MSELoss()(curr_q, target_q)
        
        # Step optimizer
        self.optimizer.zero_grad()
        loss.backward()
        # Gradient clipping for stability
        nn.utils.clip_grad_norm_(self.q_network.parameters(), max_norm=1.0)
        self.optimizer.step()
        
        self.steps_done += 1
        
        # Target network update
        if self.steps_done % self.target_update_freq == 0:
            self.target_network.load_state_dict(self.q_network.state_dict())
            
        return {"loss": loss.item()}

    def decay_epsilon(self):
        """Decays exploration parameter over episodes."""
        self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)

    def save(self, filepath: str):
        torch.save({
            'q_network_state_dict': self.q_network.state_dict(),
            'target_network_state_dict': self.target_network.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'epsilon': self.epsilon,
            'steps_done': self.steps_done
        }, filepath)

    def load(self, filepath: str):
        checkpoint = torch.load(filepath, map_location=self.device)
        self.q_network.load_state_dict(checkpoint['q_network_state_dict'])
        self.target_network.load_state_dict(checkpoint['target_network_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.epsilon = checkpoint['epsilon']
        self.steps_done = checkpoint['steps_done']
