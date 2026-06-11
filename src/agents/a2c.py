import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from typing import Dict, Any, List, Tuple
from src.agents.base import BaseAgent
from src.models.networks import ActorCritic

class A2CAgent(BaseAgent):
    """
    Advantage Actor-Critic (A2C) Agent.
    Updates the Actor-Critic networks synchronously using calculated advantages.
    """
    def __init__(self, 
                 state_dim: int = 24, 
                 action_dim: int = 5, 
                 lr: float = 7e-4, 
                 gamma: float = 0.99, 
                 value_coef: float = 0.5, 
                 entropy_coef: float = 0.01, 
                 device: str = "cpu"):
        super(A2CAgent, self).__init__(state_dim, action_dim, device)
        
        self.gamma = gamma
        self.value_coef = value_coef
        self.entropy_coef = entropy_coef
        
        # Actor-Critic network
        self.ac = ActorCritic(state_dim, action_dim).to(self.device)
        self.optimizer = optim.Adam(self.ac.parameters(), lr=lr)
        
        # Trajectory lists
        self.states = []
        self.actions = []
        self.rewards = []
        self.next_states = []
        self.dones = []

    def select_action(self, state: np.ndarray, evaluation: bool = False) -> int:
        state_t = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        with torch.no_grad():
            logits, _ = self.ac(state_t)
            probs = torch.softmax(logits, dim=-1)
            dist = torch.distributions.Categorical(probs)
            
            if evaluation:
                action = int(probs.argmax(dim=-1).item())
            else:
                action = int(dist.sample().item())
        return action

    def store_transition(self, state: np.ndarray, action: int, reward: float, next_state: np.ndarray, done: bool):
        self.states.append(state)
        self.actions.append(action)
        self.rewards.append(reward)
        self.next_states.append(next_state)
        self.dones.append(done)

    def clear_buffer(self):
        self.states.clear()
        self.actions.clear()
        self.rewards.clear()
        self.next_states.clear()
        self.dones.clear()

    def update(self) -> Dict[str, float]:
        """Calculates policy gradient and value updates, optimizing parameters."""
        if len(self.states) == 0:
            return {"loss": 0.0, "actor_loss": 0.0, "critic_loss": 0.0, "entropy": 0.0}

        # Convert lists to numpy arrays first, then tensors
        states_t = torch.FloatTensor(np.array(self.states)).to(self.device)
        actions_t = torch.LongTensor(self.actions).to(self.device)
        rewards_t = torch.FloatTensor(self.rewards).to(self.device)
        next_states_t = torch.FloatTensor(np.array(self.next_states)).to(self.device)
        dones_t = torch.FloatTensor(self.dones).to(self.device)
        
        # Evaluate states
        logits, values = self.ac(states_t)
        values = values.squeeze(1)
        
        # Evaluate next states
        with torch.no_grad():
            _, next_values = self.ac(next_states_t)
            next_values = next_values.squeeze(1)
            
        # Target returns and advantages
        returns = rewards_t + self.gamma * next_values * (1 - dones_t)
        advantages = returns - values
        
        # Action distributions
        probs = torch.softmax(logits, dim=-1)
        dist = torch.distributions.Categorical(probs)
        log_probs = dist.log_prob(actions_t)
        entropy = dist.entropy().mean()
        
        # Loss terms
        actor_loss = -(log_probs * advantages.detach()).mean()
        critic_loss = nn.MSELoss()(values, returns)
        
        total_loss = actor_loss + self.value_coef * critic_loss - self.entropy_coef * entropy
        
        # Backprop
        self.optimizer.zero_grad()
        total_loss.backward()
        nn.utils.clip_grad_norm_(self.ac.parameters(), max_norm=0.5)
        self.optimizer.step()
        
        # Clean buffer
        self.clear_buffer()
        
        return {
            "loss": total_loss.item(),
            "actor_loss": actor_loss.item(),
            "critic_loss": critic_loss.item(),
            "entropy": entropy.item()
        }

    def save(self, filepath: str):
        torch.save({
            'ac_state_dict': self.ac.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict()
        }, filepath)

    def load(self, filepath: str):
        checkpoint = torch.load(filepath, map_location=self.device)
        self.ac.load_state_dict(checkpoint['ac_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
