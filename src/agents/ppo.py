import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from typing import Dict, Any, List, Tuple
from src.agents.base import BaseAgent
from src.models.networks import ActorCritic

class PPOAgent(BaseAgent):
    """
    PPO Agent with Clipped Policy Objective, value loss clipping,
    and Generalized Advantage Estimation (GAE).
    """
    def __init__(self, 
                 state_dim: int = 24, 
                 action_dim: int = 5, 
                 lr: float = 3e-4, 
                 gamma: float = 0.99, 
                 gae_lambda: float = 0.95, 
                 clip_epsilon: float = 0.2, 
                 c1_value_coef: float = 0.5, 
                 c2_entropy_coef: float = 0.01, 
                 ppo_epochs: int = 10, 
                 batch_size: int = 64, 
                 device: str = "cpu"):
        super(PPOAgent, self).__init__(state_dim, action_dim, device)
        
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_epsilon = clip_epsilon
        self.c1_value_coef = c1_value_coef
        self.c2_entropy_coef = c2_entropy_coef
        self.ppo_epochs = ppo_epochs
        self.batch_size = batch_size
        
        # Actor-Critic network
        self.ac = ActorCritic(state_dim, action_dim).to(self.device)
        self.optimizer = optim.Adam(self.ac.parameters(), lr=lr)
        
        # Temporary trajectory buffer
        self.states = []
        self.actions = []
        self.log_probs = []
        self.rewards = []
        self.values = []
        self.dones = []

    def select_action(self, state: np.ndarray, evaluation: bool = False) -> int:
        state_t = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        with torch.no_grad():
            logits, value = self.ac(state_t)
            probs = torch.softmax(logits, dim=-1)
            dist = torch.distributions.Categorical(probs)
            
            if evaluation:
                # Greedy choice
                action = int(probs.argmax(dim=-1).item())
            else:
                # Sample choice
                action = int(dist.sample().item())
                
        return action

    def store_transition(self, state: np.ndarray, action: int, log_prob: float, reward: float, value: float, done: bool):
        """Stores experience tuple in current trajectory."""
        self.states.append(state)
        self.actions.append(action)
        self.log_probs.append(log_prob)
        self.rewards.append(reward)
        self.values.append(value)
        self.dones.append(done)

    def get_action_log_prob_and_value(self, state: np.ndarray) -> Tuple[int, float, float]:
        """Convenience method for environment rollout logging."""
        state_t = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        with torch.no_grad():
            logits, value = self.ac(state_t)
            probs = torch.softmax(logits, dim=-1)
            dist = torch.distributions.Categorical(probs)
            action = dist.sample()
            log_prob = dist.log_prob(action).item()
            
        return int(action.item()), float(log_prob), float(value.item())

    def clear_buffer(self):
        self.states.clear()
        self.actions.clear()
        self.log_probs.clear()
        self.rewards.clear()
        self.values.clear()
        self.dones.clear()

    def compute_gae(self, next_value: float, next_done: bool) -> Tuple[torch.Tensor, torch.Tensor]:
        """Computes Generalized Advantage Estimation (GAE) and returns targets."""
        rewards = self.rewards
        values = self.values + [next_value]
        dones = self.dones + [next_done]
        
        advantages = []
        gae = 0.0
        
        for step in reversed(range(len(rewards))):
            delta = rewards[step] + self.gamma * values[step + 1] * (1 - int(dones[step])) - values[step]
            gae = delta + self.gamma * self.gae_lambda * (1 - int(dones[step])) * gae
            advantages.insert(0, gae)
            
        advantages = torch.FloatTensor(advantages).to(self.device)
        values_t = torch.FloatTensor(self.values).to(self.device)
        returns = advantages + values_t
        
        # Standardize advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        return advantages, returns

    def update(self, next_value: float, next_done: bool) -> Dict[str, float]:
        """Performs PPO policy and value updates over collected trajectory."""
        if len(self.states) == 0:
            return {"actor_loss": 0.0, "critic_loss": 0.0, "entropy": 0.0}

        advantages, returns = self.compute_gae(next_value, next_done)
        
        # Convert lists to tensors
        states_t = torch.FloatTensor(np.array(self.states)).to(self.device)
        actions_t = torch.LongTensor(self.actions).to(self.device)
        old_log_probs_t = torch.FloatTensor(self.log_probs).to(self.device)
        
        loss_actor_accum = 0.0
        loss_critic_accum = 0.0
        entropy_accum = 0.0
        
        dataset_size = len(self.states)
        
        for _ in range(self.ppo_epochs):
            # Generate random mini-batch indices
            permutation = torch.randperm(dataset_size)
            for start_idx in range(0, dataset_size, self.batch_size):
                batch_indices = permutation[start_idx : start_idx + self.batch_size]
                if len(batch_indices) < 4:  # Skip too small batches
                    continue
                
                b_states = states_t[batch_indices]
                b_actions = actions_t[batch_indices]
                b_old_log_probs = old_log_probs_t[batch_indices]
                b_advantages = advantages[batch_indices]
                b_returns = returns[batch_indices]
                
                # Evaluate new policy
                logits, new_values = self.ac(b_states)
                new_values = new_values.squeeze(1)
                
                probs = torch.softmax(logits, dim=-1)
                dist = torch.distributions.Categorical(probs)
                
                new_log_probs = dist.log_prob(b_actions)
                entropy = dist.entropy().mean()
                
                # Policy ratio
                ratios = torch.exp(new_log_probs - b_old_log_probs)
                
                # Surrogates
                surr1 = ratios * b_advantages
                surr2 = torch.clamp(ratios, 1.0 - self.clip_epsilon, 1.0 + self.clip_epsilon) * b_advantages
                actor_loss = -torch.min(surr1, surr2).mean()
                
                # Critic value loss
                critic_loss = nn.MSELoss()(new_values, b_returns)
                
                # Total loss
                loss = actor_loss + self.c1_value_coef * critic_loss - self.c2_entropy_coef * entropy
                
                # Update
                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.ac.parameters(), max_norm=0.5)
                self.optimizer.step()
                
                loss_actor_accum += actor_loss.item()
                loss_critic_accum += critic_loss.item()
                entropy_accum += entropy.item()

        num_updates = self.ppo_epochs * max(1, dataset_size // self.batch_size)
        
        # Reset local trajectory buffers
        self.clear_buffer()
        
        return {
            "actor_loss": loss_actor_accum / num_updates,
            "critic_loss": loss_critic_accum / num_updates,
            "entropy": entropy_accum / num_updates
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
