import torch
import torch.nn as nn
import torch.nn.functional as F

class QNetwork(nn.Module):
    """
    Q-Network for Deep Q-Learning (DQN).
    Maps a 24-dimensional state vector to 5 discrete action Q-values.
    """
    def __init__(self, state_dim: int = 24, action_dim: int = 5, hidden_dim: int = 128):
        super(QNetwork, self).__init__()
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, action_dim)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)

class ActorCritic(nn.Module):
    """
    Actor-Critic network for policy gradient agents (PPO and A2C).
    Contains a shared feature extractor (or separate heads) that outputs
    action probabilities (logits) and state values.
    """
    def __init__(self, state_dim: int = 24, action_dim: int = 5, hidden_dim: int = 128):
        super(ActorCritic, self).__init__()
        # Shared trunk
        self.shared_fc1 = nn.Linear(state_dim, hidden_dim)
        self.shared_fc2 = nn.Linear(hidden_dim, hidden_dim)
        
        # Policy head (Actor)
        self.actor = nn.Linear(hidden_dim, action_dim)
        
        # Value head (Critic)
        self.critic = nn.Linear(hidden_dim, 1)
        
    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        x = F.relu(self.shared_fc1(x))
        x = F.relu(self.shared_fc2(x))
        
        logits = self.actor(x)
        value = self.critic(x)
        
        return logits, value
        
    def get_action_distribution(self, x: torch.Tensor) -> torch.distributions.Categorical:
        logits, _ = self.forward(x)
        probs = F.softmax(logits, dim=-1)
        return torch.distributions.Categorical(probs)
        
    def get_value(self, x: torch.Tensor) -> torch.Tensor:
        _, value = self.forward(x)
        return value
