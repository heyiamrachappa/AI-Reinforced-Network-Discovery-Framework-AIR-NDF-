from abc import ABC, abstractmethod
import torch
import numpy as np
import os
from typing import Dict, Any

class BaseAgent(ABC):
    """
    Abstract Base Class for Reinforcement Learning agents in the AIR-NDF framework.
    Defines unified interfaces for training, action selection, checkpointing, and execution.
    """
    def __init__(self, state_dim: int = 24, action_dim: int = 5, device: str = "cpu"):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.device = torch.device(device if torch.cuda.is_available() and device == "cuda" else "cpu")

    @abstractmethod
    def select_action(self, state: np.ndarray, evaluation: bool = False) -> int:
        """Selects an action based on the state representation."""
        pass

    @abstractmethod
    def update(self, *args, **kwargs) -> Dict[str, float]:
        """Performs a network update/gradient step. Returns loss and training stats."""
        pass

    @abstractmethod
    def save(self, filepath: str):
        """Saves agent network checkpoints to disk."""
        pass

    @abstractmethod
    def load(self, filepath: str):
        """Loads agent network checkpoints from disk."""
        pass
