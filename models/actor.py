#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Actor network for the AID-RL project.
Defines the policy network that outputs action probabilities.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class Actor(nn.Module):
    """
    Actor network for the AID-RL project.
    
    This network takes a state representation as input and outputs 
    a probability distribution over actions (volunteer-recipient pairs).
    """
    
    def __init__(self, state_dim, action_dim, hidden_sizes=[128, 64]):
        """
        Initialize the actor network.
        
        Args:
            state_dim (int): Dimension of the state vector
            action_dim (int): Dimension of the action space
            hidden_sizes (list): List of hidden layer sizes
        """
        super(Actor, self).__init__()
        
        # Create layers
        layers = []
        
        # Input layer
        layers.append(nn.Linear(state_dim, hidden_sizes[0]))
        layers.append(nn.ReLU())
        
        # Hidden layers
        for i in range(len(hidden_sizes) - 1):
            layers.append(nn.Linear(hidden_sizes[i], hidden_sizes[i + 1]))
            layers.append(nn.ReLU())
        
        # Output layer (produces logits)
        layers.append(nn.Linear(hidden_sizes[-1], action_dim))
        
        # Combine all layers
        self.network = nn.Sequential(*layers)
    
    def forward(self, state):
        """
        Forward pass through the actor network.
        
        Args:
            state (torch.Tensor): Current state representation
            
        Returns:
            action_probs (torch.Tensor): Probability distribution over actions
        """
        logits = self.network(state)
        action_probs = F.softmax(logits, dim=-1)
        return action_probs
    
    def select_action(self, state, deterministic=False):
        """
        Select an action based on the current policy.
        
        Args:
            state (torch.Tensor): Current state representation
            deterministic (bool): If True, select the most probable action,
                                 otherwise sample from the distribution
        
        Returns:
            action (int): The selected action index
            action_prob (float): Probability of the selected action
        """
        with torch.no_grad():
            action_probs = self.forward(state)
            
        if deterministic:
            # Choose the action with highest probability
            action = torch.argmax(action_probs).item()
        else:
            # Sample from the probability distribution
            action_distribution = torch.distributions.Categorical(action_probs)
            action = action_distribution.sample().item()
        
        # Return the action and its probability
        return action, action_probs[0, action].item()

    def get_log_prob(self, state, action):
        """
        Get the log probability of taking an action in a given state.
        
        Args:
            state (torch.Tensor): Current state representation
            action (torch.Tensor): Action taken
            
        Returns:
            log_prob (torch.Tensor): Log probability of taking the action
        """
        action_probs = self.forward(state)
        action_distribution = torch.distributions.Categorical(action_probs)
        
        return action_distribution.log_prob(action)


if __name__ == "__main__":
    # Test the Actor network
    state_dim = 10
    action_dim = 5
    batch_size = 2
    
    # Create random state tensor
    state = torch.rand(batch_size, state_dim)
    
    # Initialize the actor
    actor = Actor(state_dim, action_dim)
    
    # Forward pass
    action_probs = actor(state)
    
    print(f"State shape: {state.shape}")
    print(f"Action probabilities shape: {action_probs.shape}")
    print(f"Probabilities: {action_probs}")
    
    # Test action selection
    for i in range(batch_size):
        action, prob = actor.select_action(state[i].unsqueeze(0))
        print(f"Selected action: {action}, Probability: {prob:.4f}")
