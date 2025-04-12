#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Critic network for the AID-RL project.
Defines the value estimator that outputs state values.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class Critic(nn.Module):
    """
    Critic network for the AID-RL project.
    
    This network takes a state representation as input and outputs 
    the estimated value of that state.
    """
    
    def __init__(self, state_dim, hidden_sizes=[128, 64]):
        """
        Initialize the critic network.
        
        Args:
            state_dim (int): Dimension of the state vector
            hidden_sizes (list): List of hidden layer sizes
        """
        super(Critic, self).__init__()
        
        # Create layers
        layers = []
        
        # Input layer
        layers.append(nn.Linear(state_dim, hidden_sizes[0]))
        layers.append(nn.ReLU())
        
        # Hidden layers
        for i in range(len(hidden_sizes) - 1):
            layers.append(nn.Linear(hidden_sizes[i], hidden_sizes[i + 1]))
            layers.append(nn.ReLU())
        
        # Output layer (produces a single value)
        layers.append(nn.Linear(hidden_sizes[-1], 1))
        
        # Combine all layers
        self.network = nn.Sequential(*layers)
    
    def forward(self, state):
        """
        Forward pass through the critic network.
        
        Args:
            state (torch.Tensor): Current state representation
            
        Returns:
            state_value (torch.Tensor): Estimated value of the state
        """
        state_value = self.network(state)
        return state_value
    
    def get_value(self, state):
        """
        Get the estimated value of a state.
        
        Args:
            state (torch.Tensor): Current state representation
            
        Returns:
            value (float): Estimated value of the state
        """
        with torch.no_grad():
            state_value = self.forward(state)
        
        return state_value.item()


if __name__ == "__main__":
    # Test the Critic network
    state_dim = 10
    batch_size = 2
    
    # Create random state tensor
    state = torch.rand(batch_size, state_dim)
    
    # Initialize the critic
    critic = Critic(state_dim)
    
    # Forward pass
    state_values = critic(state)
    
    print(f"State shape: {state.shape}")
    print(f"State values shape: {state_values.shape}")
    print(f"Values: {state_values}")
    
    # Test value estimation
    for i in range(batch_size):
        value = critic.get_value(state[i].unsqueeze(0))
        print(f"Estimated value: {value:.4f}")
