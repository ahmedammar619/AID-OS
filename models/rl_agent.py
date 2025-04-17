#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Actor-Critic agent for the AID-RL project.
Combines the actor and critic networks for reinforcement learning.
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from collections import deque
import random
import os

from models.actor import Actor
from models.critic import Critic


class ActorCriticAgent:
    """
    Actor-Critic agent for volunteer-recipient assignment optimization.
    
    This agent combines the actor and critic networks for reinforcement learning,
    using the policy gradient method with the advantage function.
    """
    
    def __init__(
        self,
        state_dim,
        action_dim,
        actor_lr=0.001,
        critic_lr=0.002,
        gamma=0.99,
        device="cpu",
        buffer_size=10000,
        batch_size=64,
    ):
        """
        Initialize the Actor-Critic agent.
        
        Args:
            state_dim (int): Dimension of the state vector
            action_dim (int): Dimension of the action space
            actor_lr (float): Learning rate for the actor network
            critic_lr (float): Learning rate for the critic network
            gamma (float): Discount factor for future rewards
            device (str): Device to run the models on ('cpu' or 'cuda')
            buffer_size (int): Size of the replay buffer
            batch_size (int): Batch size for training
        """
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.gamma = gamma
        self.device = torch.device(device)
        self.batch_size = batch_size
        
        # Initialize actor and critic networks
        self.actor = Actor(state_dim, action_dim).to(self.device)
        self.critic = Critic(state_dim).to(self.device)
        
        # Set up optimizers
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=actor_lr)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=critic_lr)
        
        # Initialize replay buffer for experience replay
        self.replay_buffer = deque(maxlen=buffer_size)
        
        # Tracking variables
        self.episode_rewards = []
        self.episode_lengths = []
        self.training_steps = 0
    
    def select_action(self, state, deterministic=False):
        """
        Select an action based on the current policy.
        
        Args:
            state (numpy.ndarray): Current state representation
            deterministic (bool): If True, select the most probable action,
                                 otherwise sample from the distribution
        
        Returns:
            action (int): The selected action index
            action_prob (float): Probability of the selected action
        """
        # Convert state to tensor
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        
        # Get action from the actor
        return self.actor.select_action(state_tensor, deterministic)
    
    def get_value(self, state):
        """
        Get the estimated value of a state.
        
        Args:
            state (numpy.ndarray): Current state representation
            
        Returns:
            value (float): Estimated value of the state
        """
        # Convert state to tensor
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        
        # Get value from the critic
        return self.critic.get_value(state_tensor)
    
    def store_transition(self, state, action, reward, next_state, done):
        """
        Store transition in the replay buffer.
        
        Args:
            state (numpy.ndarray): Current state representation
            action (int): Action taken
            reward (float): Reward received
            next_state (numpy.ndarray): Next state representation
            done (bool): Whether the episode is done
        """
        self.replay_buffer.append((state, action, reward, next_state, done))
    
    def train(self, num_updates=1):
        """
        Train the actor and critic networks.
        
        Args:
            num_updates (int): Number of training updates to perform
            
        Returns:
            actor_loss (float): Average actor loss
            critic_loss (float): Average critic loss
        """
        if len(self.replay_buffer) < self.batch_size:
            return 0.0, 0.0
        
        actor_losses = []
        critic_losses = []
        
        for _ in range(num_updates):
            # Sample a batch of transitions
            minibatch = random.sample(self.replay_buffer, self.batch_size)
            
            # Unpack the batch
            states, actions, rewards, next_states, dones = zip(*minibatch)
            
            # Convert to tensors
            state_batch = torch.FloatTensor(states).to(self.device)
            action_batch = torch.LongTensor(actions).to(self.device)
            reward_batch = torch.FloatTensor(rewards).to(self.device)
            next_state_batch = torch.FloatTensor(next_states).to(self.device)
            done_batch = torch.FloatTensor(dones).to(self.device)
            
            # Calculate state values
            values = self.critic(state_batch).squeeze()
            next_values = self.critic(next_state_batch).squeeze()
            
            # Calculate target values using TD(0)
            targets = reward_batch + self.gamma * next_values * (1 - done_batch)
            
            # Calculate advantage
            advantages = targets.detach() - values
            
            # Update critic
            critic_loss = nn.MSELoss()(values, targets.detach())
            
            self.critic_optimizer.zero_grad()
            critic_loss.backward()
            self.critic_optimizer.step()
            
            # Update actor using policy gradient
            log_probs = self.actor.get_log_prob(state_batch, action_batch)
            actor_loss = -(log_probs * advantages.detach()).mean()
            
            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            self.actor_optimizer.step()
            
            # Store losses
            actor_losses.append(actor_loss.item())
            critic_losses.append(critic_loss.item())
            
            self.training_steps += 1
        
        return np.mean(actor_losses), np.mean(critic_losses)
    
    def save_models(self, directory):
        """
        Save actor and critic models to disk.
        
        Args:
            directory (str): Directory to save the models
        """
        if not os.path.exists(directory):
            os.makedirs(directory)
        
        actor_path = os.path.join(directory, "actor.pth")
        critic_path = os.path.join(directory, "critic.pth")
        
        torch.save(self.actor.state_dict(), actor_path)
        torch.save(self.critic.state_dict(), critic_path)
        
        print(f"Models saved to {directory}")
    
    def load_models(self, directory):
        """
        Load actor and critic models from disk.
        
        Args:
            directory (str): Directory to load the models from
        """
        actor_path = os.path.join(directory, "actor.pth")
        critic_path = os.path.join(directory, "critic.pth")
        
        # Check if files exist
        if os.path.exists(actor_path) and os.path.exists(critic_path):
            # Load actor
            self.actor.load_state_dict(torch.load(actor_path, map_location=self.device))
            
            # Load critic
            self.critic.load_state_dict(torch.load(critic_path, map_location=self.device))
            
            print(f"Models loaded from {directory}")
        else:
            print(f"Models not found in {directory}")


if __name__ == "__main__":
    # Test the ActorCriticAgent
    state_dim = 10
    action_dim = 5
    
    # Initialize the agent
    agent = ActorCriticAgent(state_dim, action_dim)
    
    # Create a dummy state
    state = np.random.rand(state_dim)
    
    # Test action selection
    action, prob = agent.select_action(state)
    value = agent.get_value(state)
    
    print(f"Selected action: {action}, Probability: {prob:.4f}")
    print(f"Estimated state value: {value:.4f}")
    
    # Test storing transitions and training
    for _ in range(100):
        next_state = np.random.rand(state_dim)
        reward = np.random.rand()
        done = np.random.rand() > 0.9
        
        agent.store_transition(state, action, reward, next_state, done)
        state = next_state
        action, _ = agent.select_action(state)
    
    # Train the agent
    actor_loss, critic_loss = agent.train()
    
    print(f"Actor loss: {actor_loss:.4f}")
    print(f"Critic loss: {critic_loss:.4f}")
