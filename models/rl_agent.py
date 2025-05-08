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
        clip_epsilon=0.2,  # PPO clipping parameter
        value_loss_coef=0.5,  # Value loss coefficient
        entropy_coef=0.01,  # Entropy coefficient
        gae_lambda=0.95,  # GAE lambda parameter
    ):
        """
        Initialize the Actor-Critic agent with PPO algorithm.
        
        Args:
            state_dim (int): Dimension of the state vector
            action_dim (int): Dimension of the action space
            actor_lr (float): Learning rate for the actor network
            critic_lr (float): Learning rate for the critic network
            gamma (float): Discount factor for future rewards
            device (str): Device to run the models on ('cpu' or 'cuda')
            buffer_size (int): Size of the replay buffer
            batch_size (int): Batch size for training
            clip_epsilon (float): PPO clipping parameter
            value_loss_coef (float): Value loss coefficient
            entropy_coef (float): Entropy coefficient for exploration
            gae_lambda (float): Lambda parameter for GAE
        """
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.gamma = gamma
        self.device = torch.device(device)
        self.batch_size = batch_size
        
        # PPO specific parameters
        self.clip_epsilon = clip_epsilon
        self.value_loss_coef = value_loss_coef
        self.entropy_coef = entropy_coef
        self.gae_lambda = gae_lambda
        
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
    
    def select_action(self, state, env=None, deterministic=False):
        """
        Select an action based on the current policy.
        
        Args:
            state (numpy.ndarray): Current state representation
            env (DeliveryEnv): Environment to check valid actions
            deterministic (bool): If True, select the most probable action,
                                 otherwise sample from the distribution
        
        Returns:
            action (int): The selected action index
            action_prob (float): Probability of the selected action
            log_prob (float): Log probability of the selected action (for PPO)
        """
        # Convert state to tensor
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        
        # Get action from the actor with action masking
        action, action_prob = self.actor.select_action(state_tensor, env, deterministic)
        # For PPO, we also need the log probability
        if action != -1:  # Only compute log_prob for valid actions
            action_tensor = torch.LongTensor([action]).to(self.device)
            log_prob = self.actor.get_log_prob(state_tensor, action_tensor).item()
        else:
            log_prob = 0.0  # Default for invalid actions
            
        return action, action_prob, log_prob
    
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
    
    def store_transition(self, state, action, reward, next_state, done, log_prob=None):
        """
        Store transition in the replay buffer.
        
        Args:
            state (numpy.ndarray): Current state representation
            action (int): Action taken
            reward (float): Reward received
            next_state (numpy.ndarray): Next state representation
            done (bool): Whether the episode is done
            log_prob (float): Log probability of the action (for PPO)
        """
        self.replay_buffer.append((state, action, reward, next_state, done, log_prob))
    
    def train(self, num_updates=1, ppo_epochs=4):
        """
        Train the actor and critic networks using PPO algorithm.
        
        Args:
            num_updates (int): Number of training updates to perform
            ppo_epochs (int): Number of PPO epochs per update
            
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
            
            # Unpack the batch - handle case where log_probs might be missing in older entries
            if len(minibatch[0]) >= 6:
                states, actions, rewards, next_states, dones, old_log_probs = zip(*minibatch)
                # Handle None values in old_log_probs
                old_log_probs = [lp if lp is not None else 0.0 for lp in old_log_probs]
            else:
                states, actions, rewards, next_states, dones = zip(*minibatch)
                old_log_probs = [0.0] * len(states)  # Default log probs if not available
            
            # Convert to numpy arrays first for efficiency
            import numpy as np
            states_np = np.array(states)
            actions_np = np.array(actions)
            rewards_np = np.array(rewards)
            next_states_np = np.array(next_states)
            dones_np = np.array(dones)
            old_log_probs_np = np.array(old_log_probs)
            
            # Convert to tensors
            state_batch = torch.FloatTensor(states_np).to(self.device)
            action_batch = torch.LongTensor(actions_np).to(self.device)
            reward_batch = torch.FloatTensor(rewards_np).to(self.device)
            next_state_batch = torch.FloatTensor(next_states_np).to(self.device)
            done_batch = torch.FloatTensor(dones_np).to(self.device)
            old_log_probs_batch = torch.FloatTensor(old_log_probs_np).to(self.device)
            
            # Compute values once outside the epoch loop
            with torch.no_grad():
                values = self.critic(state_batch).squeeze()
                next_values = self.critic(next_state_batch).squeeze()
                
                # Calculate returns (simpler version - TD targets)
                returns = reward_batch + self.gamma * next_values * (1 - done_batch)
                
                # Calculate advantages
                advantages = returns - values
                
                # Normalize advantages
                if len(advantages) > 1:
                    advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
            
            # PPO update for multiple epochs
            for _ in range(ppo_epochs):
                # Update critic
                self.critic_optimizer.zero_grad()
                current_values = self.critic(state_batch).squeeze()
                value_loss = nn.MSELoss()(current_values, returns.detach())
                value_loss.backward()
                self.critic_optimizer.step()
                
                # Update actor
                self.actor_optimizer.zero_grad()
                
                # Get current log probs and entropy
                dist = self.actor.forward(state_batch)
                current_log_probs = self.actor.get_log_prob(state_batch, action_batch)
                entropy = torch.distributions.Categorical(dist).entropy().mean()
                
                # Calculate ratios and surrogate objectives
                ratios = torch.exp(current_log_probs - old_log_probs_batch)
                surr1 = ratios * advantages.detach()
                surr2 = torch.clamp(ratios, 1.0 - self.clip_epsilon, 1.0 + self.clip_epsilon) * advantages.detach()
                
                # PPO actor loss (negative because we're minimizing)
                actor_loss = -torch.min(surr1, surr2).mean()
                
                # Add entropy bonus for exploration
                actor_loss = actor_loss - self.entropy_coef * entropy
                
                actor_loss.backward()
                self.actor_optimizer.step()
                
                # Store losses
                actor_losses.append(actor_loss.item())
                critic_losses.append(value_loss.item())
            
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
