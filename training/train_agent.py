#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Training module for the RL agent in the AID-RL project.
Implements the training loop for the Actor-Critic agent.
"""

import numpy as np
import torch
import matplotlib.pyplot as plt
import pandas as pd
import os
import sys
import time
from datetime import datetime

# Add parent directory to path to import from project modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from models.rl_agent import ActorCriticAgent
from env.delivery_env import DeliveryEnv
from data.db_config import DatabaseHandler


class AgentTrainer:
    """
    Class for training the RL agent on the delivery environment.
    """
    
    def __init__(
        self,
        state_dim,
        action_dim,
        db_handler=None,
        actor_lr=0.001,
        critic_lr=0.002,
        gamma=0.99,
        device="cpu",
        checkpoint_dir="./hist/checkpoints",
        log_dir="./hist/logs"
    ):
        """
        Initialize the trainer.
        
        Args:
            state_dim (int): Dimension of the state vector
            action_dim (int): Dimension of the action space
            db_handler (DatabaseHandler): Database connection handler
            actor_lr (float): Learning rate for the actor network
            critic_lr (float): Learning rate for the critic network
            gamma (float): Discount factor for future rewards
            device (str): Device to run the models on ('cpu' or 'cuda')
            checkpoint_dir (str): Directory to save model checkpoints
            log_dir (str): Directory to save training logs
        """
        # Initialize paths
        self.checkpoint_dir = checkpoint_dir
        self.log_dir = log_dir
        
        # Create directories if they don't exist
        os.makedirs(checkpoint_dir, exist_ok=True)
        os.makedirs(log_dir, exist_ok=True)
        
        # Initialize agent
        self.agent = ActorCriticAgent(
            state_dim=state_dim,
            action_dim=action_dim,
            actor_lr=actor_lr,
            critic_lr=critic_lr,
            gamma=gamma,
            device=device
        )
        
        # Initialize database handler
        self.db_handler = db_handler if db_handler is not None else DatabaseHandler()
        
        # Tracking variables
        self.episode_rewards = []
        self.episode_lengths = []
        self.avg_rewards = []
        self.current_episode = 0
        
    def train(self, env, num_episodes=1000, max_steps=600, print_interval=10, checkpoint_interval=50, agent_num_updates=10):
        """
        Train the agent on the environment.
        
        Args:
            env (DeliveryEnv): The environment to train on
            num_episodes (int): Number of episodes to train for
            max_steps (int): Maximum steps per episode
            print_interval (int): Interval for printing progress
            checkpoint_interval (int): Interval for saving model checkpoints
            
        Returns:
            pd.DataFrame: Training statistics
        """
        # Training statistics
        stats = {
            'episode': [],
            'reward': [],
            'length': [],
            'actor_loss': [],
            'critic_loss': [],
            'assignments': []
        }
        
        # Start training loop
        start_time = time.time()
        
        try:
            for episode in range(1, num_episodes + 1):
                print(f"Episode {episode}/{num_episodes}")
                self.current_episode = episode
                
                # Reset environment
                state = env.reset()
                
                # Episode variables
                episode_reward = 0
                episode_length = 0
                
                # Lists to store transitions
                states = []
                actions = []
                rewards = []
                next_states = []
                dones = []
                
                # Episode loop
                for step in range(max_steps):
                    print(f"Step {step}/{max_steps}")
                    # Select action (now returns action, prob, log_prob)
                    action, prob, log_prob = self.agent.select_action(state, env)
                    print(f"Action taken")
                    # Take step in environment
                    next_state, reward, done, info = env.step(action)
                    print(f"Next state")
                    # Store transition
                    states.append(state)
                    actions.append(action)
                    rewards.append(reward)
                    next_states.append(next_state)
                    dones.append(done)
                    print(f"Transition stored")
                    # Update agent with new transition (now includes log_prob for PPO)
                    self.agent.store_transition(state, action, reward, next_state, done, log_prob)
                    # Update state and counters
                    state = next_state
                    episode_reward += reward
                    episode_length += 1
                    
                    # Break if done
                    if done:
                        break
                
                # Train agent after episode
                actor_loss, critic_loss = self.agent.train(num_updates=min(episode_length, agent_num_updates))
                
                # Store episode statistics
                self.episode_rewards.append(episode_reward)
                self.episode_lengths.append(episode_length)
                
                # Calculate moving average reward
                avg_reward = np.mean(self.episode_rewards[-100:])
                self.avg_rewards.append(avg_reward)
                
                # Update stats dictionary
                stats['episode'].append(episode)
                stats['reward'].append(episode_reward)
                stats['length'].append(episode_length)
                stats['actor_loss'].append(actor_loss)
                stats['critic_loss'].append(critic_loss)
                stats['assignments'].append(len(env.assigned_recipients))
                
                # Print progress
                if episode % print_interval == 0:
                    elapsed = time.time() - start_time
                    print(f"Episode {episode}/{num_episodes} | "
                        f"Reward: {episode_reward:.2f} | "
                        f"Avg Reward: {avg_reward:.2f} | "
                        f"Length: {episode_length} | "
                        f"Assignments: {len(env.assigned_recipients)}/{env.num_recipients} | "
                        f"Time: {elapsed:.2f}s")
                
                # Save checkpoint
                if episode % checkpoint_interval == 0:
                    self.save_checkpoint(episode)
                    self.plot_training_progress()

        except KeyboardInterrupt:
            print("\nTraining interrupted by user (Ctrl+C). Saving checkpoint and exiting...")
            self.save_checkpoint("interrupted")
            self.plot_training_progress()
            return stats
        
        # Save final model
        self.save_checkpoint("final")
        
        # Create and save training statistics
        df = pd.DataFrame(stats)
        df.to_csv(os.path.join(self.log_dir, "training_stats.csv"), index=False)
        
        # Plot final training progress
        self.plot_training_progress()
        
        return df
    
    def save_checkpoint(self, episode):
        """
        Save a checkpoint of the agent's models.
        
        Args:
            episode (int or str): Episode number or identifier
        """
        checkpoint_path = os.path.join(self.checkpoint_dir, f"checkpoint_{episode}")
        self.agent.save_models(checkpoint_path)
    
    def load_checkpoint(self, episode):
        """
        Load a checkpoint of the agent's models.
        
        Args:
            episode (int or str): Episode number or identifier
        """
        checkpoint_path = os.path.join(self.checkpoint_dir, f"checkpoint_{episode}")
        self.agent.load_models(checkpoint_path)
    
    def plot_training_progress(self):
        """Plot and save training progress graphs."""
        # Create figure
        fig, axes = plt.subplots(1, 1, figsize=(10, 8), sharex=True)
        
        # Plot rewards
        axes.plot(self.episode_rewards, label='Episode Reward', alpha=0.6)
        axes.plot(self.avg_rewards, label='Avg Reward (100 ep)', linewidth=2)
        axes.set_ylabel('Reward')
        axes.set_title('Training Progress')
        axes.legend()
        axes.grid(True)
        
        # Add timestamp
        plt.figtext(0.5, 0.01, f'Generated: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}',
                   ha='center', fontsize=10)
        
        # Adjust layout and save
        plt.tight_layout(rect=[0, 0.03, 1, 0.98])
        plt.savefig(os.path.join(self.log_dir, f"{self.current_episode}_{datetime.now().strftime('%H:%M')}.png"))
        plt.close()

