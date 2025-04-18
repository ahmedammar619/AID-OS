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
        checkpoint_dir="./checkpoints",
        log_dir="./logs"
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
        
    def train(self, env, num_episodes=1000, max_steps=600, print_interval=10, checkpoint_interval=50):
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
        
        for episode in range(1, num_episodes + 1):
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
                # Select action
                action, _ = self.agent.select_action(state)
                
                # Take step in environment
                next_state, reward, done, info = env.step(action)
                
                # Store transition
                states.append(state)
                actions.append(action)
                rewards.append(reward)
                next_states.append(next_state)
                dones.append(done)
                
                # Update agent with new transition
                self.agent.store_transition(state, action, reward, next_state, done)
                
                # Update state and counters
                state = next_state
                episode_reward += reward
                episode_length += 1
                
                # Break if done
                if done:
                    break
            
            # Train agent after episode
            actor_loss, critic_loss = self.agent.train(num_updates=min(episode_length, 5))
            
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
                      f"Expected Reward: {critic_loss:.4f} | "
                      f"Length: {episode_length} | "
                      f"Assignments: {len(env.assigned_recipients)}/{env.num_recipients} | "
                      f"Time: {elapsed:.2f}s")
            
            # Save checkpoint
            if episode % checkpoint_interval == 0:
                self.save_checkpoint(episode)
                self.plot_training_progress()
        
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
        fig, axes = plt.subplots(2, 1, figsize=(10, 12), sharex=True)
        
        # Plot rewards
        axes[0].plot(self.episode_rewards, label='Episode Reward', alpha=0.6)
        axes[0].plot(self.avg_rewards, label='Avg Reward (100 ep)', linewidth=2)
        axes[0].set_ylabel('Reward')
        axes[0].set_title('Training Progress')
        axes[0].legend()
        axes[0].grid(True)
        
        # Plot episode lengths
        axes[1].plot(self.episode_lengths)
        axes[1].set_xlabel('Episode')
        axes[1].set_ylabel('Length')
        axes[1].set_title('Episode Lengths')
        axes[1].grid(True)
        
        # Add timestamp
        plt.figtext(0.5, 0.01, f'Generated: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}',
                   ha='center', fontsize=10)
        
        # Adjust layout and save
        plt.tight_layout(rect=[0, 0.03, 1, 0.98])
        plt.savefig(os.path.join(self.log_dir, f"training_progress_{self.current_episode}.png"))
        plt.close()


if __name__ == "__main__":
    # Create database handler
    db_handler = DatabaseHandler()
    
    max_steps = 600
    # Create environment
    env = DeliveryEnv(db_handler=db_handler, max_steps=max_steps)
    
    # Create trainer
    trainer = AgentTrainer(
        state_dim=env.observation_space.shape[0],
        action_dim=env.action_space.n,
        db_handler=db_handler,
        device="cuda" if torch.cuda.is_available() else "cpu"
    )
    
    # Training loop
    stats = trainer.train(
        env=env,
        num_episodes=15,
        max_steps=max_steps,
        print_interval=10,
        checkpoint_interval=50
    )
    
    print("Training complete!")
