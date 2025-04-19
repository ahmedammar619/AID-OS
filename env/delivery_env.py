#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Custom RL environment for volunteer-recipient assignment in the AID-RL project.
Implements a Gym-compatible environment for the reinforcement learning agent.
"""

import numpy as np
import gym
from gym import spaces
import pandas as pd
import sys
import os

# Add parent directory to path to import from data
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from data.db_config import DatabaseHandler
from clustering.dbscan_cluster import RecipientClusterer


class DeliveryEnv(gym.Env):
    """
    Custom Gym environment for volunteer-recipient assignment optimization.
    
    This environment represents the monthly assignment process where the agent
    decides which volunteer to assign to which recipient.
    """
    
    metadata = {'render.modes': ['human']}
    
    def __init__(self, db_handler=None, use_clustering=True, max_steps=1000, cluster_eps=0.00005):
        """
        Initialize the delivery environment.
        
        Args:
            db_handler (DatabaseHandler): Database connection handler
            clusterer (RecipientClusterer): Clustering object for recipients
            use_clustering (bool): Whether to use clustering for state representation
            max_steps (int): Maximum number of steps per episode
        """
        super(DeliveryEnv, self).__init__()
        
        # Initialize database handler
        self.db_handler = db_handler if db_handler is not None else DatabaseHandler()
        
        # Initialize clusterer
        if use_clustering:
            self.clusterer = RecipientClusterer(
                min_cluster_size=2,
                cluster_selection_epsilon=cluster_eps,
                min_samples=1
            )
        else:
            self.clusterer = None
        self.use_clustering = use_clustering
        
        # Load initial data
        self.load_data()
        
        # Define action and observation spaces
        self.action_space = spaces.Discrete(self.num_volunteers * self.num_recipients)
        
        # Define feature dimensions
        self.num_features = 10  # Adjust based on feature engineering
        if self.use_clustering:
            self.num_features += 5  # Additional features for clustering
            
        # Observation space: state vector
        self.observation_space = spaces.Box(
            low=-1.0, 
            high=1.0, 
            shape=(self.num_features,),
            dtype=np.float32
        )
        
        # Set maximum steps
        self.max_steps = max_steps
        
        # Reset environment
        self.reset()
    
    def load_data(self):
        """Load volunteer and recipient data from the database."""
        # Get volunteers
        self.volunteers = self.db_handler.get_all_volunteers()
        self.num_volunteers = len(self.volunteers)
        
        # Get recipients
        self.recipients = self.db_handler.get_all_recipients()
        self.num_recipients = len(self.recipients)
        
        # Get pickups
        self.pickups = self.db_handler.get_all_pickups()
        self.num_pickups = len(self.pickups)

        # Get historical delivery data
        self.historical_data = self.db_handler.get_historical_deliveries()

        # Extract coordinates for clustering
        self.volunteer_coords = np.array([[v.latitude, v.longitude] 
                                         for v in self.volunteers])
        self.recipient_coords = np.array([[r.latitude, r.longitude] 
                                         for r in self.recipients])
        
        # Create distance matrix
        self.distance_matrix = self._create_distance_matrix()
        
        # Perform clustering if enabled
        if self.use_clustering:
            self.clusterer.fit(self.recipient_coords)
            self.clusters = self.clusterer.get_clusters()
    
    def _create_distance_matrix(self):
        """
        Create a distance matrix between all volunteers and recipients.
        
        Returns:
            distance_matrix (numpy.ndarray): Matrix of distances
                                            [volunteer_idx][recipient_idx]
        """
        distance_matrix = np.zeros((self.num_volunteers, self.num_recipients))
        
        for v_idx in range(self.num_volunteers):
            for r_idx in range(self.num_recipients):
                vol_lat = self.volunteers[v_idx].latitude
                vol_lon = self.volunteers[v_idx].longitude
                rec_lat = self.recipients[r_idx].latitude
                rec_lon = self.recipients[r_idx].longitude
                
                # Calculate Haversine distance
                distance = self._haversine_distance(vol_lat, vol_lon, rec_lat, rec_lon)
                distance_matrix[v_idx, r_idx] = distance
        
        return distance_matrix
    
    def _haversine_distance(self, lat1, lon1, lat2, lon2):
        """
        Calculate the Haversine distance between two points on Earth.
        
        Args:
            lat1, lon1: Coordinates of the first point (degrees)
            lat2, lon2: Coordinates of the second point (degrees)
            
        Returns:
            distance (float): Distance between the points in kilometers
        """
        # Convert decimal degrees to radians
        lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
        
        # Haversine formula
        dlat = lat2 - lat1
        dlon = lon2 - lon1
        a = np.sin(dlat/2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2)**2
        c = 2 * np.arcsin(np.sqrt(a))
        r = 6371  # Radius of Earth in kilometers
        
        return c * r
    
    def _decode_action(self, action):
        """
        Decode action index into volunteer and recipient indices.
        
        Args:
            action (int): Action index
            
        Returns:
            volunteer_idx (int): Index of the volunteer
            recipient_idx (int): Index of the recipient
        """
        volunteer_idx = action // self.num_recipients
        recipient_idx = action % self.num_recipients
        
        return volunteer_idx, recipient_idx
    
    def _encode_action(self, volunteer_idx, recipient_idx):
        """
        Encode volunteer and recipient indices into an action index.
        
        Args:
            volunteer_idx (int): Index of the volunteer
            recipient_idx (int): Index of the recipient
            
        Returns:
            action (int): Action index
        """
        return volunteer_idx * self.num_recipients + recipient_idx
    
    def _get_historical_match_score(self, volunteer_idx, recipient_idx):
        """
        Calculate a historical match score based on previous assignments.
        
        Args:
            volunteer_idx (int): Index of the volunteer
            recipient_idx (int): Index of the recipient
            
        Returns:
            score (float): Historical match score (0-3)
        """
        volunteer_id = self.volunteers[volunteer_idx].volunteer_id
        recipient_id = self.recipients[recipient_idx].recipient_id
        
        # Query the database for historical matches
        return self.db_handler.get_volunteer_historical_score(volunteer_id, recipient_id)
    
    def _check_assignment_validity(self, volunteer_idx, recipient_idx):
        """
        Check if an assignment is valid.
        
        Args:
            volunteer_idx (int): Index of the volunteer
            recipient_idx (int): Index of the recipient
            
        Returns:
            valid (bool): Whether the assignment is valid
        """
        # Check if recipient is already assigned
        if recipient_idx in self.assigned_recipients:
            return False
        
        # Check if volunteer has remaining capacity
        volunteer = self.volunteers[volunteer_idx]
        recipient = self.recipients[recipient_idx]
        
        # Calculate current load
        current_load = sum(self.recipients[r_idx].num_items 
                           for r_idx in self.volunteer_assignments.get(volunteer_idx, []))
        
        # Check if adding this recipient would exceed capacity
        if current_load + recipient.num_items > volunteer.car_size+1:
            return False
        
        return True
    
    def _compute_state(self):
        """
        Compute the current state representation.
        
        Returns:
            state (numpy.ndarray): The state vector (15 features)
        """
        features = []

        # 1. Percentage of assigned recipients
        assigned_percentage = len(self.assigned_recipients) / self.num_recipients
        features.append(assigned_percentage)

        # 2. Average distance in current assignments
        if len(self.assignment_list) > 0:
            distances = [self.distance_matrix[v_idx, r_idx] 
                        for v_idx, r_idx in self.assignment_list]
            avg_distance = np.mean(distances)
            features.append(avg_distance / 50.0)  # Normalize by 50 km
        else:
            features.append(0.0)

        # 3. Average utilization of volunteer capacity
        utilization = []
        for v_idx in range(self.num_volunteers):
            volunteer = self.volunteers[v_idx]
            assigned = self.volunteer_assignments.get(v_idx, [])
            if not assigned:
                utilization.append(0.0)
            else:
                current_load = sum(self.recipients[r_idx].num_items for r_idx in assigned)
                util = current_load / volunteer.car_size
                utilization.append(util)
        avg_utilization = np.mean(utilization) if utilization else 0.0
        features.append(avg_utilization)

        # 4. Variance in utilization
        util_variance = np.var(utilization) if len(utilization) > 1 else 0.0
        features.append(util_variance)

        # 5. Percentage of volunteers used
        used_volunteers = len([v for v in utilization if v > 0])
        volunteers_used_percentage = used_volunteers / self.num_volunteers
        features.append(volunteers_used_percentage)

        # 6. Average historical match score for current assignments
        if len(self.assignment_list) > 0:
            hist_scores = [self._get_historical_match_score(v_idx, r_idx) 
                        for v_idx, r_idx in self.assignment_list]
            avg_hist_score = np.mean(hist_scores)
            features.append(avg_hist_score / 3.0)  # Normalize by max score
        else:
            features.append(0.0)

        # 7. Remaining episode progress
        episode_progress = self.current_step / self.max_steps
        features.append(episode_progress)

        # 8. Average distance from volunteers to unassigned recipients
        unassigned = [r_idx for r_idx in range(self.num_recipients) if r_idx not in self.assigned_recipients]
        if unassigned:
            distances = [self.distance_matrix[v_idx, r_idx] for v_idx in range(self.num_volunteers) for r_idx in unassigned]
            avg_unassigned_distance = np.mean(distances) if distances else 0.0
            features.append(avg_unassigned_distance / 50.0)  # Normalize
        else:
            features.append(0.0)

        # 9. Average remaining box need per volunteer
        total_car_capacity = sum(v.car_size for v in self.volunteers)
        remaining_boxes = sum(self.recipients[r_idx].num_items for r_idx in unassigned)
        box_need_ratio = min(1.0, remaining_boxes / total_car_capacity if total_car_capacity > 0 else 0.0)
        features.append(box_need_ratio)

        # 10. Percentage of unassigned recipients near volunteers (< 5 km)
        near_count = 0
        for r_idx in unassigned:
            if any(self.distance_matrix[v_idx, r_idx] < 5.0 for v_idx in range(self.num_volunteers)):
                near_count += 1
        near_percentage = near_count / len(unassigned) if unassigned else 0.0
        features.append(near_percentage)

        # Clustering features
        if self.use_clustering:
            # 11. Number of clusters
            num_clusters = len(set(self.clusters['labels'])) - (1 if -1 in self.clusters['labels'] else 0)
            features.append(num_clusters / 25.0)  # Normalize

            # 12. Average cluster size
            cluster_sizes = [count for label, count in self.clusters['counts'].items() if label != -1]
            avg_cluster_size = np.mean(cluster_sizes) if cluster_sizes else 0.0
            features.append(avg_cluster_size / self.num_recipients)

            # 13. Percentage of recipients in clusters (vs. noise)
            if -1 in self.clusters['counts']:
                noise_count = self.clusters['counts'][-1]
                clustered_percentage = (self.num_recipients - noise_count) / self.num_recipients
            else:
                clustered_percentage = 1.0
            features.append(clustered_percentage)

            # 14. Average distance from volunteers to cluster centers
            if self.clusters['centers']:
                center_distances = []
                for v_idx in range(self.num_volunteers):
                    vol_lat = self.volunteers[v_idx].latitude
                    vol_lon = self.volunteers[v_idx].longitude
                    for center in self.clusters['centers'].values():
                        dist = self._haversine_distance(vol_lat, vol_lon, center[0], center[1])
                        center_distances.append(dist)
                avg_center_distance = np.mean(center_distances) if center_distances else 0.0
                features.append(avg_center_distance / 50.0)  # Normalize
            else:
                features.append(0.0)

            # 15. Variance in cluster sizes
            if cluster_sizes:
                cluster_variance = np.var(cluster_sizes)
                max_variance = (self.num_recipients ** 2) / 4  # Approx max for normalization
                features.append(cluster_variance / max_variance if max_variance > 0 else 0.0)
            else:
                features.append(0.0)

        # Convert to numpy array
        state = np.array(features, dtype=np.float32)
        return state
    
    def _compute_reward(self, volunteer_idx, recipient_idx):
        """
        Compute the reward for an assignment.
        
        Args:
            volunteer_idx (int): Index of the volunteer
            recipient_idx (int): Index of the recipient
            
        Returns:
            reward (float): Reward for the assignment
        """
        reward = 0.0
        
        # 1. Historical match reward (0-3)
        historical_score = self._get_historical_match_score(volunteer_idx, recipient_idx)
        reward += historical_score
        
        # 2. Proximity reward (0-2)
        distance = self.distance_matrix[volunteer_idx, recipient_idx]
        proximity_reward = max(0, 2 - (distance / 10))  # Decreases with distance
        reward += proximity_reward
        
        # 3. Capacity compatibility reward
        volunteer = self.volunteers[volunteer_idx]
        recipient = self.recipients[recipient_idx]
        
        current_load = sum(self.recipients[r_idx].num_items 
                          for r_idx in self.volunteer_assignments.get(volunteer_idx, []))
        total_load = current_load + recipient.num_items
        
        # Perfect match: between 80% and 100% of capacity
        capacity_ratio = total_load / volunteer.car_size
        if 0.8 <= capacity_ratio <= 1.0:
            reward += 2.0
        # Good match: between 60% and 80% of capacity
        elif 0.6 <= capacity_ratio < 0.8:
            reward += 1.0
        # Over capacity: between 100% and 110% of capacity
        elif 1.0 <= capacity_ratio < 1.1:
            reward += 0.0
        # Overcapacity: penalize
        elif capacity_ratio > 1.0:
            reward -= 3.0
        # Undercapacity: small penalty for wasted space
        elif capacity_ratio < 0.5:
            reward -= 1.0
        
        # 4. Clustering reward/penalty
        if self.use_clustering:
            # Get cluster label of the recipient
            recipient_cluster = self.clusters['labels'][recipient_idx]
            
            # Check if other recipients from the same cluster are assigned to the same volunteer
            if recipient_cluster != -1:  # Not noise
                # Find other recipients in the same cluster
                same_cluster_recipients = [
                    i for i, label in enumerate(self.clusters['labels']) 
                    if label == recipient_cluster
                ]
                
                # Check if any are already assigned to this volunteer
                assigned_to_volunteer = self.volunteer_assignments.get(volunteer_idx, [])
                
                cluster_match = any(r_idx in assigned_to_volunteer for r_idx in same_cluster_recipients)
                
                
                if cluster_match:
                    reward += 1.0  # Bonus for keeping cluster together
                else:
                    # Check if cluster is split across volunteers
                    for other_v_idx, other_assigned in self.volunteer_assignments.items():
                        if other_v_idx != volunteer_idx:
                            # Check ratio of taken capacity is already assigned to this volunteer and other
                            volunteer_util = sum(self.recipients[r_idx].num_items for r_idx in assigned_to_volunteer) / self.volunteers[volunteer_idx].car_size
                            other_util = sum(self.recipients[r_idx].num_items for r_idx in other_assigned) / self.volunteers[other_v_idx].car_size
                            
                            # Check if any recipients from the same cluster are assigned to this volunteer and the car size ratio of both is less than 0.8
                            if any(r_idx in other_assigned for r_idx in same_cluster_recipients) and (volunteer_util < 0.8 and other_util < 0.8):
                                reward -= 2.0  # Penalty for splitting cluster
                                break
                            
        
        return reward
    
    def reset(self):
        """
        Reset the environment to initial state.
        
        Returns:
            state (numpy.ndarray): Initial state
        """
        # Reset step counter
        self.current_step = 0
        
        # Reset assignments
        self.assignment_list = []
        self.assigned_recipients = set()
        self.volunteer_assignments = {}  # Maps volunteer_idx -> list of recipient_idx
        
        # Compute initial state
        self.state = self._compute_state()
        
        return self.state
    
    def step(self, action):
        """
        Take a step in the environment.
        
        Args:
            action (int): Action to take
            
        Returns:
            next_state (numpy.ndarray): Next state
            reward (float): Reward for the action
            done (bool): Whether the episode is done
            info (dict): Additional information
        """
        # Decode action
        volunteer_idx, recipient_idx = self._decode_action(action)
        
        # Check if action is valid
        valid_action = self._check_assignment_validity(volunteer_idx, recipient_idx)
        
        # Initialize reward
        reward = 0.0
        

        # Update assignments
        self.assignment_list.append((volunteer_idx, recipient_idx))
        self.assigned_recipients.add(recipient_idx)
        
        if volunteer_idx not in self.volunteer_assignments:
            self.volunteer_assignments[volunteer_idx] = []
        self.volunteer_assignments[volunteer_idx].append(recipient_idx)
        
        # Compute reward for this assignment
        reward = self._compute_reward(volunteer_idx, recipient_idx)

        
        # Update state
        self.state = self._compute_state()
        
        # Increment step counter
        self.current_step += 1
        
        # Check if episode is done
        done = len(self.assigned_recipients) == self.num_recipients or self.current_step >= self.max_steps
        
        # Additional info
        info = {
            'valid_action': valid_action,
            'volunteer_idx': volunteer_idx,
            'recipient_idx': recipient_idx,
            'assigned_count': len(self.assigned_recipients),
            'total_recipients': self.num_recipients
        }
        
        return self.state, reward, done, info
    
    def render(self, mode='human'):
        """
        Render the environment.
        
        Args:
            mode (str): Rendering mode
        """
        print(f"\nStep: {self.current_step}")
        print(f"Assigned recipients: {len(self.assigned_recipients)}/{self.num_recipients}")
        
        # Print volunteer assignments
        print("\nVolunteer Assignments:")
        for v_idx, r_indices in self.volunteer_assignments.items():
            volunteer = self.volunteers[v_idx]
            capacity = volunteer.car_size
            
            # Calculate current load
            current_load = sum(self.recipients[r_idx].num_items for r_idx in r_indices)
            
            print(f"  Volunteer {volunteer.volunteer_id} (capacity {capacity}):")
            for r_idx in r_indices:
                recipient = self.recipients[r_idx]
                distance = self.distance_matrix[v_idx, r_idx]
                print(f"    Recipient {recipient.recipient_id}: {recipient.num_items} items ({distance:.2f} km)")
            
            print(f"    Total load: {current_load}/{capacity} ({current_load/capacity*100:.1f}%)")
        
        print("\nState:", self.state)
    
    def save_assignments(self):
        """
        Save the current assignments to the database.
        
        Returns:
            bool: Whether the save was successful
        """
        try:
            # Convert the assignments to database format
            db_assignments = []
            for volunteer_idx, recipient_idx in self.assignment_list:
                volunteer_id = self.volunteers[volunteer_idx].volunteer_id
                recipient_id = self.recipients[recipient_idx].recipient_id
                db_assignments.append((volunteer_id, recipient_id))
            
            # Save to database
            self.db_handler.bulk_save_assignments(db_assignments)
            return True
        except Exception as e:
            print(f"Error saving assignments: {e}")
            return False


if __name__ == "__main__":
    # Test the environment
    env = DeliveryEnv(max_steps=100)
    
    print(f"Number of volunteers: {env.num_volunteers}")
    print(f"Number of recipients: {env.num_recipients}")
    
    # Reset environment
    state = env.reset()
    
    # Run a few random steps
    for _ in range(10):
        # Sample a random action
        action = env.action_space.sample()
        
        # Take a step
        next_state, reward, done, info = env.step(action)
        
        # Render environment
        env.render()
        
        print(f"Reward: {reward}")
        
        if done:
            print("Episode done!")
            break
    
    # Save final assignments
    success = env.save_assignments()
    print(f"Assignments saved: {success}")
