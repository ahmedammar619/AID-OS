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
        # Convert latitude and longitude from degrees to radians
        lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
        
        # Haversine formula
        dlat = lat2 - lat1
        dlon = lon2 - lon1
        a = np.sin(dlat/2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2)**2
        c = 2 * np.arcsin(np.sqrt(a))
        r = 6371  # Radius of Earth in kilometers
        
        return c * r
        
    def _calculate_bearing(self, lat1, lon1, lat2, lon2):
        """
        Calculate the bearing (direction) from point 1 to point 2.
        
        Args:
            lat1, lon1: Coordinates of the first point (degrees)
            lat2, lon2: Coordinates of the second point (degrees)
            
        Returns:
            bearing (float): Bearing in degrees (0-360, where 0 is North)
        """
        # Convert latitude and longitude from degrees to radians
        lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
        
        # Calculate bearing
        dlon = lon2 - lon1
        y = np.sin(dlon) * np.cos(lat2)
        x = np.cos(lat1) * np.sin(lat2) - np.sin(lat1) * np.cos(lat2) * np.cos(dlon)
        bearing_rad = np.arctan2(y, x)
        
        # Convert to degrees and normalize to 0-360
        bearing_deg = np.degrees(bearing_rad)
        bearing_deg = (bearing_deg + 360) % 360
        
        return bearing_deg
        
    def _bearing_difference(self, bearing1, bearing2):
        """
        Calculate the absolute difference between two bearings, accounting for circularity.
        
        Args:
            bearing1, bearing2: Bearings in degrees (0-360)
            
        Returns:
            difference (float): Absolute difference in degrees (0-180)
        """
        diff = abs(bearing1 - bearing2) % 360
        if diff > 180:
            diff = 360 - diff
        return diff
    
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
        import math
        import numpy as np

        # Constants
        REWARD_MIN = -10.0
        REWARD_MAX = 10.0
        HIST_TAU_DAYS = 60  # How fast historical scores decay
        PROX_DISTANCE_SCALE = 3.0  # Controls proximity decay
        W = {
            'historical': 1.0,
            'efficiency': 1.0,
            'proximity': 1.0,
            'cluster_distance': 1.0,
            'cluster_direction': 1.0,
            'capacity': 1.0,
            'cluster_membership': 1.0,
            'time_efficiency': 2.5  # Increased weight for time efficiency
        }

        reward = 0.0

        # 1. Historical match reward (decayed by recency)
        # hist_score = self._get_historical_match_score(volunteer_idx, recipient_idx)
        # hist_days_ago = self._get_historical_match_days_ago(volunteer_idx, recipient_idx)
        # decayed_score = hist_score * math.exp(-hist_days_ago / HIST_TAU_DAYS)
        # reward += W['historical'] * decayed_score

        # 2. Volunteer efficiency
        efficiency_reward = self._compute_volunteer_efficiency(volunteer_idx, recipient_idx)
        reward += W['efficiency'] * efficiency_reward

        # 3. Proximity reward using exponential decay
        distance = self.distance_matrix[volunteer_idx, recipient_idx]
        proximity_reward = math.exp(-distance / PROX_DISTANCE_SCALE) * 2  # Scaled to max ~2
        reward += W['proximity'] * proximity_reward

        # 4. Recipient-to-recipient proximity (clustering)
        assigned_recipients = self.volunteer_assignments.get(volunteer_idx, [])
        if assigned_recipients:
            v_lat, v_lon = self.volunteers[volunteer_idx].latitude, self.volunteers[volunteer_idx].longitude
            r_lat, r_lon = self.recipients[recipient_idx].latitude, self.recipients[recipient_idx].longitude
            new_bearing = self._calculate_bearing(v_lat, v_lon, r_lat, r_lon)

            distances = []
            bearing_diffs = []

            for other_r_idx in assigned_recipients:
                r2_lat = self.recipients[other_r_idx].latitude
                r2_lon = self.recipients[other_r_idx].longitude
                dist = self._haversine_distance(r_lat, r_lon, r2_lat, r2_lon)
                distances.append(dist)

                other_bearing = self._calculate_bearing(v_lat, v_lon, r2_lat, r2_lon)
                bearing_diff = self._bearing_difference(new_bearing, other_bearing)
                bearing_diffs.append(bearing_diff)

            avg_dist = np.mean(distances) if distances else 0
            avg_bearing_diff = np.mean(bearing_diffs) if bearing_diffs else 0

            # Distance-based reward
            if avg_dist < 2:
                reward += W['cluster_distance'] * 5.0
            elif avg_dist > 5:
                reward += W['cluster_distance'] * (-3.0 * (avg_dist / 5.0))
            else:
                reward += W['cluster_distance'] * 2.0

            # Directional reward
            if avg_bearing_diff < 30:
                reward += W['cluster_direction'] * 4.0
            elif avg_bearing_diff < 45:
                reward += W['cluster_direction'] * 2.0
            elif avg_bearing_diff > 120:
                reward += W['cluster_direction'] * -5.0
            elif avg_bearing_diff > 90:
                reward += W['cluster_direction'] * -2.0

        # 5. Capacity reward
        volunteer = self.volunteers[volunteer_idx]
        recipient = self.recipients[recipient_idx]
        current_load = sum(self.recipients[r].num_items for r in assigned_recipients)
        total_load = current_load + recipient.num_items
        capacity_ratio = total_load / volunteer.car_size

        if 0.9 <= capacity_ratio <= 1.15:
            reward += W['capacity'] * 3.0
        elif 0.8 <= capacity_ratio < 0.9:
            reward += W['capacity'] * 1.0
        elif 0.7 <= capacity_ratio < 0.8:
            reward += W['capacity'] * 0.5
        elif capacity_ratio > 1.0:
            reward += W['capacity'] * -4.0

        # 6. Clustering penalty for splitting
        if self.use_clustering:
            cluster_label = self.clusters['labels'][recipient_idx]
            if cluster_label != -1:
                same_cluster = [
                    i for i, label in enumerate(self.clusters['labels']) if label == cluster_label
                ]
                assigned_here = self.volunteer_assignments.get(volunteer_idx, [])
                already_with_volunteer = any(r in assigned_here for r in same_cluster)

                if already_with_volunteer:
                    reward += W['cluster_membership'] * 1.0
                else:
                    for other_v_idx, other_assigned in self.volunteer_assignments.items():
                        if other_v_idx == volunteer_idx:
                            continue
                        if any(r in other_assigned for r in same_cluster):
                            volunteer_util = sum(self.recipients[r].num_items for r in assigned_here) / volunteer.car_size
                            other_util = sum(self.recipients[r].num_items for r in other_assigned) / self.volunteers[other_v_idx].car_size
                            if volunteer_util < 0.8 and other_util < 0.8:
                                reward += W['cluster_membership'] * -2.0
                                break

        # 7. Time-efficiency reward: calculate estimated travel time for the entire route
        if assigned_recipients:
            # Get volunteer coordinates
            v_lat, v_lon = volunteer.latitude, volunteer.longitude
            
            # Get all recipients including the new one
            all_recipients = assigned_recipients + [recipient_idx]
            
            # Calculate estimated route time using a greedy approach
            # Start at volunteer's home
            current_lat, current_lon = v_lat, v_lon
            total_distance = 0
            
            # Sort recipients by distance from current position (greedy approach)
            remaining_recipients = []
            for r_idx in all_recipients:
                r_lat = self.recipients[r_idx].latitude
                r_lon = self.recipients[r_idx].longitude
                dist = self._haversine_distance(current_lat, current_lon, r_lat, r_lon)
                remaining_recipients.append((r_idx, dist))
            
            # Visit each recipient in order of proximity
            while remaining_recipients:
                # Find closest recipient
                remaining_recipients.sort(key=lambda x: x[1])
                next_r_idx, next_dist = remaining_recipients.pop(0)
                
                # Add distance to this recipient
                total_distance += next_dist
                
                # Update current position
                current_lat = self.recipients[next_r_idx].latitude
                current_lon = self.recipients[next_r_idx].longitude
                
                # Update distances to remaining recipients
                for i in range(len(remaining_recipients)):
                    r_idx = remaining_recipients[i][0]
                    r_lat = self.recipients[r_idx].latitude
                    r_lon = self.recipients[r_idx].longitude
                    dist = self._haversine_distance(current_lat, current_lon, r_lat, r_lon)
                    remaining_recipients[i] = (r_idx, dist)
            
            # Add distance back to volunteer's home
            dist_back = self._haversine_distance(current_lat, current_lon, v_lat, v_lon)
            total_distance += dist_back
            
            # Calculate time (assume average speed of 30 km/h in urban areas)
            # Add 5 minutes per stop for delivery time
            avg_speed_kmh = 30
            stop_time_mins = 5 * len(all_recipients)
            travel_time_mins = (total_distance / avg_speed_kmh) * 60 + stop_time_mins
            
            # Convert to hours for scaling
            travel_time_hours = travel_time_mins / 60
            
            # Apply time efficiency reward (penalize long routes)
            # Scale: -1 point per hour of travel time
            time_efficiency_reward = -travel_time_hours
            reward += W['time_efficiency'] * time_efficiency_reward

        # Final: clip to avoid extreme values
        return np.clip(reward, REWARD_MIN, REWARD_MAX)

    def _compute_volunteer_efficiency(self, volunteer_idx, recipient_idx):
        recipient = self.recipients[recipient_idx]
        recipient_items = recipient.num_items
        chosen_distance = self.distance_matrix[volunteer_idx, recipient_idx]

        better_volunteers = []
        for v_idx in range(self.num_volunteers):
            if v_idx == volunteer_idx:
                continue
            volunteer = self.volunteers[v_idx]
            current_load = sum(self.recipients[r].num_items for r in self.volunteer_assignments.get(v_idx, []))
            remaining_capacity = volunteer.car_size - current_load

            if remaining_capacity >= recipient_items:
                distance = self.distance_matrix[v_idx, recipient_idx]
                if distance < chosen_distance * 0.8:
                    better_volunteers.append((v_idx, distance, remaining_capacity))

        if better_volunteers:
            better_volunteers.sort(key=lambda x: x[1])
            for v_idx, dist, cap in better_volunteers[:3]:
                nearby = []
                for r_idx in range(self.num_recipients):
                    if r_idx != recipient_idx and r_idx not in self.assigned_recipients:
                        r1_lat = recipient.latitude
                        r1_lon = recipient.longitude
                        r2_lat = self.recipients[r_idx].latitude
                        r2_lon = self.recipients[r_idx].longitude
                        d = self._haversine_distance(r1_lat, r1_lon, r2_lat, r2_lon)
                        if d < 3.0 and cap >= recipient_items + self.recipients[r_idx].num_items:
                            nearby.append((r_idx, d))

                if nearby:
                    v_lat = self.volunteers[v_idx].latitude
                    v_lon = self.volunteers[v_idx].longitude
                    r_lat = recipient.latitude
                    r_lon = recipient.longitude
                    main_bearing = self._calculate_bearing(v_lat, v_lon, r_lat, r_lon)

                    consistent = True
                    for r_idx, _ in nearby:
                        r2_lat = self.recipients[r_idx].latitude
                        r2_lon = self.recipients[r_idx].longitude
                        bearing = self._calculate_bearing(v_lat, v_lon, r2_lat, r2_lon)
                        if self._bearing_difference(main_bearing, bearing) > 45:
                            consistent = False
                            break

                    if consistent and len(nearby) >= 2:
                        return -4.0
                    elif consistent:
                        return -2.0

            return -1.0  # Closer but not clusterable

        return 1.0  # Good choice

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
        assignment_reward = 0.0
        step_penalty = -0.1
        invalid_action_penalty = 0.0
        end_penalty = 0.0

        if not valid_action:
            print(f"[DEBUG] Invalid action: Volunteer {volunteer_idx}, Recipient {recipient_idx}. Ending episode.")
            info = {
                'valid_action': False,
                'volunteer_idx': volunteer_idx,
                'recipient_idx': recipient_idx,
                'assigned_count': len(self.assigned_recipients),
                'total_recipients': self.num_recipients
            }
            return self.state, -1.0, True, info

        # Update assignments (only if valid)
        self.assignment_list.append((volunteer_idx, recipient_idx))
        self.assigned_recipients.add(recipient_idx)
        
        if volunteer_idx not in self.volunteer_assignments:
            self.volunteer_assignments[volunteer_idx] = []
        self.volunteer_assignments[volunteer_idx].append(recipient_idx)
        
        # Compute reward for this assignment
        assignment_reward = self._compute_reward(volunteer_idx, recipient_idx)
        reward += assignment_reward

        # Penalize every step
        reward += step_penalty

        
        # Update state
        self.state = self._compute_state()
        print(f"State updated")
        
        # Increment step counter
        self.current_step += 1
        
        # Check if episode is done
        done = len(self.assigned_recipients) == self.num_recipients or self.current_step >= self.max_steps
        
        # Check for valid actions
        valid_actions_exist = False
        for v_idx in range(self.num_volunteers):
            current_load = sum(self.recipients[r_idx].num_items 
                              for r_idx in self.volunteer_assignments.get(v_idx, []))
            for r_idx in range(self.num_recipients):
                if r_idx not in self.assigned_recipients:
                    if current_load + self.recipients[r_idx].num_items <= self.volunteers[v_idx].car_size:
                        valid_actions_exist = True
                        break
            if valid_actions_exist: 
                break
        print(f"Valid actions check")
        if not valid_actions_exist or self.current_step >= self.max_steps:
            done = True

        # At episode end, calculate final rewards
        if done:
            # 1. Penalize for unassigned recipients
            num_unassigned = self.num_recipients - len(self.assigned_recipients)
            unassigned_penalty = -5.0 * num_unassigned  # Increased penalty for unassigned recipients
            reward += unassigned_penalty
            print(f"Unassigned penalty: {unassigned_penalty}")
            
            # 2. Reward for efficient volunteer usage
            if len(self.assigned_recipients) > 0:  # Only if we've made some assignments
                # Count active volunteers (those with at least one assignment)
                active_volunteers = len(self.volunteer_assignments)
                
                # Calculate average capacity utilization for active volunteers
                total_utilization = 0.0
                high_utilization_count = 0
                
                for v_idx, r_indices in self.volunteer_assignments.items():
                    volunteer = self.volunteers[v_idx]
                    current_load = sum(self.recipients[r_idx].num_items for r_idx in r_indices)
                    utilization = current_load / volunteer.car_size
                    total_utilization += utilization
                    
                    # Count volunteers with high utilization (>80%)
                    if utilization >= 0.9:
                        high_utilization_count += 1
                
                # Average utilization across active volunteers
                avg_utilization = total_utilization / active_volunteers if active_volunteers > 0 else 0
                
                # Reward based on percentage of volunteers with high utilization
                high_util_percentage = high_utilization_count / active_volunteers if active_volunteers > 0 else 0
                high_util_reward = 10.0 * high_util_percentage
                print(f"High utilization reward: {high_util_reward}")
                
                # Reward for using fewer volunteers (relative to total recipients)
                volunteer_efficiency = 1.0 - (active_volunteers / self.num_volunteers)
                volunteer_count_reward = 5.0 * volunteer_efficiency
                
                # 3. Route efficiency reward - evaluate geographical compactness of routes
                route_efficiency_reward = 0.0
                
                for v_idx, r_indices in self.volunteer_assignments.items():
                    if len(r_indices) <= 1:  # Skip volunteers with 0 or 1 recipient
                        continue
                        
                    # Calculate all pairwise distances between assigned recipients
                    all_distances = []
                    for i, r1_idx in enumerate(r_indices):
                        r1_lat, r1_lon = self.recipients[r1_idx].latitude, self.recipients[r1_idx].longitude
                        
                        for r2_idx in r_indices[i+1:]:
                            r2_lat, r2_lon = self.recipients[r2_idx].latitude, self.recipients[r2_idx].longitude
                            dist = self._haversine_distance(r1_lat, r1_lon, r2_lat, r2_lon)
                            all_distances.append(dist)
                    
                    # Calculate route compactness metrics
                    if all_distances:
                        avg_distance = sum(all_distances) / len(all_distances)
                        max_distance = max(all_distances) if all_distances else 0
                        
                        # Reward compact routes, penalize spread out routes
                        if avg_distance < 2:  # Very compact route (< 3km between stops on average)
                            route_efficiency_reward += 3.0
                        elif avg_distance < 4:  # Reasonably compact route
                            route_efficiency_reward += 1.5
                        elif avg_distance > 10:  # Very spread out route
                            route_efficiency_reward -= 2.0 * (avg_distance / 10.0)  # Scales with distance
                        
                        # Extra penalty for routes with any very distant points
                        if max_distance > 15:  # Any stops more than 15km apart
                            route_efficiency_reward -= 3.0
                print(f"Route efficiency reward: {route_efficiency_reward}")
                # Add all rewards
                efficiency_reward = high_util_reward + volunteer_count_reward + route_efficiency_reward
                reward += efficiency_reward

        # Debug logging
        # print(f"\n[DEBUG] Step: {self.current_step}")
        # print(f"  Action: Volunteer {volunteer_idx}, Recipient {recipient_idx}, Valid: {valid_action}")
        # print(f"  Assignment reward: {assignment_reward:.2f}, Step penalty: {step_penalty:.2f}, Invalid action penalty: {invalid_action_penalty:.2f}, End penalty: {end_penalty:.2f}")
        # print(f"  Total reward this step: {reward:.2f}")
        # print(f"  Assigned recipients: {len(self.assigned_recipients)}/{self.num_recipients}")
        # print(f"  Unassigned recipients: {self.num_recipients - len(self.assigned_recipients)}")
        # if done:
        #     if len(self.assigned_recipients) == self.num_recipients:
        #         print("  [EPISODE END] All recipients assigned!")
        #     elif self.current_step >= self.max_steps:
        #         print("  [EPISODE END] Max steps reached.")
        #     else:
        #         print("  [EPISODE END] No valid actions left.")

        # Additional info
        info = {
            'valid_action': valid_action,
            'volunteer_idx': volunteer_idx,
            'recipient_idx': recipient_idx,
            'assigned_count': len(self.assigned_recipients),
            'total_recipients': self.num_recipients
        }
        
        return self.state, reward, done, info
    
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
