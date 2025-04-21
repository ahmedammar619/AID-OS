#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Optimization solver module for the AID-OS project.
Uses Google OR-Tools to solve the volunteer-recipient assignment problem.
"""

import numpy as np
import pandas as pd
import os
import sys
import time
from datetime import datetime
import math

# Add parent directory to path to import from project modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from data.db_config import DatabaseHandler
from clustering.dbscan_cluster import RecipientClusterer

# Import OR-Tools
from ortools.linear_solver import pywraplp

class OptimizationSolver:
    """
    Class for solving the volunteer-recipient assignment problem using optimization.
    
    This class provides functionality to:
    1. Formulate the assignment problem as a mixed-integer program
    2. Solve the problem using Google OR-Tools
    3. Extract the solution and generate assignments
    """
    
    def __init__(
        self,
        db_handler=None,
        use_clustering=True,
        cluster_eps=0.00005,
        output_dir="./output"
    ):
        """
        Initialize the optimization solver.
        
        Args:
            db_handler (DatabaseHandler): Database connection handler
            use_clustering (bool): Whether to use clustering for assignments
            cluster_eps (float): Epsilon parameter for DBSCAN clustering
            output_dir (str): Directory to save output files
        """
        # Initialize handlers
        self.db_handler = db_handler if db_handler is not None else DatabaseHandler()
        
        # Create output directory if it doesn't exist
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        
        # Load data
        self.load_data()
        
        # Initialize clusterer if needed
        self.use_clustering = use_clustering
        if use_clustering:
            self.clusterer = RecipientClusterer(
                min_cluster_size=2,
                cluster_selection_epsilon=cluster_eps,
                min_samples=1
            )
            # Extract coordinates for clustering
            self.recipient_coords = np.array([[r.latitude, r.longitude] 
                                             for r in self.recipients])
            # Perform clustering
            self.clusterer.fit(self.recipient_coords)
            clusters_data = self.clusterer.get_clusters()
            
            # Convert cluster labels to a dictionary mapping cluster_id -> list of recipient indices
            self.clusters = {}
            for i, label in enumerate(clusters_data['labels']):
                if label not in self.clusters:
                    self.clusters[label] = []
                self.clusters[label].append(i)
        
        # Create distance matrix
        self.distance_matrix = self._create_distance_matrix()
        
        # Initialize assignments
        self.assignments = []
        self.assignment_map = {}  # volunteer_id -> [recipient_ids]
        
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

        # Extract coordinates
        self.volunteer_coords = np.array([[v.latitude, v.longitude] 
                                         for v in self.volunteers])
        self.recipient_coords = np.array([[r.latitude, r.longitude] 
                                         for r in self.recipients])
    
    def _create_distance_matrix(self):
        """
        Create a distance matrix between all volunteers and recipients.
        
        Returns:
            distance_matrix (numpy.ndarray): Matrix of distances
                                            [volunteer_idx][recipient_idx]
        """
        distance_matrix = np.zeros((self.num_volunteers, self.num_recipients))
        
        for v_idx in range(self.num_volunteers):
            v_lat = self.volunteers[v_idx].latitude
            v_lon = self.volunteers[v_idx].longitude
            
            for r_idx in range(self.num_recipients):
                r_lat = self.recipients[r_idx].latitude
                r_lon = self.recipients[r_idx].longitude
                
                # Calculate Haversine distance
                distance = self._haversine_distance(v_lat, v_lon, r_lat, r_lon)
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
        lat1, lon1, lat2, lon2 = map(math.radians, [lat1, lon1, lat2, lon2])
        
        # Haversine formula
        dlat = lat2 - lat1
        dlon = lon2 - lon1
        a = math.sin(dlat/2)**2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon/2)**2
        c = 2 * math.asin(math.sqrt(a))
        r = 6371  # Radius of Earth in kilometers
        
        return c * r
    
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
        
        # Get score from database
        return self.db_handler.get_volunteer_historical_score(volunteer_id, recipient_id)
    
    def solve(self):
        """
        Solve the volunteer-recipient assignment problem.
        
        Returns:
            bool: Whether the solution was successful
        """
        print("Solving assignment problem using optimization...")
        start_time = time.time()
        
        # Create the solver
        solver = pywraplp.Solver.CreateSolver('SCIP')
        if not solver:
            print("Could not create solver!")
            return False
        
        # Decision variables
        # x[v][r] = 1 if volunteer v is assigned to recipient r, 0 otherwise
        x = {}
        for v in range(self.num_volunteers):
            for r in range(self.num_recipients):
                x[v, r] = solver.BoolVar(f'x_{v}_{r}')
        
        # y[v] = 1 if volunteer v is used, 0 otherwise
        y = {}
        for v in range(self.num_volunteers):
            y[v] = solver.BoolVar(f'y_{v}')
        
        # Constraints
        
        # 1. Each recipient must be assigned to exactly one volunteer
        for r in range(self.num_recipients):
            solver.Add(sum(x[v, r] for v in range(self.num_volunteers)) == 1)
        
        # 2. Volunteer capacity constraints
        for v in range(self.num_volunteers):
            # Sum of boxes assigned to volunteer v must not exceed capacity
            solver.Add(
                sum(x[v, r] * self.recipients[r].num_items 
                    for r in range(self.num_recipients)) 
                <= self.volunteers[v].car_size
            )
        
        # 3. Link y variables to x variables
        for v in range(self.num_volunteers):
            # If any recipient is assigned to volunteer v, then y[v] = 1
            solver.Add(
                sum(x[v, r] for r in range(self.num_recipients)) 
                <= self.num_recipients * y[v]
            )
        
        # Objective function
        objective = solver.Objective()
        
        # Minimize total distance traveled
        distance_weight = 1.0
        for v in range(self.num_volunteers):
            for r in range(self.num_recipients):
                objective.SetCoefficient(
                    x[v, r], 
                    distance_weight * self.distance_matrix[v, r]
                )
        
        # Minimize number of volunteers used
        volunteer_weight = 50.0  # Weight for minimizing number of volunteers
        for v in range(self.num_volunteers):
            objective.SetCoefficient(y[v], volunteer_weight)
        
        # Add historical match bonuses (negative cost)
        history_weight = -5.0  # Negative weight to encourage historical matches
        for v in range(self.num_volunteers):
            for r in range(self.num_recipients):
                historical_score = self._get_historical_match_score(v, r)
                if historical_score > 0:
                    objective.SetCoefficient(
                        x[v, r],
                        history_weight * historical_score
                    )
        
        # Add cluster bonuses (negative cost for keeping clusters together)
        if self.use_clustering:
            cluster_weight = -10.0  # Negative weight to encourage keeping clusters together
            for cluster_id, recipient_indices in self.clusters.items():
                if cluster_id != -1 and len(recipient_indices) > 1:  # Skip noise cluster (-1)
                    for v in range(self.num_volunteers):
                        # For each pair of recipients in the same cluster
                        for i in range(len(recipient_indices)):
                            for j in range(i+1, len(recipient_indices)):
                                r1 = recipient_indices[i]
                                r2 = recipient_indices[j]
                                # Skip invalid indices (like -1 for noise in DBSCAN)
                                if r1 < 0 or r2 < 0 or r1 >= self.num_recipients or r2 >= self.num_recipients:
                                    continue
                                # Add bonus for assigning both to the same volunteer
                                # This is a quadratic term, so we need to linearize it
                                # We add a new variable z[v, r1, r2] = x[v, r1] * x[v, r2]
                                z = solver.BoolVar(f'z_{v}_{r1}_{r2}')
                                # z <= x[v, r1]
                                solver.Add(z <= x[v, r1])
                                # z <= x[v, r2]
                                solver.Add(z <= x[v, r2])
                                # z >= x[v, r1] + x[v, r2] - 1
                                solver.Add(z >= x[v, r1] + x[v, r2] - 1)
                                # Add to objective
                                objective.SetCoefficient(z, cluster_weight)
        
        # Set objective to minimize
        objective.SetMinimization()
        
        # Solve the problem
        status = solver.Solve()
        end_time = time.time()
        
        # Check if the problem was solved
        if status == pywraplp.Solver.OPTIMAL or status == pywraplp.Solver.FEASIBLE:
            print(f"Solution found in {end_time - start_time:.2f} seconds!")
            print(f"Objective value = {objective.Value()}")
            
            # Extract the solution
            self.assignments = []
            self.assignment_map = {}
            
            for v in range(self.num_volunteers):
                volunteer_id = self.volunteers[v].volunteer_id
                self.assignment_map[volunteer_id] = []
                
                for r in range(self.num_recipients):
                    if x[v, r].solution_value() > 0.5:  # Variable is 1
                        recipient_id = self.recipients[r].recipient_id
                        self.assignments.append((volunteer_id, recipient_id))
                        self.assignment_map[volunteer_id].append(recipient_id)
            
            return True
        else:
            print("No solution found.")
            return False
    
    def save_assignments_to_db(self):
        """
        Save the generated assignments to the database.
        
        Returns:
            bool: Whether assignments were successfully saved
        """
        try:
            # Save to database
            self.db_handler.bulk_save_assignments(self.assignments)
            return True
        except Exception as e:
            print(f"Error saving assignments: {e}")
            return False
    
    def export_assignments_to_csv(self, filename=None):
        """
        Export the generated assignments to a CSV file.
        
        Args:
            filename (str, optional): Name of the file to save to
            
        Returns:
            str: Path to the saved file
        """
        if not self.assignments:
            print("No assignments to export!")
            return None
        
        # Create a DataFrame for the assignments
        data = []
        
        for volunteer_id, recipient_id in self.assignments:
            # Find the volunteer and recipient objects
            volunteer = next(v for v in self.volunteers if v.volunteer_id == volunteer_id)
            recipient = next(r for r in self.recipients if r.recipient_id == recipient_id)
            
            # Add to data
            data.append({
                'volunteer_id': volunteer_id,
                'volunteer_car_size': volunteer.car_size,
                'recipient_id': recipient_id,
                'recipient_num_items': recipient.num_items,
                'distance_km': self._haversine_distance(
                    volunteer.latitude, volunteer.longitude,
                    recipient.latitude, recipient.longitude
                )
            })
        
        # Create DataFrame
        df = pd.DataFrame(data)
        
        # Generate filename if not provided
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"assignments_{timestamp}.csv"
        
        # Save to CSV
        filepath = os.path.join(self.output_dir, filename)
        df.to_csv(filepath, index=False)
        
        print(f"Assignments exported to {filepath}")
        return filepath
    
    def _calculate_route_travel_time(self, volunteer_idx, recipient_indices):
        """
        Calculate the estimated travel time for a volunteer's route.
        Assumes an average speed of 30 km/h in urban areas.
        
        Args:
            volunteer_idx (int): Index of the volunteer
            recipient_indices (list): List of recipient indices assigned to this volunteer
            
        Returns:
            float: Estimated travel time in minutes
            float: Total distance in kilometers
        """
        if not recipient_indices:
            return 0.0, 0.0
        
        # Get volunteer coordinates
        v_lat = self.volunteers[volunteer_idx].latitude
        v_lon = self.volunteers[volunteer_idx].longitude
        
        # Get recipient coordinates
        recipient_coords = [(self.recipients[r_idx].latitude, 
                             self.recipients[r_idx].longitude) 
                            for r_idx in recipient_indices]
        
        # Simple greedy algorithm for route planning
        # Start from volunteer location
        current_lat, current_lon = v_lat, v_lon
        unvisited = recipient_coords.copy()
        route = []
        total_distance = 0.0
        
        # Visit each recipient in order of nearest neighbor
        while unvisited:
            # Find the nearest unvisited recipient
            distances = [self._haversine_distance(current_lat, current_lon, r_lat, r_lon) 
                         for r_lat, r_lon in unvisited]
            nearest_idx = distances.index(min(distances))
            
            # Add to route and update distance
            next_lat, next_lon = unvisited[nearest_idx]
            route.append((next_lat, next_lon))
            total_distance += distances[nearest_idx]
            
            # Update current position and remove from unvisited
            current_lat, current_lon = next_lat, next_lon
            unvisited.pop(nearest_idx)
        
        # Add return to volunteer location
        total_distance += self._haversine_distance(current_lat, current_lon, v_lat, v_lon)
        
        # Calculate travel time (assuming 30 km/h average speed)
        # Add 5 minutes per stop for delivery time
        travel_time = (total_distance / 30.0) * 60.0 + len(recipient_indices) * 5.0
        
        return travel_time, total_distance
    
    def run_complete_pipeline(self, export_csv=True, save_visualizations=True, save_report=True):
        """
        Run the complete assignment pipeline.
        
        Args:
            export_csv (bool): Whether to export assignments to CSV
            save_visualizations (bool): Whether to save visualizations
            save_report (bool): Whether to save the report
            
        Returns:
            bool: Whether the pipeline was successful
        """
        # 1. Solve the optimization problem
        success = self.solve()
        if not success:
            return False
        
        # 2. Save assignments to database
        self.save_assignments_to_db()
        
        # 3. Export to CSV if requested
        if export_csv:
            self.export_assignments_to_csv()
        
        print("Assignment pipeline completed successfully!")
        return True


if __name__ == "__main__":
    # Test the solver
    solver = OptimizationSolver()
    success = solver.solve()
    
    if success:
        # Export assignments
        solver.export_assignments_to_csv()
