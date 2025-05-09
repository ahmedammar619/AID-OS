#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Assignment module for the AID-OS project.
Uses Google OR-Tools to generate optimal volunteer-recipient assignments, with clustering, visualization, and reporting.
"""

import numpy as np
import pandas as pd
import os
import sys
import time
from datetime import datetime
import math
import matplotlib.pyplot as plt
import folium
import seaborn as sns

# Add parent directory to path to import from project modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from data.db_config import DatabaseHandler
from clustering.dbscan_cluster import RecipientClusterer
from feedback.feedback_handler import FeedbackHandler

# Import OR-Tools
from ortools.linear_solver import pywraplp

class VolunteerAssignerOpt:
    """
    Class for assigning volunteers to recipients using Google OR-Tools optimization.
    
    Provides functionality to:
    1. Formulate and solve the assignment problem as a mixed-integer program.
    2. Apply DBSCAN clustering to group recipients.
    3. Generate visualizations (maps, load distribution).
    4. Produce detailed reports.
    5. Save assignments to database and CSV.
    """
    
    def __init__(
        self,
        db_handler=None,
        feedback_handler=None,
        use_clustering=True,
        cluster_eps=0.00005,
        output_dir="./hist/output"
    ):
        """
        Initialize the volunteer assigner.
        
        Args:
            db_handler (DatabaseHandler): Database connection handler.
            feedback_handler (FeedbackHandler): Feedback handler for admin input.
            use_clustering (bool): Whether to use clustering for assignments.
            cluster_eps (float): Epsilon parameter for DBSCAN clustering.
            output_dir (str): Directory to save output files.
        """
        # Initialize handlers
        self.db_handler = db_handler if db_handler is not None else DatabaseHandler()
        # self.feedback_handler = feedback_handler if feedback_handler is not None else FeedbackHandler()
        
        # Create output directory
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        
        # Initialize data
        self.volunteers = []
        self.recipients = []
        self.pickups = []
        self.historical_data = []
        self.num_volunteers = 0
        self.num_recipients = 0
        self.distance_matrix = None
        self.clusters = {}
        self.assignments = []
        self.assignment_map = {}  # volunteer_id -> [recipient_ids]
        
        # Clustering settings
        self.use_clustering = use_clustering
        self.cluster_eps = cluster_eps
        
        # Load data and initialize clustering
        self._load_data()
        if use_clustering:
            self._initialize_clustering()
        
        # Create distance matrix
        self.distance_matrix = self._create_distance_matrix()
    
    def _load_data(self):
        """Load volunteer, recipient, pickup, and historical data from the database."""
        self.volunteers = self.db_handler.get_all_volunteers()
        self.num_volunteers = len(self.volunteers)
        
        self.recipients = self.db_handler.get_all_recipients()
        self.num_recipients = len(self.recipients)
        
        self.pickups = self.db_handler.get_all_pickups()
        self.num_pickups = len(self.pickups)
        
        self.historical_data = self.db_handler.get_historical_deliveries()
        
        self.volunteer_coords = np.array([[v.latitude, v.longitude] for v in self.volunteers])
        self.recipient_coords = np.array([[r.latitude, r.longitude] for r in self.recipients])
    
    def _initialize_clustering(self):
        """Initialize DBSCAN clustering for recipients."""
        self.clusterer = RecipientClusterer(
            min_cluster_size=2,
            cluster_selection_epsilon=self.cluster_eps,
            min_samples=1
        )
        self.clusterer.fit(self.recipient_coords)
        clusters_data = self.clusterer.get_clusters()
        
        # Convert cluster labels to dictionary: cluster_id -> [recipient_indices]
        for i, label in enumerate(clusters_data['labels']):
            if label not in self.clusters:
                self.clusters[label] = []
            self.clusters[label].append(i)
    
    def _haversine_distance(self, lat1, lon1, lat2, lon2):
        """
        Calculate the Haversine distance between two points on Earth.
        
        Args:
            lat1, lon1: Coordinates of the first point (degrees).
            lat2, lon2: Coordinates of the second point (degrees).
            
        Returns:
            float: Distance in kilometers.
        """
        lat1, lon1, lat2, lon2 = map(math.radians, [lat1, lon1, lat2, lon2])
        dlat = lat2 - lat1
        dlon = lon2 - lon1
        a = math.sin(dlat/2)**2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon/2)**2
        c = 2 * math.asin(math.sqrt(a))
        r = 6371  # Earth radius in kilometers
        return c * r
    
    def _create_distance_matrix(self):
        """
        Create comprehensive distance matrices between volunteers, pickups, and recipients.
        
        Returns:
            dict: Dictionary containing various distance matrices:
                - vol_to_recip: Direct distances from volunteers to recipients
                - vol_to_pickup: Distances from volunteers to pickup locations
                - pickup_to_recip: Distances from pickup locations to recipients
                - vol_pickup_recip: Combined distances for the route: volunteer -> pickup -> recipient
        """
        # Initialize distance matrices
        vol_to_recip = np.zeros((self.num_volunteers, self.num_recipients))
        vol_to_pickup = np.zeros((self.num_volunteers, len(self.pickups)))
        pickup_to_recip = np.zeros((len(self.pickups), self.num_recipients))
        
        # Calculate volunteer to recipient distances (direct)
        for v_idx in range(self.num_volunteers):
            v_lat, v_lon = self.volunteers[v_idx].latitude, self.volunteers[v_idx].longitude
            for r_idx in range(self.num_recipients):
                r_lat, r_lon = self.recipients[r_idx].latitude, self.recipients[r_idx].longitude
                vol_to_recip[v_idx, r_idx] = self._haversine_distance(v_lat, v_lon, r_lat, r_lon)
        
        # Calculate volunteer to pickup distances
        for v_idx in range(self.num_volunteers):
            v_lat, v_lon = self.volunteers[v_idx].latitude, self.volunteers[v_idx].longitude
            for p_idx, pickup in enumerate(self.pickups):
                p_lat, p_lon = pickup.latitude, pickup.longitude
                vol_to_pickup[v_idx, p_idx] = self._haversine_distance(v_lat, v_lon, p_lat, p_lon)
        
        # Calculate pickup to recipient distances
        for p_idx, pickup in enumerate(self.pickups):
            p_lat, p_lon = pickup.latitude, pickup.longitude
            for r_idx in range(self.num_recipients):
                r_lat, r_lon = self.recipients[r_idx].latitude, self.recipients[r_idx].longitude
                pickup_to_recip[p_idx, r_idx] = self._haversine_distance(p_lat, p_lon, r_lat, r_lon)
        
        # Create a 3D matrix for the combined route: volunteer -> pickup -> recipient
        # This represents the total distance for each volunteer-pickup-recipient combination
        vol_pickup_recip = np.zeros((self.num_volunteers, len(self.pickups), self.num_recipients))
        for v_idx in range(self.num_volunteers):
            for p_idx in range(len(self.pickups)):
                for r_idx in range(self.num_recipients):
                    # Total distance: volunteer -> pickup -> recipient
                    vol_pickup_recip[v_idx, p_idx, r_idx] = vol_to_pickup[v_idx, p_idx] + pickup_to_recip[p_idx, r_idx]
        
        return {
            'vol_to_recip': vol_to_recip,  # Direct distances (for comparison)
            'vol_to_pickup': vol_to_pickup,
            'pickup_to_recip': pickup_to_recip,
            'vol_pickup_recip': vol_pickup_recip  # Combined route distances
        }
    
    def _get_historical_match_score(self, volunteer_idx, recipient_idx):
        """
        Calculate a historical match score for a volunteer-recipient pair.
        
        Args:
            volunteer_idx (int): Index of the volunteer.
            recipient_idx (int): Index of the recipient.
            
        Returns:
            float: Historical match score (0-3).
        """
        volunteer_id = self.volunteers[volunteer_idx].volunteer_id
        recipient_id = self.recipients[recipient_idx].recipient_id
        return self.db_handler.get_volunteer_historical_score(volunteer_id, recipient_id)
    
    def _solve_tsp(self, points, start_idx=None, end_idx=None):
        """
        Solve the Traveling Salesman Problem using a simpler algorithm to avoid infinite loops.
        This uses a greedy nearest neighbor approach with a small improvement phase.
        
        Args:
            points (list): List of (lat, lon) coordinates.
            start_idx (int, optional): Index of the starting point (fixed).
            end_idx (int, optional): Index of the ending point (fixed).
            
        Returns:
            list: Optimized route indices.
            float: Total route distance.
        """
        n = len(points)
        if n <= 2:
            return list(range(n)), 0.0
        
        # Create distance matrix
        dist_matrix = np.zeros((n, n))
        for i in range(n):
            for j in range(i+1, n):
                dist = self._haversine_distance(
                    points[i][0], points[i][1],
                    points[j][0], points[j][1]
                )
                dist_matrix[i, j] = dist
                dist_matrix[j, i] = dist
        
        # Initialize route with starting point
        if start_idx is not None:
            route = [start_idx]
            current = start_idx
        else:
            route = [0]
            current = 0
        
        # Collect unvisited points
        unvisited = set(range(n))
        unvisited.remove(current)
        
        # If end point is specified, remove it from unvisited as we'll add it last
        if end_idx is not None and end_idx != start_idx:
            if end_idx in unvisited:
                unvisited.remove(end_idx)
        
        # Simple nearest neighbor algorithm
        while unvisited:
            # Find closest unvisited point
            next_point = min(unvisited, key=lambda x: dist_matrix[current, x])
            route.append(next_point)
            current = next_point
            unvisited.remove(next_point)
        
        # Add end point if specified
        if end_idx is not None and end_idx != start_idx and end_idx not in route:
            route.append(end_idx)
        
        # Simple improvement: check if any single swap improves the route
        # Limit to max 10 iterations to prevent infinite loops
        max_iterations = 10
        for _ in range(max_iterations):
            improved = False
            best_improvement = 0
            best_swap = None
            
            # Check all possible swaps of non-fixed points
            for i in range(1, len(route) - 1):
                if start_idx is not None and i == 0:
                    continue  # Skip start point
                    
                for j in range(i + 1, len(route)):
                    if end_idx is not None and j == len(route) - 1:
                        continue  # Skip end point
                    
                    # Calculate current segment distances
                    if i == 0:
                        d1 = 0  # No previous segment for first point
                    else:
                        d1 = dist_matrix[route[i-1], route[i]]
                        
                    if j == len(route) - 1:
                        d2 = 0  # No next segment for last point
                    else:
                        d2 = dist_matrix[route[j], route[j+1]]
                    
                    # Calculate new segment distances if we swap
                    if i == 0:
                        d3 = 0
                    else:
                        d3 = dist_matrix[route[i-1], route[j]]
                        
                    if j == len(route) - 1:
                        d4 = 0
                    else:
                        d4 = dist_matrix[route[i], route[j+1]]
                    
                    # Calculate improvement
                    improvement = (d1 + d2) - (d3 + d4)
                    if improvement > best_improvement:
                        best_improvement = improvement
                        best_swap = (i, j)
            
            # Apply the best swap if it improves the route
            if best_swap and best_improvement > 0:
                i, j = best_swap
                route[i], route[j] = route[j], route[i]
                improved = True
            
            if not improved:
                break
        
        # Calculate total distance
        total_dist = 0.0
        for i in range(len(route) - 1):
            total_dist += dist_matrix[route[i], route[i+1]]
        
        return route, total_dist
    
    def _calculate_route_travel_time(self, volunteer_idx, recipient_indices, pickup_idx=None):
        """
        Calculate estimated travel time and distance for a volunteer's route.
        Route: volunteer home -> pickup location -> recipients -> volunteer home
        Uses an optimized TSP solution for routing, assumes 30 km/h speed and 5 min/stop.
        
        Args:
            volunteer_idx (int): Index of the volunteer.
            recipient_indices (list): List of recipient indices.
            pickup_idx (int, optional): Index of the pickup location. If None, uses the closest pickup.
            
        Returns:
            float: Travel time in minutes.
            float: Total distance in kilometers.
            list: Ordered list of coordinates for the route [volunteer, pickup, recipients..., volunteer]
        """
        if not recipient_indices:
            return 0.0, 0.0, []
        
        # Get volunteer coordinates
        v_lat, v_lon = self.volunteers[volunteer_idx].latitude, self.volunteers[volunteer_idx].longitude
        
        # Determine pickup location (use closest if not specified)
        if pickup_idx is None:
            # Find closest pickup to volunteer
            pickup_distances = [self._haversine_distance(v_lat, v_lon, p.latitude, p.longitude) 
                              for p in self.pickups]
            pickup_idx = pickup_distances.index(min(pickup_distances))
        
        pickup = self.pickups[pickup_idx]
        p_lat, p_lon = pickup.latitude, pickup.longitude
        
        # Get recipient coordinates
        recipient_coords = [(self.recipients[r_idx].latitude, self.recipients[r_idx].longitude)
                           for r_idx in recipient_indices]
        
        # Calculate direct distance: volunteer -> pickup
        vol_to_pickup_dist = self._haversine_distance(v_lat, v_lon, p_lat, p_lon)
        
        # If there are no recipients, just return the round trip to pickup
        if not recipient_coords:
            return (vol_to_pickup_dist * 2 / 30.0) * 60.0 + 10.0, vol_to_pickup_dist * 2, [(v_lat, v_lon), (p_lat, p_lon), (v_lat, v_lon)]
        
        # Create points list for TSP: [pickup, recipient1, recipient2, ..., recipientN]
        # Note: We'll handle volunteer->pickup and last_recipient->volunteer separately
        tsp_points = [(p_lat, p_lon)] + recipient_coords
        
        # Solve TSP with pickup as the fixed starting point (index 0)
        route_indices, route_distance = self._solve_tsp(tsp_points, start_idx=0)
        
        # Create the full route coordinates
        route_coords = [(v_lat, v_lon)]  # Start at volunteer's home
        
        # Add the TSP route points in the optimized order
        for idx in route_indices:
            route_coords.append(tsp_points[idx])
        
        # Add the return to volunteer's home
        route_coords.append((v_lat, v_lon))
        
        # Calculate total distance including return to volunteer
        last_point = tsp_points[route_indices[-1]]
        last_to_vol_dist = self._haversine_distance(last_point[0], last_point[1], v_lat, v_lon)
        total_distance = vol_to_pickup_dist + route_distance + last_to_vol_dist
        
        # Calculate travel time (30 km/h driving + 5 min per stop + 10 min at pickup)
        travel_time = (total_distance / 30.0) * 60.0 + len(recipient_indices) * 5.0 + 10.0
        
        return travel_time, total_distance, route_coords
    
    def generate_assignments(self):
        """
        Solve the volunteer-recipient assignment problem using OR-Tools MILP.
        The model now includes pickup locations in the routing, but with a simplified approach.
        We pre-assign each volunteer to their closest pickup location to reduce complexity.
        
        Returns:
            bool: Whether the solution was successful.
        """
        print("Solving assignment problem using optimization with pickup locations...")
        start_time = time.time()
        
        # Step 1: Pre-assign each volunteer to their closest pickup location
        self.volunteer_pickup_assignments = {}
        for v_idx in range(self.num_volunteers):
            v_lat, v_lon = self.volunteers[v_idx].latitude, self.volunteers[v_idx].longitude
            
            # Find closest pickup location
            pickup_distances = [self._haversine_distance(v_lat, v_lon, p.latitude, p.longitude) 
                              for p in self.pickups]
            closest_pickup_idx = pickup_distances.index(min(pickup_distances))
            self.volunteer_pickup_assignments[v_idx] = closest_pickup_idx
            
        print(f"Pre-assigned {len(self.volunteer_pickup_assignments)} volunteers to their closest pickup locations")
        
        # Step 2: Create solver and decision variables
        solver = pywraplp.Solver.CreateSolver('SCIP')
        if not solver:
            print("Could not create solver!")
            return False
        
        # Decision variables
        # x[v,r] = 1 if volunteer v is assigned to recipient r
        x = {(v, r): solver.BoolVar(f'x_{v}_{r}')
             for v in range(self.num_volunteers) for r in range(self.num_recipients)}
        
        # y[v] = 1 if volunteer v is used
        y = {v: solver.BoolVar(f'y_{v}') for v in range(self.num_volunteers)}

        
        # Constraints
        # 1. Each recipient assigned to exactly one volunteer
        for r in range(self.num_recipients):
            solver.Add(sum(x[v, r] for v in range(self.num_volunteers)) == 1)
        
        # 2. Volunteer capacity constraints
        for v in range(self.num_volunteers):
            solver.Add(
                sum(x[v, r] * self.recipients[r].num_items
                    for r in range(self.num_recipients))
                <= self.volunteers[v].car_size
            )
        
        # 3. Link y variables to x variables
        for v in range(self.num_volunteers):
            solver.Add(
                sum(x[v, r] for r in range(self.num_recipients))
                <= self.num_recipients * y[v]
            )
        
        # Objective function
        objective = solver.Objective()
        
        # Calculate normalization factors
        print("Calculating normalization factors for objective terms...")
        
        # For distance normalization - use the maximum distance from any matrix in our dictionary
        # First check the direct volunteer to recipient distances
        max_vol_recip_dist = max(max(row) for row in self.distance_matrix['vol_to_recip']) 
        # Then check the combined route distances (vol -> pickup -> recip)
        max_route_dist = max(
            max(
                max(self.distance_matrix['vol_pickup_recip'][v, k, r] 
                    for r in range(self.num_recipients))
                for k in range(len(self.pickups))
            ) 
            for v in range(self.num_volunteers)
        )
        max_distance = max(max_vol_recip_dist, max_route_dist)
        distance_norm = max_distance if max_distance > 0 else 1.0
        print(f"Max distance: {max_distance:.2f} km (normalization factor)")
        
        # For capacity utilization normalization
        max_capacity = max(v.car_size for v in self.volunteers) if self.volunteers else 1.0
        max_items = max(r.num_items for r in self.recipients) if self.recipients else 1.0
        capacity_norm = max_capacity if max_capacity > 0 else 1.0
        print(f"Max capacity: {max_capacity} boxes (normalization factor)")
        
        # For historical matches normalization
        max_history_score = 1.0  # Already normalized
        
        # For recipient distance normalization (used in compact routes)
        recipient_distance_norm = 5.0  # The threshold we're already using
        
        # Define normalized weights (all on same scale: 0-100)
        # Higher number means more importance
        weights = {
            'distance': 1.0,            # Minimize distance between volunteer and recipient
            'volunteer_count': 0.0,     # Minimize total number of volunteers used
            'capacity_util': 0.0,       # Maximize capacity utilization
            'history': 0.0,             # Prefer historical matches
            'compact_routes': 0.0,      # Prefer recipients close to each other (compact routes)
            'clusters': 0.0             # Prefer keeping clustered recipients together
        }
        print("Normalized weights (higher = more important):")
        for key, value in weights.items():
            print(f"  {key}: {value}")

        # --- HARD CONSTRAINTS FOR DISTANCE AND CAPACITY UTILIZATION ---
        # These constraints ensure that the total normalized distance and average capacity utilization
        # remain within specified bounds, regardless of the objective weights.
        # Adjust these thresholds as needed:
        DISTANCE_UPPER_BOUND_FACTOR = 1.15  # Allow up to 115% of minimum possible total distance
        MAX_ROUTE_TIME_MIN = 400.0  # 1.5 hours
        MIN_AVG_CAPACITY_UTIL = 0.4        # Require at least 40% average capacity utilization
        use_constraint = True
        if DISTANCE_UPPER_BOUND_FACTOR and use_constraint:
            # 1. Compute minimum possible total normalized distance with pickups
            # For each recipient, find the minimum distance through any volunteer's assigned pickup
            min_total_distance = 0.0
            for r in range(self.num_recipients):
                min_route_dist = float('inf')
                for v in range(self.num_volunteers):
                    pickup_idx = self.volunteer_pickup_assignments[v]
                    route_dist = self.distance_matrix['vol_pickup_recip'][v, pickup_idx, r]
                    min_route_dist = min(min_route_dist, route_dist)
                min_total_distance += min_route_dist
                
            min_total_normalized_distance = min_total_distance / distance_norm
            distance_upper_bound = DISTANCE_UPPER_BOUND_FACTOR * min_total_normalized_distance
            print(f"[Constraint] Upper bound for total normalized distance: {distance_upper_bound:.2f}")

            # Add constraint: total normalized distance assigned <= upper bound
            # With pre-assigned pickups, we can directly use the x variables
            total_normalized_distance_expr = solver.Sum(
                x[v, r] * (self.distance_matrix['vol_pickup_recip'][v, self.volunteer_pickup_assignments[v], r] / distance_norm)
                for v in range(self.num_volunteers) 
                for r in range(self.num_recipients)
            )
            solver.Add(total_normalized_distance_expr <= distance_upper_bound)


            for v in range(self.num_volunteers):
                # For each volunteer, sum the estimated travel time for all assigned recipients
                travel_time_expr = solver.Sum(
                    x[v, r] * (self.distance_matrix['vol_pickup_recip'][v, self.volunteer_pickup_assignments[v], r] / 30.0 * 60.0 + 5.0 + 10.0)
                    for r in range(self.num_recipients)
                )
            solver.Add(travel_time_expr <= MAX_ROUTE_TIME_MIN)

            # 2. Add constraint: average capacity utilization >= threshold
            # Compute total assigned boxes and total volunteer capacity
            # total_assigned_boxes_expr = solver.Sum(
            #     x[v, r] * self.recipients[r].num_items
            #     for v in range(self.num_volunteers) for r in range(self.num_recipients)
            # )
            # total_capacity = sum(v.car_size for v in self.volunteers)
            # min_total_assigned_boxes = MIN_AVG_CAPACITY_UTIL * total_capacity
            # print(f"[Constraint] Minimum total assigned boxes for utilization: {min_total_assigned_boxes:.2f} (of {total_capacity})")
            # solver.Add(total_assigned_boxes_expr >= min_total_assigned_boxes)

        # --- END HARD CONSTRAINTS ---
        
        # Minimize total route distance (volunteer -> pickup -> recipient)
        if weights['distance']:
            # For each volunteer-recipient combination, use the pre-assigned pickup
            for v in range(self.num_volunteers):
                pickup_idx = self.volunteer_pickup_assignments[v]
                for r in range(self.num_recipients):
                    # Get the combined route distance: volunteer -> pickup -> recipient
                    route_distance = self.distance_matrix['vol_pickup_recip'][v, pickup_idx, r]
                    normalized_distance = route_distance / distance_norm
                    
                    # Add the route distance to the objective directly
                    objective.SetCoefficient(x[v, r], weights['distance'] * normalized_distance)
        
        # Minimize number of volunteers (negative weight since we want to minimize)
        if weights['volunteer_count']:
            for v in range(self.num_volunteers):
                objective.SetCoefficient(y[v], -weights['volunteer_count'])
        
        # Historical match bonuses (negative weight since we want to maximize)
        if weights['history']:
            for v in range(self.num_volunteers):
                for r in range(self.num_recipients):
                    historical_score = self._get_historical_match_score(v, r)
                    if historical_score > 0:
                        # historical_score is already normalized (0-3)
                        normalized_history = historical_score / 3.0
                        objective.SetCoefficient(x[v, r], -weights['history'] * normalized_history)
        
        # Maximize volunteer capacity utilization (negative weight since we want to maximize)
        if weights['capacity_util']:
            for v in range(self.num_volunteers):
                for r in range(self.num_recipients):
                    # Normalize the contribution to 0-1 scale
                    contribution = self.recipients[r].num_items / self.volunteers[v].car_size
                    normalized_contribution = min(1.0, contribution)  # Cap at 1.0
                    objective.SetCoefficient(x[v, r], -weights['capacity_util'] * normalized_contribution)
        
        # Minimize recipient distances from each other (compact routes)
        if weights['compact_routes']:
            # Pre-compute distances and filter only close pairs to significantly reduce problem size
            close_recipient_pairs = []
            for r1 in range(self.num_recipients):
                for r2 in range(r1 + 1, self.num_recipients):
                    # Calculate distance between recipients
                    r1_lat, r1_lon = self.recipients[r1].latitude, self.recipients[r1].longitude
                    r2_lat, r2_lon = self.recipients[r2].latitude, self.recipients[r2].longitude
                    recip_distance = self._haversine_distance(r1_lat, r1_lon, r2_lat, r2_lon)
                    
                    # Only consider very close recipients
                    if recip_distance <= recipient_distance_norm:
                        close_recipient_pairs.append((r1, r2, recip_distance))
            
            # Limit the number of pairs to consider (take the closest N pairs)
            max_pairs = min(500, len(close_recipient_pairs))  # Cap at 500 pairs
            close_recipient_pairs.sort(key=lambda x: x[2])  # Sort by distance
            close_recipient_pairs = close_recipient_pairs[:max_pairs]  # Take only the closest pairs
            
            print(f"Using {len(close_recipient_pairs)} close recipient pairs for compact routes")
            
            # Only create variables for the filtered pairs (limit to top volunteers for performance)
            for v in range(min(10, self.num_volunteers)):
                for r1, r2, recip_distance in close_recipient_pairs:
                    # Create auxiliary variable for when both recipients are assigned to same volunteer
                    z = solver.BoolVar(f'z_compact_{v}_{r1}_{r2}')
                    solver.Add(z <= x[v, r1])
                    solver.Add(z <= x[v, r2])
                    solver.Add(z >= x[v, r1] + x[v, r2] - 1)
                    
                    # Normalize the distance score (1 when very close, 0 when far)
                    normalized_closeness = 1.0 - (recip_distance / recipient_distance_norm)
                    
                    # Use negative coefficient to give preference to assigning close recipients together
                    objective.SetCoefficient(z, -weights['compact_routes'] * normalized_closeness)
        
        # Cluster bonuses
        if self.use_clustering and weights['clusters']:
            for cluster_id, recipient_indices in self.clusters.items():
                if cluster_id != -1 and len(recipient_indices) > 1:
                    # Count the actual pairs within the distance threshold
                    valid_cluster_pairs = 0
                    
                    for v in range(self.num_volunteers):
                        for i in range(len(recipient_indices)):
                            for j in range(i + 1, len(recipient_indices)):
                                r1, r2 = recipient_indices[i], recipient_indices[j]
                                if r1 < 0 or r2 < 0 or r1 >= self.num_recipients or r2 >= self.num_recipients:
                                    continue
                                    
                                valid_cluster_pairs += 1
                                z = solver.BoolVar(f'z_{v}_{r1}_{r2}')
                                solver.Add(z <= x[v, r1])
                                solver.Add(z <= x[v, r2])
                                solver.Add(z >= x[v, r1] + x[v, r2] - 1)
                                objective.SetCoefficient(z, -weights['clusters'])
            
            print(f"Created {valid_cluster_pairs} cluster pair variables")
        
        objective.SetMinimization()
        
        status = solver.Solve()
        end_time = time.time()
        
        if status == pywraplp.Solver.OPTIMAL or status == pywraplp.Solver.FEASIBLE:
            print(f"Solution found in {end_time - start_time:.2f} seconds!")
            print(f"Objective value = {objective.Value()}")
            
            self.assignments = []
            self.assignment_map = {}
            self.pickup_assignments = {}  # volunteer_id -> pickup_id
            
            # Store the pickup assignment for each volunteer (from pre-assignment)
            for v in range(self.num_volunteers):
                volunteer_id = self.volunteers[v].volunteer_id
                self.assignment_map[volunteer_id] = []
                
                # Get the pre-assigned pickup for this volunteer
                pickup_idx = self.volunteer_pickup_assignments[v]
                pickup_id = self.pickups[pickup_idx].location_id
                self.pickup_assignments[volunteer_id] = pickup_id
            
            # Store the volunteer-recipient assignments
            for v in range(self.num_volunteers):
                volunteer_id = self.volunteers[v].volunteer_id
                for r in range(self.num_recipients):
                    if x[v, r].solution_value() > 0.5:
                        recipient_id = self.recipients[r].recipient_id
                        pickup_id = self.pickup_assignments.get(volunteer_id)
                        
                        # Store as (volunteer_id, recipient_id, pickup_id)
                        self.assignments.append((volunteer_id, recipient_id, pickup_id))
                        self.assignment_map[volunteer_id].append(recipient_id)
            
            return True
        else:
            print("No solution found.")
            return False  
    
    def save_assignments_to_db(self):
        """
        Save assignments to the database.
        
        Returns:
            bool: Whether assignments were successfully saved.
        """
        try:
            self.db_handler.bulk_save_assignments(self.assignments)
            return True
        except Exception as e:
            print(f"Error saving assignments: {e}")
            return False
    
    def export_assignments_to_csv(self, filename=None):
        """
        Export assignments to a CSV file, including pickup location information.
        
        Args:
            filename (str, optional): Name of the file to save.
            
        Returns:
            str: Path to the saved file or None if no assignments.
        """
        if not self.assignments:
            print("No assignments to export!")
            return None
        
        data = []
        for volunteer_id, recipient_id, pickup_id in self.assignments:
            volunteer = next(v for v in self.volunteers if v.volunteer_id == volunteer_id)
            recipient = next(r for r in self.recipients if r.recipient_id == recipient_id)
            pickup = next(p for p in self.pickups if p.location_id == pickup_id)
            
            # Calculate distances for each leg of the journey
            vol_to_pickup_dist = self._haversine_distance(
                volunteer.latitude, volunteer.longitude,
                pickup.latitude, pickup.longitude
            )
            
            pickup_to_recip_dist = self._haversine_distance(
                pickup.latitude, pickup.longitude,
                recipient.latitude, recipient.longitude
            )
            
            recip_to_vol_dist = self._haversine_distance(
                recipient.latitude, recipient.longitude,
                volunteer.latitude, volunteer.longitude
            )
            
            # Total route distance
            total_route_dist = vol_to_pickup_dist + pickup_to_recip_dist + recip_to_vol_dist
            
            data.append({
                'volunteer_id': volunteer_id,
                'volunteer_car_size': volunteer.car_size,
                'pickup_id': pickup_id,
                'pickup_lat': pickup.latitude,
                'pickup_lon': pickup.longitude,
                'recipient_id': recipient_id,
                'recipient_num_items': recipient.num_items,
                'vol_to_pickup_km': vol_to_pickup_dist,
                'pickup_to_recip_km': pickup_to_recip_dist,
                'recip_to_vol_km': recip_to_vol_dist,
                'total_route_km': total_route_dist
            })
        
        df = pd.DataFrame(data)
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"assignments_{timestamp}.csv"
        
        filepath = os.path.join(self.output_dir, filename)
        df.to_csv(filepath, index=False)
        print(f"Assignments exported to {filepath}")
        return filepath
    
    def visualize_assignments(self, save_path=None, show=True):
        """
        Visualize assignments using a Folium map, including pickup locations.
        Shows the complete route: volunteer -> pickup -> recipients -> volunteer.
        
        Args:
            save_path (str, optional): Path to save HTML file.
            show (bool): Whether to display the map.
            
        Returns:
            folium.Map or None: The map object if show=True, else None.
        """
        if not self.assignments:
            print("No assignments to visualize!")
            return None
        
        # Collect all coordinates for map centering
        all_lats = [v.latitude for v in self.volunteers] + \
                  [r.latitude for r in self.recipients] + \
                  [p.latitude for p in self.pickups]
        all_lons = [v.longitude for v in self.volunteers] + \
                  [r.longitude for r in self.recipients] + \
                  [p.longitude for p in self.pickups]
        center_lat = sum(all_lats) / len(all_lats)
        center_lon = sum(all_lons) / len(all_lons)
        
        m = folium.Map(location=[center_lat, center_lon], zoom_start=12)
        
        # Add pickup locations to the map first (they'll be shared by multiple volunteers)
        pickup_group = folium.FeatureGroup(name="Pickup Locations")
        for pickup in self.pickups:
            # Calculate how many boxes are needed from this pickup (sum of all assigned recipients for this pickup)
            boxes_needed = 0
            for (volunteer_id, recipient_ids) in self.assignment_map.items():
                if not recipient_ids:
                    continue
                volunteer_idx = next(i for i, v in enumerate(self.volunteers) if v.volunteer_id == volunteer_id)
                assigned_pickup_idx = self.volunteer_pickup_assignments.get(volunteer_idx)
                if assigned_pickup_idx is not None and self.pickups[assigned_pickup_idx].location_id == pickup.location_id:
                    # Sum all boxes for this volunteer's assigned recipients
                    for rid in recipient_ids:
                        recipient_idx = next(i for i, r in enumerate(self.recipients) if r.recipient_id == rid)
                        boxes_needed += self.recipients[recipient_idx].num_items
            pickup_popup = f"""
            <b>Pickup Location {pickup.location_id}</b><br>
            Available Items: {pickup.num_items}<br>
            <b>Boxes Needed: {boxes_needed}</b><br>
            Location: ({pickup.latitude:.4f}, {pickup.longitude:.4f})
            """
            
            folium.Marker(
                location=[pickup.latitude, pickup.longitude],
                popup=folium.Popup(pickup_popup, max_width=300),
                icon=folium.Icon(color='green', icon='shopping-cart', prefix='fa'),
                tooltip=f"Pickup {pickup.location_id}"
            ).add_to(pickup_group)
        
        pickup_group.add_to(m)
        
        # Create groups for each volunteer's routes
        volunteer_groups = {}
        
        # Process each volunteer's assignments
        for volunteer_id, recipient_ids in self.assignment_map.items():
            if not recipient_ids:
                continue
            
            # Get volunteer details
            volunteer_idx = next(i for i, v in enumerate(self.volunteers)
                                if v.volunteer_id == volunteer_id)
            volunteer = self.volunteers[volunteer_idx]
            
            # Get pickup location for this volunteer
            pickup_id = self.pickup_assignments.get(volunteer_id)
            if pickup_id is None:
                continue  # Skip if no pickup assigned
                
            pickup = next(p for p in self.pickups if p.location_id == pickup_id)
            
            # Create a feature group for this volunteer
            group = folium.FeatureGroup(name=f"Volunteer {volunteer_id}")
            volunteer_groups[volunteer_id] = group
            
            # Get recipient indices
            recipient_indices = [next(i for i, r in enumerate(self.recipients)
                                      if r.recipient_id == rid)
                                for rid in recipient_ids]
            
            # Calculate statistics
            total_boxes = sum(self.recipients[r_idx].num_items for r_idx in recipient_indices)
            utilization = total_boxes / volunteer.car_size * 100
            
            # Calculate route with pickup location
            travel_time, total_distance, route_coords = self._calculate_route_travel_time(
                volunteer_idx, recipient_indices, 
                pickup_idx=next(i for i, p in enumerate(self.pickups) if p.location_id == pickup_id)
            )
            
            # Create volunteer marker with detailed popup
            volunteer_popup = f"""
            <b>Volunteer {volunteer_id}</b><br>
            Car Capacity: {volunteer.car_size} boxes<br>
            Assigned: {total_boxes} boxes ({utilization:.1f}%)<br>
            Pickup Location: {pickup_id}<br>
            Recipients: {len(recipient_ids)}<br>
            Est. Travel: {travel_time:.1f} min ({total_distance:.1f} km)
            """
            
            folium.Marker(
                location=[volunteer.latitude, volunteer.longitude],
                popup=folium.Popup(volunteer_popup, max_width=300),
                icon=folium.Icon(color='blue', icon='user'),
                tooltip=f"Volunteer {volunteer_id}"
            ).add_to(group)
            
            # Draw route: Volunteer -> Pickup (blue line)
            folium.PolyLine(
                locations=[
                    [volunteer.latitude, volunteer.longitude],
                    [pickup.latitude, pickup.longitude]
                ],
                color='blue',
                weight=3,
                opacity=0.8,
                tooltip=f"To Pickup: {self._haversine_distance(volunteer.latitude, volunteer.longitude, pickup.latitude, pickup.longitude):.1f} km"
            ).add_to(group)
            
            # Create markers for each recipient
            for r_idx in recipient_indices:
                recipient = self.recipients[r_idx]
                recipient_popup = f"""
                <b>Recipient {recipient.recipient_id}</b><br>
                Boxes: {recipient.num_items}<br>
                Assigned to: Volunteer {volunteer_id}<br>
                Pickup: {pickup_id}
                """
                
                folium.Marker(
                    location=[recipient.latitude, recipient.longitude],
                    popup=folium.Popup(recipient_popup, max_width=300),
                    icon=folium.Icon(color='red', icon='home'),
                    tooltip=f"Recipient {recipient.recipient_id}"
                ).add_to(group)
                
                # We'll draw the VRP routes at the end after processing all recipients
            
            # Now draw the optimized VRP-style route
            # First: Volunteer -> Pickup
            folium.PolyLine(
                locations=[
                    [volunteer.latitude, volunteer.longitude],
                    [pickup.latitude, pickup.longitude]
                ],
                color='blue',
                weight=3,
                opacity=0.8,
                tooltip=f"To Pickup: {self._haversine_distance(volunteer.latitude, volunteer.longitude, pickup.latitude, pickup.longitude):.1f} km"
            ).add_to(group)
            
            # Get the optimized route coordinates
            _, _, route_coords = self._calculate_route_travel_time(
                volunteer_idx, recipient_indices, 
                pickup_idx=next(i for i, p in enumerate(self.pickups) if p.location_id == pickup_id)
            )
            
            # Extract just the recipient coordinates (skip volunteer->pickup->first_recipient)
            # The route is: volunteer -> pickup -> recipients -> volunteer
            recipient_route = route_coords[1:-1]  # Skip first (volunteer) and last (return to volunteer)
            
            # Draw route segments with arrows for direction
            for i in range(len(recipient_route) - 1):
                start_lat, start_lon = recipient_route[i]
                end_lat, end_lon = recipient_route[i+1]
                
                # Calculate the distance for this segment
                segment_distance = self._haversine_distance(start_lat, start_lon, end_lat, end_lon)
                
                # Determine color based on segment type
                color = 'green'
                segment_type = "Pickup to Recipient" if i == 0 else "Recipient to Recipient"
                
                # Create the line
                line = folium.PolyLine(
                    locations=[
                        [start_lat, start_lon],
                        [end_lat, end_lon]
                    ],
                    color=color,
                    weight=3,
                    opacity=0.8,
                    tooltip=f"{segment_type}: {segment_distance:.1f} km"
                ).add_to(group)
                
                # Add arrow marker at midpoint
                midpoint_lat = (start_lat + end_lat) / 2
                midpoint_lon = (start_lon + end_lon) / 2
                
                # Calculate bearing for arrow direction
                y = math.sin(end_lon - start_lon) * math.cos(end_lat)
                x = math.cos(start_lat) * math.sin(end_lat) - math.sin(start_lat) * math.cos(end_lat) * math.cos(end_lon - start_lon)
                bearing = math.atan2(y, x)
                bearing = math.degrees(bearing)
                bearing = (bearing + 360) % 360
                
                # Add arrow icon
                arrow_icon = folium.features.DivIcon(
                    icon_size=(20, 20),
                    icon_anchor=(10, 10),
                    html=f'<div style="font-size: 12pt; color: {color}; transform: rotate({bearing}deg);">➤</div>'
                )
                
                folium.Marker(
                    [midpoint_lat, midpoint_lon],
                    icon=arrow_icon
                ).add_to(group)
            
            # Draw the final return route: Last recipient -> Volunteer
            last_recipient_lat, last_recipient_lon = recipient_route[-1]
            
            folium.PolyLine(
                locations=[
                    [last_recipient_lat, last_recipient_lon],
                    [volunteer.latitude, volunteer.longitude]
                ],
                color='purple',
                weight=2,
                opacity=0.6,
                tooltip=f"Return Trip: {self._haversine_distance(last_recipient_lat, last_recipient_lon, volunteer.latitude, volunteer.longitude):.1f} km"
            ).add_to(group)
            
            group.add_to(m)
        
        # --- STATISTICS PANEL & HIDE BUTTON ---
        # Compute statistics
        total_volunteers = len([vid for vid, rids in self.assignment_map.items() if rids])
        total_recipients = sum(len(rids) for rids in self.assignment_map.values())
        route_lengths = []
        total_distance = 0
        utilizations = []
        for volunteer_id, recipient_ids in self.assignment_map.items():
            if not recipient_ids:
                continue
            volunteer_idx = next(i for i, v in enumerate(self.volunteers) if v.volunteer_id == volunteer_id)
            recipient_indices = [next(i for i, r in enumerate(self.recipients) if r.recipient_id == rid) for rid in recipient_ids]
            pickup_id = self.pickup_assignments.get(volunteer_id)
            if pickup_id is None:
                continue
            pickup_idx = next(i for i, p in enumerate(self.pickups) if p.location_id == pickup_id)
            _, route_dist, _ = self._calculate_route_travel_time(volunteer_idx, recipient_indices, pickup_idx=pickup_idx)
            route_lengths.append(route_dist)
            total_distance += route_dist
            total_boxes = sum(self.recipients[r_idx].num_items for r_idx in recipient_indices)
            volunteer = self.volunteers[volunteer_idx]
            utilizations.append(total_boxes / volunteer.car_size * 100)
        avg_route_length = sum(route_lengths) / len(route_lengths) if route_lengths else 0
        avg_utilization = sum(utilizations) / len(utilizations) if utilizations else 0
        
        stats_html = f'''
        <div id="stats-panel" style="position: fixed; bottom: 50px; right: 50px; z-index: 1001; background-color: white; padding: 14px; border: 2px solid #666; border-radius: 8px; display: none; min-width: 250px;">
            <h4 style="margin-top:0">Assignment Statistics</h4>
            <ul style="padding-left: 1.2em;">
                <li><b>Total Volunteers:</b> {total_volunteers}</li>
                <li><b>Total Recipients:</b> {total_recipients}</li>
                <li><b>Avg. Route Length:</b> {avg_route_length:.2f} km</li>
                <li><b>Total Distance:</b> {total_distance:.2f} km</li>
                <li><b>Avg. Utilization:</b> {avg_utilization:.1f}%</li>
            </ul>
            <button onclick="document.getElementById('stats-panel').style.display='none'" style="margin-top:8px;">Close</button>
        </div>
        <button id="toggle-stats-btn" style="position: fixed; bottom: 10px; right: 10px; z-index: 1002; background-color: #4CAF50; color: white; border: none; padding: 8px 16px; border-radius: 5px; cursor: pointer;">Show Statistics</button>
        '''
        
        # JavaScript for toggling stats and adding Toggle All Volunteers to LayerControl
        js_code = '''
        <script>
        // Toggle statistics panel
        document.getElementById('toggle-stats-btn').onclick = function() {
            var panel = document.getElementById('stats-panel');
            panel.style.display = panel.style.display === 'none' ? 'block' : 'none';
        };
        // Wait for LayerControl to be available, then add Toggle All Volunteers
        function addToggleAllVolunteers() {
            var layerControl = document.getElementsByClassName('leaflet-control-layers-overlays')[0];
            if (!layerControl) { setTimeout(addToggleAllVolunteers, 500); return; }
            // Only add if not already present
            if (document.getElementById('toggle-all-volunteers-row')) return;
            var row = document.createElement('label');
            row.id = 'toggle-all-volunteers-row';
            row.style.display = 'block';
            row.style.cursor = 'pointer';
            row.style.marginTop = '8px';
            var cb = document.createElement('input');
            cb.type = 'checkbox';
            cb.checked = true;
            cb.style.marginRight = '6px';
            row.appendChild(cb);
            var text = document.createTextNode('Show All Volunteers');
            row.appendChild(text);
            layerControl.insertBefore(row, layerControl.firstChild);
            cb.onchange = function() {
                var allInputs = layerControl.querySelectorAll('input[type=checkbox]');
                for (var i = 0; i < allInputs.length; i++) {
                    var label = allInputs[i].parentElement;
                    if (label.textContent.trim().startsWith('Volunteer')) {
                        if (allInputs[i].checked !== cb.checked) {
                            allInputs[i].click();
                        }
                    }
                }
            };
        }
        setTimeout(addToggleAllVolunteers, 1000);
        </script>
        '''
        
        # Add legend
        legend_html = '''
        <div style="position: fixed; bottom: 50px; left: 50px; z-index: 1000; background-color: white; padding: 10px; border: 2px solid grey; border-radius: 5px;">
            <h4>Route Legend</h4>
            <div><i style="background: blue; width: 15px; height: 3px; display: inline-block;"></i> Volunteer to Pickup</div>
            <div><i style="background: green; width: 15px; height: 3px; display: inline-block;"></i> Optimized Delivery Route ➤</div>
            <div><i style="background: purple; width: 15px; height: 3px; display: inline-block;"></i> Return Trip</div>
            <div style="margin-top: 5px;">
                <i style="background: blue; width: 10px; height: 10px; border-radius: 50%; display: inline-block;"></i> Volunteer
                <i style="background: green; width: 10px; height: 10px; border-radius: 50%; display: inline-block; margin-left: 5px;"></i> Pickup
                <i style="background: red; width: 10px; height: 10px; border-radius: 50%; display: inline-block; margin-left: 5px;"></i> Recipient
            </div>
        </div>
        '''
        m.get_root().html.add_child(folium.Element(stats_html))
        m.get_root().html.add_child(folium.Element(js_code))
        m.get_root().html.add_child(folium.Element(legend_html))
        
        # Add layer control
        folium.LayerControl().add_to(m)
        
        if save_path:
            m.save(save_path)
            print(f"Map saved to {save_path}")
        
        return m if show else None
    
    def visualize_volunteer_load(self, save_path=None, show=True):
        """
        Visualize load distribution across volunteers.
        
        Args:
            save_path (str, optional): Path to save the visualization.
            show (bool): Whether to display the plot.
            
        Returns:
            bool: Whether visualization was successful.
        """
        if not self.assignments:
            print("No assignments to visualize!")
            return False
        
        plt.figure(figsize=(12, 6))
        volunteer_ids = []
        capacities = []
        loads = []
        utilizations = []
        
        for volunteer_id, recipient_ids in self.assignment_map.items():
            if not recipient_ids:
                continue
            
            volunteer_idx = next(i for i, v in enumerate(self.volunteers)
                                if v.volunteer_id == volunteer_id)
            volunteer = self.volunteers[volunteer_idx]
            
            total_boxes = sum(self.recipients[next(i for i, r in enumerate(self.recipients)
                                                  if r.recipient_id == rid)].num_items
                             for rid in recipient_ids)
            utilization = total_boxes / volunteer.car_size * 100
            
            volunteer_ids.append(str(volunteer_id))
            capacities.append(volunteer.car_size)
            loads.append(total_boxes)
            utilizations.append(utilization)
        
        x = range(len(volunteer_ids))
        width = 0.35
        plt.bar(x, capacities, width, label='Capacity', color='lightblue')
        plt.bar([i + width for i in x], loads, width, label='Assigned', color='orange')
        
        ax1 = plt.gca()
        ax2 = ax1.twinx()
        ax2.plot([i + width/2 for i in x], utilizations, 'ro-', label='Utilization (%)')
        ax2.set_ylabel('Utilization (%)')
        ax2.set_ylim(0, max(utilizations) * 1.2 if utilizations else 100)
        
        plt.xlabel('Volunteer ID')
        ax1.set_ylabel('Number of Boxes')
        plt.title('Volunteer Load Distribution')
        ax1.set_xticks([i + width/2 for i in x])
        ax1.set_xticklabels(volunteer_ids)
        
        lines1, labels1 = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper left')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path)
            print(f"Load distribution saved to {save_path}")
        
        if show:
            plt.show()
        else:
            plt.close()
        
        return True
    
    def generate_assignment_report(self, output_format='markdown'):
        """
        Generate a report of all assignments.
        
        Args:
            output_format (str): Format of the report ('markdown', 'html', 'text').
            
        Returns:
            str: Formatted report.
        """
        if not self.assignments:
            return "No assignments to report!"
        
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        total_volunteers = len([vid for vid, rids in self.assignment_map.items() if rids])
        total_recipients = len(self.assignments)
        total_boxes = sum(self.recipients[next(i for i, r in enumerate(self.recipients)
                                             if r.recipient_id == rid)].num_items
                          for _, rid, _ in self.assignments)
        total_capacity = sum(self.volunteers[next(i for i, v in enumerate(self.volunteers)
                                              if v.volunteer_id == vid)].car_size
                            for vid, rids in self.assignment_map.items() if rids)
        overall_utilization = total_boxes / total_capacity * 100 if total_capacity > 0 else 0
        
        total_distance = 0.0
        total_utilization = 0.0
        volunteer_used_count = 0
        
        for volunteer_id, recipient_ids in self.assignment_map.items():
            if not recipient_ids:
                continue
            volunteer_idx = next(i for i, v in enumerate(self.volunteers)
                                if v.volunteer_id == volunteer_id)
            recipient_indices = [next(i for i, r in enumerate(self.recipients)
                                     if r.recipient_id == rid)
                                for rid in recipient_ids]
            # Get pickup index for this volunteer
            pickup_idx = self.volunteer_pickup_assignments.get(volunteer_idx)
            _, distance, _ = self._calculate_route_travel_time(volunteer_idx, recipient_indices, pickup_idx)
            total_distance += distance
        
        avg_distance = total_distance / total_volunteers if total_volunteers > 0 else 0
        
        if output_format == 'markdown':
            report = f"# Volunteer Assignment Report\n\nGenerated on: {timestamp}\n\n"
            report += f"## Summary\n\n"
            report += f"- **Total Volunteers:** {total_volunteers}\n"
            report += f"- **Total Recipients:** {total_recipients}\n"
            report += f"- **Total Boxes:** {total_boxes}\n"
            report += f"- **Overall Utilization:** {overall_utilization:.1f}%\n"
            report += f"\n## Assignments\n\n"
            
            for volunteer_id, recipient_ids in sorted(self.assignment_map.items()):
                if not recipient_ids:
                    continue
                volunteer_idx = next(i for i, v in enumerate(self.volunteers)
                                    if v.volunteer_id == volunteer_id)
                volunteer = self.volunteers[volunteer_idx]
                total_boxes = sum(self.recipients[next(i for i, r in enumerate(self.recipients)
                                                    if r.recipient_id == rid)].num_items
                                 for rid in recipient_ids)
                utilization = total_boxes / volunteer.car_size * 100
                recipient_indices = [next(i for i, r in enumerate(self.recipients)
                                         if r.recipient_id == rid)
                                    for rid in recipient_ids]
                # Get pickup information
                pickup_id = self.pickup_assignments.get(volunteer_id)
                pickup = next(p for p in self.pickups if p.location_id == pickup_id)
                
                # Calculate volunteer to pickup distance
                vol_to_pickup_dist = self._haversine_distance(
                    volunteer.latitude, volunteer.longitude,
                    pickup.latitude, pickup.longitude
                )
                
                # Get pickup index for this volunteer
                pickup_idx = self.volunteer_pickup_assignments.get(volunteer_idx)
                travel_time, distance, _ = self._calculate_route_travel_time(
                    volunteer_idx, recipient_indices, pickup_idx)
                
                report += f"### Volunteer {volunteer_id}\n\n"
                report += f"- **Car Capacity:** {volunteer.car_size} boxes\n"
                report += f"- **Assigned:** {total_boxes} boxes ({utilization:.1f}%)\n"
                report += f"- **Pickup Location:** {pickup_id}\n"
                report += f"- **Distance to Pickup:** {vol_to_pickup_dist:.2f} km\n"
                report += f"- **Recipients:** {len(recipient_ids)}\n"
                report += f"- **Est. Travel:** {travel_time:.1f} min ({distance:.1f} km)\n\n"
                
                # Add recipient table with pickup route information
                report += "| Recipient ID | Boxes | Pickup to Recipient (km) | Return Trip (km) |\n"
                report += "|-------------|-------|------------------------|----------------|\n"
                
                for rid in recipient_ids:
                    recipient_idx = next(i for i, r in enumerate(self.recipients)
                                         if r.recipient_id == rid)
                    recipient = self.recipients[recipient_idx]
                    
                    # Calculate pickup to recipient distance
                    pickup_to_recip_dist = self._haversine_distance(
                        pickup.latitude, pickup.longitude,
                        recipient.latitude, recipient.longitude
                    )
                    
                    # Calculate recipient to volunteer distance (return trip)
                    recip_to_vol_dist = self._haversine_distance(
                        recipient.latitude, recipient.longitude,
                        volunteer.latitude, volunteer.longitude
                    )
                    
                    report += f"| {recipient.recipient_id} | {recipient.num_items} | {pickup_to_recip_dist:.2f} | {recip_to_vol_dist:.2f} |\n"
                
                report += "\n"
                
                # Update statistics
                total_utilization += utilization
                volunteer_used_count += 1
            
            avg_utilization = total_utilization / volunteer_used_count if volunteer_used_count > 0 else 0
            report += f"## Statistics\n\n"
            report += f"- **Average Distance:** {avg_distance:.2f} km\n"
            report += f"- **Average Utilization:** {avg_utilization:.1f}%\n"
        
        elif output_format == 'html':
            report = "<html><body><h1>Volunteer Assignment Report</h1>"
            report += f"<p>Generated on: {timestamp}</p>"
            report += "<h2>Summary</h2>"
            report += f"<ul><li>Total Volunteers: {total_volunteers}</li>"
            report += f"<li>Total Recipients: {total_recipients}</li>"
            report += f"<li>Total Boxes: {total_boxes}</li>"
            report += f"<li>Overall Utilization: {overall_utilization:.1f}%</li></ul>"
            report += "<h2>Assignments</h2>"
            for volunteer_id, recipient_ids in sorted(self.assignment_map.items()):
                if not recipient_ids:
                    continue
                volunteer_idx = next(i for i, v in enumerate(self.volunteers)
                                    if v.volunteer_id == volunteer_id)
                volunteer = self.volunteers[volunteer_idx]
                total_boxes = sum(self.recipients[next(i for i, r in enumerate(self.recipients)
                                                    if r.recipient_id == rid)].num_items
                                 for rid in recipient_ids)
                utilization = total_boxes / volunteer.car_size * 100
                recipient_indices = [next(i for i, r in enumerate(self.recipients)
                                         if r.recipient_id == rid)
                                    for rid in recipient_ids]
                travel_time, distance = self._calculate_route_travel_time(
                    volunteer_idx, recipient_indices)
                report += f"<h3>Volunteer {volunteer_id}</h3>"
                report += f"<ul><li>Car Capacity: {volunteer.car_size} boxes</li>"
                report += f"<li>Assigned: {total_boxes} boxes ({utilization:.1f}%)</li>"
                report += f"<li>Recipients: {len(recipient_ids)}</li>"
                report += f"<li>Est. Travel: {travel_time:.1f} min ({distance:.1f} km)</li></ul>"
                report += "<table border='1'><tr><th>Recipient ID</th><th>Boxes</th><th>Distance (km)</th></tr>"
                for rid in recipient_ids:
                    recipient_idx = next(i for i, r in enumerate(self.recipients)
                                        if r.recipient_id == rid)
                    recipient = self.recipients[recipient_idx]
                    distance = self._haversine_distance(
                        volunteer.latitude, volunteer.longitude,
                        recipient.latitude, recipient.longitude
                    )
                    report += f"<tr><td>{recipient.recipient_id}</td><td>{recipient.num_items}</td><td>{distance:.2f}</td></tr>"
                report += "</table>"
            report += "<h2>Statistics</h2>"
            report += f"<ul><li>Average Distance: {avg_distance:.2f} km</li>"
            report += f"<li>Average Utilization: {avg_utilization:.1f}%</li></ul>"
            report += "</body></html>"
        
        else:  # text
            report = f"Volunteer Assignment Report\nGenerated on: {timestamp}\n\n"
            report += f"Summary\n"
            report += f"- Total Volunteers: {total_volunteers}\n"
            report += f"- Total Recipients: {total_recipients}\n"
            report += f"- Total Boxes: {total_boxes}\n"
            report += f"- Overall Utilization: {overall_utilization:.1f}%\n\n"
            report += "Assignments\n"
            for volunteer_id, recipient_ids in sorted(self.assignment_map.items()):
                if not recipient_ids:
                    continue
                volunteer_idx = next(i for i, v in enumerate(self.volunteers)
                                    if v.volunteer_id == volunteer_id)
                volunteer = self.volunteers[volunteer_idx]
                total_boxes = sum(self.recipients[next(i for i, r in enumerate(self.recipients)
                                                    if r.recipient_id == rid)].num_items
                                 for rid in recipient_ids)
                utilization = total_boxes / volunteer.car_size * 100
                recipient_indices = [next(i for i, r in enumerate(self.recipients)
                                         if r.recipient_id == rid)
                                    for rid in recipient_ids]
                travel_time, distance = self._calculate_route_travel_time(
                    volunteer_idx, recipient_indices)
                report += f"\nVolunteer {volunteer_id}\n"
                report += f"- Car Capacity: {volunteer.car_size} boxes\n"
                report += f"- Assigned: {total_boxes} boxes ({utilization:.1f}%)\n"
                report += f"- Recipients: {len(recipient_ids)}\n"
                report += f"- Est. Travel: {travel_time:.1f} min ({distance:.1f} km)\n"
                report += "Recipient ID | Boxes | Distance (km)\n"
                report += "-------------|-------|-------------\n"
                for rid in recipient_ids:
                    recipient_idx = next(i for i, r in enumerate(self.recipients)
                                        if r.recipient_id == rid)
                    recipient = self.recipients[recipient_idx]
                    distance = self._haversine_distance(
                        volunteer.latitude, volunteer.longitude,
                        recipient.latitude, recipient.longitude
                    )
                    report += f"{recipient.recipient_id} | {recipient.num_items} | {distance:.2f}\n"
            report += f"\nStatistics\n"
            report += f"- Average Distance: {avg_distance:.2f} km\n"
            report += f"- Average Utilization: {avg_utilization:.1f}%\n"
        
        return report
    
    def save_report(self, report, filename=None):
        """
        Save a report to a file.
        
        Args:
            report (str): Report content.
            filename (str, optional): Name of the file to save.
            
        Returns:
            str: Path to the saved file.
        """
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"report_{timestamp}.md"
        
        filepath = os.path.join(self.output_dir, filename)
        with open(filepath, 'w') as f:
            f.write(report)
        
        print(f"Report saved to {filepath}")
        return filepath
    
    def run_complete_pipeline(self, export_csv=True, save_visualizations=True, save_report=True):
        """
        Run the complete assignment pipeline.
        
        Args:
            export_csv (bool): Whether to export assignments to CSV.
            save_visualizations (bool): Whether to save visualizations.
            save_report (bool): Whether to save the report.
            
        Returns:
            bool: Whether the pipeline was successful.
        """
        success = self.generate_assignments()
        if not success:
            return False
        
        self.save_assignments_to_db()
        
        if export_csv:
            self.export_assignments_to_csv()
        
        if save_visualizations:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            map_path = os.path.join(self.output_dir, f"assignment_map_{timestamp}.html")
            self.visualize_assignments(save_path=map_path, show=False)
            load_path = os.path.join(self.output_dir, f"load_distribution_{timestamp}.png")
            self.visualize_volunteer_load(save_path=load_path, show=False)
        
        if save_report:
            report = self.generate_assignment_report()
            self.save_report(report)
        
        print("Assignment pipeline completed successfully!")
        return True