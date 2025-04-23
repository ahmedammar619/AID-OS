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
        Create a distance matrix between volunteers and recipients.
        
        Returns:
            numpy.ndarray: Matrix of distances [volunteer_idx][recipient_idx].
        """
        distance_matrix = np.zeros((self.num_volunteers, self.num_recipients))
        for v_idx in range(self.num_volunteers):
            v_lat, v_lon = self.volunteers[v_idx].latitude, self.volunteers[v_idx].longitude
            for r_idx in range(self.num_recipients):
                r_lat, r_lon = self.recipients[r_idx].latitude, self.recipients[r_idx].longitude
                distance_matrix[v_idx, r_idx] = self._haversine_distance(v_lat, v_lon, r_lat, r_lon)
        return distance_matrix
    
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
    
    def _calculate_route_travel_time(self, volunteer_idx, recipient_indices):
        """
        Calculate estimated travel time and distance for a volunteer's route.
        Uses greedy nearest-neighbor algorithm, assumes 30 km/h speed and 5 min/stop.
        
        Args:
            volunteer_idx (int): Index of the volunteer.
            recipient_indices (list): List of recipient indices.
            
        Returns:
            float: Travel time in minutes.
            float: Total distance in kilometers.
        """
        if not recipient_indices:
            return 0.0, 0.0
        
        v_lat, v_lon = self.volunteers[volunteer_idx].latitude, self.volunteers[volunteer_idx].longitude
        recipient_coords = [(self.recipients[r_idx].latitude, self.recipients[r_idx].longitude)
                           for r_idx in recipient_indices]
        
        current_lat, current_lon = v_lat, v_lon
        unvisited = recipient_coords.copy()
        total_distance = 0.0
        
        while unvisited:
            distances = [self._haversine_distance(current_lat, current_lon, r_lat, r_lon)
                        for r_lat, r_lon in unvisited]
            nearest_idx = distances.index(min(distances))
            next_lat, next_lon = unvisited[nearest_idx]
            total_distance += distances[nearest_idx]
            current_lat, current_lon = next_lat, next_lon
            unvisited.pop(nearest_idx)
        
        total_distance += self._haversine_distance(current_lat, current_lon, v_lat, v_lon)
        travel_time = (total_distance / 30.0) * 60.0 + len(recipient_indices) * 5.0
        return travel_time, total_distance
    
    def generate_assignments(self):
        """
        Solve the volunteer-recipient assignment problem using OR-Tools MILP.
        
        Returns:
            bool: Whether the solution was successful.
        """
        print("Solving assignment problem using optimization...")
        start_time = time.time()
        
        solver = pywraplp.Solver.CreateSolver('SCIP')
        if not solver:
            print("Could not create solver!")
            return False
        
        # Decision variables
        x = {(v, r): solver.BoolVar(f'x_{v}_{r}')
             for v in range(self.num_volunteers) for r in range(self.num_recipients)}
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
        
        # Minimize total distance
        distance_weight = 10000.0
        for v in range(self.num_volunteers):
            for r in range(self.num_recipients):
                objective.SetCoefficient(x[v, r], distance_weight * self.distance_matrix[v, r])
        
        # Minimize number of volunteers
        volunteer_weight = -5.0
        for v in range(self.num_volunteers):
            objective.SetCoefficient(y[v], volunteer_weight)
        
        # Historical match bonuses
        history_weight = 1.0
        for v in range(self.num_volunteers):
            for r in range(self.num_recipients):
                historical_score = self._get_historical_match_score(v, r)
                if historical_score > 0:
                    objective.SetCoefficient(x[v, r], history_weight * historical_score)
        
        # Maximize volunteer capacity utilization
        capacity_util_weight = 30  # Negative because we want to maximize utilization
        for v in range(self.num_volunteers):
            # Create a capacity utilization term that rewards higher usage of available capacity
            used_capacity = sum(x[v, r] * self.recipients[r].num_items for r in range(self.num_recipients))
            # We can't directly add used_capacity/volunteer_capacity as a term, so we add each assignment
            # with a coefficient proportional to the recipient's items and volunteer's capacity
            for r in range(self.num_recipients):
                # Weight each assignment by its contribution to capacity utilization
                contribution = self.recipients[r].num_items / self.volunteers[v].car_size
                objective.SetCoefficient(x[v, r], capacity_util_weight * contribution)
        
        # Minimize recipient distances from each other (compact routes)
        recipient_distance_weight = 25.0
        
        # Pre-compute distances and filter only close pairs to significantly reduce problem size
        close_recipient_pairs = []
        for r1 in range(self.num_recipients):
            for r2 in range(r1 + 1, self.num_recipients):
                # Calculate distance between recipients
                r1_lat, r1_lon = self.recipients[r1].latitude, self.recipients[r1].longitude
                r2_lat, r2_lon = self.recipients[r2].latitude, self.recipients[r2].longitude
                recip_distance = self._haversine_distance(r1_lat, r1_lon, r2_lat, r2_lon)
                
                # Only consider very close recipients
                if recip_distance <= 5.0:  # Reduce threshold to 5 km
                    close_recipient_pairs.append((r1, r2, recip_distance))
        
        # Limit the number of pairs to consider (take the closest N pairs)
        max_pairs = min(500, len(close_recipient_pairs))  # Cap at 500 pairs
        close_recipient_pairs.sort(key=lambda x: x[2])  # Sort by distance
        close_recipient_pairs = close_recipient_pairs[:max_pairs]  # Take only the closest pairs
        
        # Only create variables for the filtered pairs
        for v in range(min(10, self.num_volunteers)):  # Limit to top 10 volunteers
            for r1, r2, recip_distance in close_recipient_pairs:
                # Create auxiliary variable for when both recipients are assigned to same volunteer
                z = solver.BoolVar(f'z_compact_{v}_{r1}_{r2}')
                solver.Add(z <= x[v, r1])
                solver.Add(z <= x[v, r2])
                solver.Add(z >= x[v, r1] + x[v, r2] - 1)
                
                # Use negative coefficient to give preference to assigning close recipients together
                # Scale inversely with distance so closer recipients get higher preference
                objective.SetCoefficient(z, -recipient_distance_weight / (1.0 + recip_distance))
        
        # Cluster bonuses
        if self.use_clustering:
            cluster_weight = -10.0
            for cluster_id, recipient_indices in self.clusters.items():
                if cluster_id != -1 and len(recipient_indices) > 1:
                    for v in range(self.num_volunteers):
                        for i in range(len(recipient_indices)):
                            for j in range(i + 1, len(recipient_indices)):
                                r1, r2 = recipient_indices[i], recipient_indices[j]
                                if r1 < 0 or r2 < 0 or r1 >= self.num_recipients or r2 >= self.num_recipients:
                                    continue
                                z = solver.BoolVar(f'z_{v}_{r1}_{r2}')
                                solver.Add(z <= x[v, r1])
                                solver.Add(z <= x[v, r2])
                                solver.Add(z >= x[v, r1] + x[v, r2] - 1)
                                objective.SetCoefficient(z, cluster_weight)
        
        objective.SetMinimization()
        
        status = solver.Solve()
        end_time = time.time()
        
        if status == pywraplp.Solver.OPTIMAL or status == pywraplp.Solver.FEASIBLE:
            print(f"Solution found in {end_time - start_time:.2f} seconds!")
            print(f"Objective value = {objective.Value()}")
            
            self.assignments = []
            self.assignment_map = {}
            
            for v in range(self.num_volunteers):
                volunteer_id = self.volunteers[v].volunteer_id
                self.assignment_map[volunteer_id] = []
                for r in range(self.num_recipients):
                    if x[v, r].solution_value() > 0.5:
                        recipient_id = self.recipients[r].recipient_id
                        self.assignments.append((volunteer_id, recipient_id))
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
        Export assignments to a CSV file.
        
        Args:
            filename (str, optional): Name of the file to save.
            
        Returns:
            str: Path to the saved file or None if no assignments.
        """
        if not self.assignments:
            print("No assignments to export!")
            return None
        
        data = []
        for volunteer_id, recipient_id in self.assignments:
            volunteer = next(v for v in self.volunteers if v.volunteer_id == volunteer_id)
            recipient = next(r for r in self.recipients if r.recipient_id == recipient_id)
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
        Visualize assignments using a Folium map.
        
        Args:
            save_path (str, optional): Path to save HTML file.
            show (bool): Whether to display the map.
            
        Returns:
            folium.Map or None: The map object if show=True, else None.
        """
        if not self.assignments:
            print("No assignments to visualize!")
            return None
        
        all_lats = [v.latitude for v in self.volunteers] + [r.latitude for r in self.recipients]
        all_lons = [v.longitude for v in self.volunteers] + [r.longitude for r in self.recipients]
        center_lat = sum(all_lats) / len(all_lats)
        center_lon = sum(all_lons) / len(all_lons)
        
        m = folium.Map(location=[center_lat, center_lon], zoom_start=12)
        volunteer_groups = {}
        
        for volunteer_id, recipient_ids in self.assignment_map.items():
            if not recipient_ids:
                continue
            
            volunteer_idx = next(i for i, v in enumerate(self.volunteers)
                                if v.volunteer_id == volunteer_id)
            volunteer = self.volunteers[volunteer_idx]
            
            group = folium.FeatureGroup(name=f"Volunteer {volunteer_id}")
            volunteer_groups[volunteer_id] = group
            
            recipient_indices = [next(i for i, r in enumerate(self.recipients)
                                     if r.recipient_id == rid)
                                for rid in recipient_ids]
            
            total_boxes = sum(self.recipients[r_idx].num_items for r_idx in recipient_indices)
            utilization = total_boxes / volunteer.car_size * 100
            travel_time, total_distance = self._calculate_route_travel_time(
                volunteer_idx, recipient_indices)
            
            volunteer_popup = f"""
            <b>Volunteer {volunteer_id}</b><br>
            Car Capacity: {volunteer.car_size} boxes<br>
            Assigned: {total_boxes} boxes ({utilization:.1f}%)<br>
            Recipients: {len(recipient_ids)}<br>
            Est. Travel: {travel_time:.1f} min ({total_distance:.1f} km)
            """
            
            folium.Marker(
                location=[volunteer.latitude, volunteer.longitude],
                popup=folium.Popup(volunteer_popup, max_width=300),
                icon=folium.Icon(color='blue', icon='user'),
                tooltip=f"Volunteer {volunteer_id}"
            ).add_to(group)
            
            for r_idx in recipient_indices:
                recipient = self.recipients[r_idx]
                recipient_popup = f"""
                <b>Recipient {recipient.recipient_id}</b><br>
                Boxes: {recipient.num_items}<br>
                Assigned to: Volunteer {volunteer_id}
                """
                
                folium.Marker(
                    location=[recipient.latitude, recipient.longitude],
                    popup=folium.Popup(recipient_popup, max_width=300),
                    icon=folium.Icon(color='red', icon='home'),
                    tooltip=f"Recipient {recipient.recipient_id}"
                ).add_to(group)
                
                folium.PolyLine(
                    locations=[
                        [volunteer.latitude, volunteer.longitude],
                        [recipient.latitude, recipient.longitude]
                    ],
                    color='blue',
                    weight=2,
                    opacity=0.7
                ).add_to(group)
            
            group.add_to(m)
        
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
                         for _, rid in self.assignments)
        total_capacity = sum(self.volunteers[next(i for i, v in enumerate(self.volunteers)
                                              if v.volunteer_id == vid)].car_size
                            for vid, rids in self.assignment_map.items() if rids)
        overall_utilization = total_boxes / total_capacity * 100 if total_capacity > 0 else 0
        
        total_distance = 0.0
        for volunteer_id, recipient_ids in self.assignment_map.items():
            if not recipient_ids:
                continue
            volunteer_idx = next(i for i, v in enumerate(self.volunteers)
                                if v.volunteer_id == volunteer_id)
            recipient_indices = [next(i for i, r in enumerate(self.recipients)
                                     if r.recipient_id == rid)
                                for rid in recipient_ids]
            _, distance = self._calculate_route_travel_time(volunteer_idx, recipient_indices)
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
                travel_time, distance = self._calculate_route_travel_time(
                    volunteer_idx, recipient_indices)
                
                report += f"### Volunteer {volunteer_id}\n\n"
                report += f"- **Car Capacity:** {volunteer.car_size} boxes\n"
                report += f"- **Assigned:** {total_boxes} boxes ({utilization:.1f}%)\n"
                report += f"- **Recipients:** {len(recipient_ids)}\n"
                report += f"- **Est. Travel:** {travel_time:.1f} min ({distance:.1f} km)\n\n"
                report += "| Recipient ID | Boxes | Distance (km) |\n"
                report += "|-------------|-------|---------------|\n"
                for rid in recipient_ids:
                    recipient_idx = next(i for i, r in enumerate(self.recipients)
                                        if r.recipient_id == rid)
                    recipient = self.recipients[recipient_idx]
                    distance = self._haversine_distance(
                        volunteer.latitude, volunteer.longitude,
                        recipient.latitude, recipient.longitude
                    )
                    report += f"| {recipient.recipient_id} | {recipient.num_items} | {distance:.2f} |\n"
                report += "\n"
            
            total_utilization = 0.0
            volunteer_used_count = 0
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

