#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Assignment module for the AID-OS project.
Uses Google OR-Tools optimization to generate optimal volunteer-recipient assignments.
"""

import numpy as np
import pandas as pd
import os
import sys
import json
from datetime import datetime
import matplotlib.pyplot as plt
import folium
import seaborn as sns
import math

# Add parent directory to path to import from project modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from data.db_config import DatabaseHandler
from optimization.solver import OptimizationSolver
from clustering.dbscan_cluster import RecipientClusterer
from feedback.feedback_handler import FeedbackHandler


class VolunteerAssignerOpt:
    """
    Class for assigning volunteers to recipients using optimization.
    
    This class provides functionality to:
    1. Generate assignments using optimization
    2. Visualize assignments
    3. Save assignments to the database
    4. Generate reports
    """
    
    def __init__(
        self,
        db_handler=None,
        feedback_handler=None,
        use_clustering=True,
        cluster_eps=0.00005,
        output_dir="./output"
    ):
        """
        Initialize the volunteer assigner.
        
        Args:
            db_handler (DatabaseHandler): Database connection handler
            feedback_handler (FeedbackHandler): Feedback handler for admin input
            use_clustering (bool): Whether to use clustering for assignments
            output_dir (str): Directory to save output files
        """
        # Initialize handlers
        self.db_handler = db_handler if db_handler is not None else DatabaseHandler()
        self.feedback_handler = feedback_handler if feedback_handler is not None else FeedbackHandler()
        
        # Create output directory if it doesn't exist
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        
        # Create optimization solver
        self.solver = OptimizationSolver(
            db_handler=self.db_handler,
            use_clustering=use_clustering,
            cluster_eps=cluster_eps,
            output_dir=output_dir
        )
        
        # Load data
        self.volunteers = self.solver.volunteers
        self.recipients = self.solver.recipients
        self.distance_matrix = self.solver.distance_matrix
        
        # Assignments
        self.assignments = []
        self.assignment_map = {}  # volunteer_id -> [recipient_ids]
    
    def generate_assignments(self):
        """
        Generate assignments using optimization.
        
        Returns:
            bool: Whether assignments were successfully generated
        """
        # Solve the optimization problem
        success = self.solver.solve()
        
        if success:
            # Get assignments from solver
            self.assignments = self.solver.assignments
            self.assignment_map = self.solver.assignment_map
            
            print(f"Generated {len(self.assignments)} assignments using optimization")
            return True
        else:
            print("Failed to generate assignments")
            return False
    
    def save_assignments_to_db(self):
        """
        Save the generated assignments to the database.
        
        Returns:
            bool: Whether assignments were successfully saved
        """
        return self.solver.save_assignments_to_db()
    
    def export_assignments_to_csv(self, filename=None):
        """
        Export the generated assignments to a CSV file.
        
        Args:
            filename (str, optional): Name of the file to save to
            
        Returns:
            str: Path to the saved file
        """
        return self.solver.export_assignments_to_csv(filename)
    
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
    
    def visualize_assignments(self, save_path=None, show=True):
        """
        Visualize volunteer-recipient assignments using Leaflet
        
        Args:
            save_path (str): Optional path to save HTML file
            show (bool): Whether to display the visualization
        """
        if not self.assignments:
            print("No assignments to visualize!")
            return
        
        # Create a map centered at the average of all coordinates
        all_lats = [v.latitude for v in self.volunteers] + [r.latitude for r in self.recipients]
        all_lons = [v.longitude for v in self.volunteers] + [r.longitude for r in self.recipients]
        
        center_lat = sum(all_lats) / len(all_lats)
        center_lon = sum(all_lons) / len(all_lons)
        
        m = folium.Map(location=[center_lat, center_lon], zoom_start=12)
        
        # Create a feature group for each volunteer
        volunteer_groups = {}
        
        # Add volunteer markers and recipient markers with lines
        for volunteer_id, recipient_ids in self.assignment_map.items():
            if not recipient_ids:
                continue
                
            # Find volunteer index
            volunteer_idx = next(i for i, v in enumerate(self.volunteers) 
                                if v.volunteer_id == volunteer_id)
            volunteer = self.volunteers[volunteer_idx]
            
            # Create a feature group for this volunteer
            group = folium.FeatureGroup(name=f"Volunteer {volunteer_id}")
            volunteer_groups[volunteer_id] = group
            
            # Find recipient indices
            recipient_indices = [next(i for i, r in enumerate(self.recipients) 
                                     if r.recipient_id == rid) 
                                for rid in recipient_ids]
            
            # Calculate total boxes and utilization
            total_boxes = sum(self.recipients[r_idx].num_items for r_idx in recipient_indices)
            utilization = total_boxes / volunteer.car_size * 100
            
            # Calculate travel time and distance
            travel_time, total_distance = self._calculate_route_travel_time(
                volunteer_idx, recipient_indices)
            
            # Add volunteer marker
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
            
            # Add recipient markers and lines
            for r_idx in recipient_indices:
                recipient = self.recipients[r_idx]
                
                # Add recipient marker
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
                
                # Add line from volunteer to recipient
                folium.PolyLine(
                    locations=[
                        [volunteer.latitude, volunteer.longitude],
                        [recipient.latitude, recipient.longitude]
                    ],
                    color='blue',
                    weight=2,
                    opacity=0.7
                ).add_to(group)
            
            # Add the group to the map
            group.add_to(m)
        
        # Add layer control
        folium.LayerControl().add_to(m)
        
        # Save to file if requested
        if save_path:
            m.save(save_path)
            print(f"Map saved to {save_path}")
        
        # Show the map if requested
        if show:
            return m
    
    def visualize_volunteer_load(self, save_path=None, show=True):
        """
        Visualize the load distribution across volunteers.
        
        Args:
            save_path (str, optional): Path to save the visualization
            show (bool): Whether to display the plot
            
        Returns:
            bool: Whether visualization was successful
        """
        if not self.assignments:
            print("No assignments to visualize!")
            return False
        
        # Create figure
        plt.figure(figsize=(12, 6))
        
        # Prepare data
        volunteer_ids = []
        capacities = []
        loads = []
        utilizations = []
        
        for volunteer_id, recipient_ids in self.assignment_map.items():
            if not recipient_ids:
                continue
                
            # Find volunteer
            volunteer_idx = next(i for i, v in enumerate(self.volunteers) 
                                if v.volunteer_id == volunteer_id)
            volunteer = self.volunteers[volunteer_idx]
            
            # Calculate total boxes
            total_boxes = sum(self.recipients[next(i for i, r in enumerate(self.recipients) 
                                                if r.recipient_id == rid)].num_items 
                             for rid in recipient_ids)
            
            # Calculate utilization
            utilization = total_boxes / volunteer.car_size * 100
            
            # Add to lists
            volunteer_ids.append(str(volunteer_id))
            capacities.append(volunteer.car_size)
            loads.append(total_boxes)
            utilizations.append(utilization)
        
        # Create bar chart
        x = range(len(volunteer_ids))
        width = 0.35
        
        plt.bar(x, capacities, width, label='Capacity', color='lightblue')
        plt.bar([i + width for i in x], loads, width, label='Assigned', color='orange')
        
        # Add utilization line
        ax1 = plt.gca()
        ax2 = ax1.twinx()
        ax2.plot([i + width/2 for i in x], utilizations, 'ro-', label='Utilization (%)')
        ax2.set_ylabel('Utilization (%)')
        ax2.set_ylim(0, max(utilizations) * 1.2)
        
        # Add labels and legend
        plt.xlabel('Volunteer ID')
        ax1.set_ylabel('Number of Boxes')
        plt.title('Volunteer Load Distribution')
        ax1.set_xticks([i + width/2 for i in x])
        ax1.set_xticklabels(volunteer_ids)
        
        # Add legends
        lines1, labels1 = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper left')
        
        plt.tight_layout()
        
        # Save if requested
        if save_path:
            plt.savefig(save_path)
            print(f"Load distribution saved to {save_path}")
        
        # Show if requested
        if show:
            plt.show()
        else:
            plt.close()
        
        return True
    
    def generate_assignment_report(self, output_format='markdown'):
        """
        Generate a report of all assignments.
        
        Args:
            output_format (str): Format of the report ('markdown', 'html', 'text')
            
        Returns:
            str: Formatted report
        """
        if not self.assignments:
            return "No assignments to report!"
        
        # Generate timestamp
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        # Calculate statistics
        total_volunteers = len(self.assignment_map)
        total_recipients = len(self.assignments)
        total_boxes = sum(self.recipients[next(i for i, r in enumerate(self.recipients) 
                                            if r.recipient_id == recipient_id)].num_items 
                         for _, recipient_id in self.assignments)
        
        # Calculate utilization
        total_capacity = sum(self.volunteers[next(i for i, v in enumerate(self.volunteers) 
                                              if v.volunteer_id == volunteer_id)].car_size 
                            for volunteer_id in self.assignment_map.keys())
        overall_utilization = total_boxes / total_capacity * 100 if total_capacity > 0 else 0
        
        # Calculate average distance
        total_distance = 0.0
        for volunteer_id, recipient_ids in self.assignment_map.items():
            if not recipient_ids:
                continue
                
            # Find volunteer
            volunteer_idx = next(i for i, v in enumerate(self.volunteers) 
                                if v.volunteer_id == volunteer_id)
            
            # Find recipient indices
            recipient_indices = [next(i for i, r in enumerate(self.recipients) 
                                     if r.recipient_id == rid) 
                                for rid in recipient_ids]
            
            # Calculate travel time and distance
            _, distance = self._calculate_route_travel_time(volunteer_idx, recipient_indices)
            total_distance += distance
        
        avg_distance = total_distance / total_volunteers if total_volunteers > 0 else 0
        
        # Generate report based on format
        if output_format == 'markdown':
            report = f"# Volunteer Assignment Report\n\n"
            report += f"Generated on: {timestamp}\n\n"
            
            report += f"## Summary\n\n"
            report += f"- **Total Volunteers:** {total_volunteers}\n"
            report += f"- **Total Recipients:** {total_recipients}\n"
            report += f"- **Total Boxes:** {total_boxes}\n"
            report += f"- **Overall Utilization:** {overall_utilization:.1f}%\n"
            
            report += f"\n## Assignments\n\n"
            
            for volunteer_id, recipient_ids in sorted(self.assignment_map.items()):
                if not recipient_ids:
                    continue
                    
                # Find volunteer
                volunteer_idx = next(i for i, v in enumerate(self.volunteers) 
                                    if v.volunteer_id == volunteer_id)
                volunteer = self.volunteers[volunteer_idx]
                
                # Calculate total boxes and utilization
                total_boxes = sum(self.recipients[next(i for i, r in enumerate(self.recipients) 
                                                    if r.recipient_id == rid)].num_items 
                                 for rid in recipient_ids)
                utilization = total_boxes / volunteer.car_size * 100
                
                # Calculate travel time and distance
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
                    # Find recipient
                    recipient_idx = next(i for i, r in enumerate(self.recipients) 
                                        if r.recipient_id == rid)
                    recipient = self.recipients[recipient_idx]
                    
                    # Calculate distance
                    distance = self._haversine_distance(
                        volunteer.latitude, volunteer.longitude,
                        recipient.latitude, recipient.longitude
                    )
                    
                    report += f"| {recipient.recipient_id} | {recipient.num_items} | {distance:.2f} |\n"
                
                report += "\n"
            
            report += f"\n## Statistics\n\n"
            
            # Calculate average utilization
            total_utilization = 0.0
            volunteer_used_count = 0
            
            for volunteer_id, recipient_ids in self.assignment_map.items():
                if not recipient_ids:
                    continue
                    
                volunteer_idx = next(i for i, v in enumerate(self.volunteers) 
                                    if v.volunteer_id == volunteer_id)
                volunteer = self.volunteers[volunteer_idx]
                
                recipient_ids = self.assignment_map[volunteer_id]
                total_boxes = sum(self.recipients[next(i for i, r in enumerate(self.recipients) 
                                                      if r.recipient_id == rid)].num_items 
                                 for rid in recipient_ids)
                
                utilization = total_boxes / volunteer.car_size * 100
                total_utilization += utilization
                volunteer_used_count += 1
            
            avg_utilization = total_utilization / volunteer_used_count if volunteer_used_count > 0 else 0
            
            report += f"- **Average Distance:** {avg_distance:.2f} km\n"
            report += f"- **Average Utilization:** {avg_utilization:.1f}%\n"
        
        elif output_format == 'html':
            # Implement HTML report format
            report = "<html><body><h1>Volunteer Assignment Report</h1>"
            # ... HTML formatting ...
            report += "</body></html>"
        
        else:  # Plain text
            report = f"Volunteer Assignment Report\n"
            report += f"Generated on: {timestamp}\n\n"
            # ... plain text formatting ...
        
        return report
    
    def save_report(self, report, filename=None):
        """
        Save a report to a file.
        
        Args:
            report (str): Report content
            filename (str, optional): Name of the file to save to
            
        Returns:
            str: Path to the saved file
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
            export_csv (bool): Whether to export assignments to CSV
            save_visualizations (bool): Whether to save visualizations
            save_report (bool): Whether to save the report
            
        Returns:
            bool: Whether the pipeline was successful
        """
        # 1. Generate assignments
        success = self.generate_assignments()
        if not success:
            return False
        
        # 2. Save assignments to database
        self.save_assignments_to_db()
        
        # 3. Export to CSV if requested
        if export_csv:
            self.export_assignments_to_csv()
        
        # 4. Create and save visualizations if requested
        if save_visualizations:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            
            # Assignment map
            map_path = os.path.join(self.output_dir, f"assignment_map_{timestamp}.html")
            self.visualize_assignments(save_path=map_path, show=False)
            
            # Load distribution
            load_path = os.path.join(self.output_dir, f"load_distribution_{timestamp}.png")
            self.visualize_volunteer_load(save_path=load_path, show=False)
        
        # 5. Generate and save report if requested
        if save_report:
            report = self.generate_assignment_report()
            self.save_report(report)
        
        print("Assignment pipeline completed successfully!")
        return True


if __name__ == "__main__":
    # Test the assigner
    assigner = VolunteerAssignerOpt()
    success = assigner.generate_assignments()
    
    if success:
        # Export assignments
        assigner.export_assignments_to_csv()
        
        # Visualize assignments
        assigner.visualize_assignments()
        
        # Visualize load distribution
        assigner.visualize_volunteer_load()
