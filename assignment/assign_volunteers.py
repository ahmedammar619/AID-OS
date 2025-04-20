#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Assignment module for the AID-RL project.
Uses the trained RL agent to generate optimal volunteer-recipient assignments.
"""

import numpy as np
import pandas as pd
import os
import sys
import json
import torch
from datetime import datetime
import matplotlib.pyplot as plt
import folium
import seaborn as sns

# Add parent directory to path to import from project modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from data.db_config import DatabaseHandler
from models.rl_agent import ActorCriticAgent
from env.delivery_env import DeliveryEnv
from clustering.dbscan_cluster import RecipientClusterer
from feedback.feedback_handler import FeedbackHandler


class VolunteerAssigner:
    """
    Class for assigning volunteers to recipients using a trained RL agent.
    
    This class provides functionality to:
    1. Load a trained RL agent
    2. Generate assignments
    3. Visualize assignments
    4. Save assignments to the database
    5. Generate reports
    """
    
    def __init__(
        self,
        agent_path=None,
        db_handler=None,
        feedback_handler=None,
        use_clustering=True,
        cluster_eps=0.00005,
        max_steps=1000,
        output_dir="./output"
    ):
        """
        Initialize the volunteer assigner.
        
        Args:
            agent_path (str): Path to the trained agent checkpoint
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
        
        # Initialize environment
        self.env = DeliveryEnv(
            db_handler=self.db_handler,
            use_clustering=use_clustering,
            cluster_eps=cluster_eps,
            max_steps=max_steps
        )

        
        # Load the agent if path is provided
        self.agent = None
        if agent_path:
            self.load_agent(agent_path)
        
        # Assignments
        self.assignments = []
        self.assignment_map = {}  # volunteer_id -> [recipient_ids]
        
    def load_agent(self, agent_path):
        """
        Load a trained RL agent.
        
        Args:
            agent_path (str): Path to the trained agent checkpoint
            
        Returns:
            bool: Whether the agent was successfully loaded
        """
        try:
            # Get state and action dimensions from environment
            state_dim = self.env.observation_space.shape[0]
            action_dim = self.env.action_space.n
            
            # Initialize agent
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            self.agent = ActorCriticAgent(
                state_dim=state_dim,
                action_dim=action_dim,
                device=device
            )
            
            # Load model weights
            self.agent.load_models(agent_path)
            
            print(f"Agent loaded from {agent_path}")
            return True
        
        except Exception as e:
            print(f"Error loading agent: {e}")
            return False
    
    def generate_assignments(self, deterministic=True, max_steps=1000):
        """
        Generate assignments using the trained agent.
        
        Args:
            deterministic (bool): Whether to use deterministic policy
            max_steps (int): Maximum steps to run the episode
            
        Returns:
            bool: Whether assignments were successfully generated
        """
        if self.agent is None:
            print("No agent loaded. Please load an agent first.")
            return False
        
        # Reset environment
        state = self.env.reset()
        
        # Initialize variables
        done = False
        step = 0
        total_reward = 0
        
        while not done and step < max_steps:
            # Select action
            action, _ = self.agent.select_action(state, env=self.env, deterministic=deterministic)
            
            # Take step in environment
            next_state, reward, done, info = self.env.step(action)
            
            # Update state and counters
            state = next_state
            total_reward += reward
            step += 1
        
        # Get assignments
        self.assignments = self.env.assignment_list
        
        # Create mapping of volunteer_id -> [recipient_ids]
        self.assignment_map = {}
        for volunteer_idx, recipient_idx in self.assignments:
            volunteer_id = self.env.volunteers[volunteer_idx].volunteer_id
            recipient_id = self.env.recipients[recipient_idx].recipient_id
            
            if volunteer_id not in self.assignment_map:
                self.assignment_map[volunteer_id] = []
            
            self.assignment_map[volunteer_id].append(recipient_id)
        
        print(f"Generated {len(self.assignments)} assignments in {step} steps")
        print(f"Total reward: {total_reward:.2f}")
        print(f"Assignment rate: {len(self.assignments) / len(self.env.recipients) * 100:.1f}%")
        
        return True
    
    def save_assignments_to_db(self):
        """
        Save the generated assignments to the database.
        
        Returns:
            bool: Whether assignments were successfully saved
        """
        return self.env.save_assignments()
    
    def export_assignments_to_csv(self, filename=None):
        """
        Export the generated assignments to a CSV file.
        
        Args:
            filename (str, optional): Name of the file to save to
            
        Returns:
            str: Path to the saved file
        """
        if not self.assignments:
            print("No assignments to export. Generate assignments first.")
            return None
        
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"assignments_{timestamp}.csv"
        
        filepath = os.path.join(self.output_dir, filename)
        
        # Create DataFrame
        data = []
        for volunteer_idx, recipient_idx in self.assignments:
            volunteer = self.env.volunteers[volunteer_idx]
            recipient = self.env.recipients[recipient_idx]
            
            data.append({
                'volunteer_id': volunteer.volunteer_id,
                # 'volunteer_zip': volunteer.zip_code,
                'volunteer_capacity': volunteer.car_size,
                'recipient_id': recipient.recipient_id,
                'recipient_lat': recipient.latitude,
                'recipient_lon': recipient.longitude,
                'recipient_boxes': recipient.num_items,
                'distance': self.env.distance_matrix[volunteer_idx, recipient_idx]
            })
        
        # Create and save DataFrame
        df = pd.DataFrame(data)
        df.to_csv(filepath, index=False)
        
        print(f"Assignments exported to {filepath}")
        return filepath
    
    def visualize_assignments(self, save_path=None, show=True):
        """
        Visualize volunteer-recipient assignments using Leaflet
        
        Args:
            save_path (str): Optional path to save HTML file
            show (bool): Whether to display the visualization
        """
        import folium
        
        # Get coordinates
        volunteer_coords = self.env.volunteer_coords
        recipient_coords = self.env.recipient_coords
        
        # Create map centered on mean location
        mean_lat = (volunteer_coords[:, 0].mean() + recipient_coords[:, 0].mean()) / 2
        mean_lon = (volunteer_coords[:, 1].mean() + recipient_coords[:, 1].mean()) / 2
        m = folium.Map(location=[mean_lat, mean_lon], zoom_start=12)
        
        # Define color palette for volunteers
        volunteer_colors = {
            0: 'red',
            1: 'blue',
            2: 'green',
            3: 'purple',
            4: 'orange',
            5: 'darkred'
        }
        
        # Add volunteers with interactive popups
        for i, (lat, lon) in enumerate(volunteer_coords):
            if lat < 1:  # Handle invalid coordinates
                lat, lon = 32.7767, -96.7970
            
            color = volunteer_colors.get(i % len(volunteer_colors), 'red')
            
            # Get volunteer's assignments
            assigned_recipients = [r for v,r in self.assignments if v == i]
            total_boxes = sum(self.env.recipients[r].num_items for r in assigned_recipients)
            capacity = self.env.volunteers[i].car_size
            
            # Build popup HTML
            popup_html = f"""
            <div style='width: 250px'>
                <h4>Volunteer {i}</h4>
                <p><b>Capacity:</b> {capacity} boxes</p>
                <p><b>Used:</b> {total_boxes} boxes ({total_boxes/capacity*100:.1f}%)</p>
                
                <h5>Recipients ({len(assigned_recipients)}):</h5>
                <ul>
            """
            
            for r in assigned_recipients:
                recipient = self.env.recipients[r]
                popup_html += f"<li>Recipient {r}: {recipient.num_items} boxes</li>"
            
            popup_html += "</ul>"
            
            # Add cluster info if available
            if hasattr(self.env, 'clusters') and self.env.clusters is not None:
                labels = self.env.clusters.get('labels', [])
                if assigned_recipients:
                    first_recipient = assigned_recipients[0]
                    if len(labels) > first_recipient:
                        cluster_id = labels[first_recipient]
                        if cluster_id != -1:
                            popup_html += f"<p><b>Primary Cluster:</b> {cluster_id}</p>"
            
            popup_html += "</div>"
            
            # Create marker with popup
            folium.Marker(
                [lat, lon],
                icon=folium.Icon(color=color, icon='user', prefix='fa'),
                tooltip=f'Volunteer {i} ({total_boxes}/{capacity} boxes)',
                popup=folium.Popup(popup_html, max_width=300)
            ).add_to(m)
        
        # Add recipients (circles)
        for i, (lat, lon) in enumerate(recipient_coords):
            folium.CircleMarker(
                [lat, lon],
                radius=5,
                color='gray',
                fill=True,
                fill_color='gray',
                tooltip=f'Recipient {i}'
            ).add_to(m)
        
        # Add assignment lines with matching volunteer colors
        for volunteer_idx, recipient_idx in self.assignments:
            vol_coords = volunteer_coords[volunteer_idx]
            rec_coords = recipient_coords[recipient_idx]
            
            color = volunteer_colors.get(volunteer_idx % len(volunteer_colors), 'gray')
            
            folium.PolyLine(
                locations=[vol_coords, rec_coords],
                color=color,
                weight=3,  # Thicker lines
                opacity=0.7
            ).add_to(m)
        
        # Add cluster visualization if available
        if hasattr(self.env, 'clusters') and self.env.clusters is not None:
            centers = self.env.clusters.get('centers', {})
            labels = self.env.clusters.get('labels', [])
            coordinates = self.env.recipient_coords
            
            # Define cluster colors
            cluster_colors = [
                'red', 'blue', 'green', 'purple', 'orange', 'darkred',
                'lightred', 'beige', 'darkblue', 'darkgreen', 'cadetblue'
            ]
            
            # Add cluster visualization
            for label, center in centers.items():
                if label != -1:  # Skip noise cluster
                    # Get cluster members and coordinates
                    cluster_indices = [i for i, l in enumerate(labels) if l == label]
                    cluster_points = coordinates[cluster_indices]
                    
                    # Calculate cluster radius (max distance from center)
                    max_distance = 0
                    for point in cluster_points:
                        dist = np.linalg.norm(point - center)
                        if dist > max_distance:
                            max_distance = dist
                    radius = max_distance * 111320  # Convert to meters (approx)
                    
                    # Create popup with cluster details
                    popup_html = f"""
                    <div style='width: 250px'>
                        <h4>Cluster {label}</h4>
                        <p><b>Center:</b> {center[0]:.4f}, {center[1]:.4f}</p>
                        <p><b>Members:</b> {len(cluster_indices)} recipients</p>
                        <p><b>Radius:</b> {radius:.0f} meters</p>
                    """
                    
                    # Add styled cluster center marker
                    color = cluster_colors[label % len(cluster_colors)]
                    folium.Marker(
                        [center[0], center[1]],
                        icon=folium.Icon(
                            color=color,
                            icon='star',
                            prefix='fa',
                            icon_color='white'
                        ),
                        tooltip=f'Cluster {label} ({len(cluster_indices)} recipients)',
                        popup=folium.Popup(popup_html, max_width=300)
                    ).add_to(m)
                    
                    # Add circle showing actual cluster coverage
                    folium.Circle(
                        location=[center[0], center[1]],
                        radius=radius,
                        color=color,
                        fill=True,
                        fill_color=color,
                        fill_opacity=0.1,
                        weight=2
                    ).add_to(m)
                    
                    # Add cluster members
                    for idx in cluster_indices:
                        point = coordinates[idx]
                        folium.CircleMarker(
                            location=[point[0], point[1]],
                            radius=4,
                            color=color,
                            fill=True,
                            fill_color=color,
                            fill_opacity=0.7,
                            weight=1
                        ).add_to(m)
        
        # Save or show
        if save_path:
            m.save(save_path)
            print(f"Visualization saved to {save_path}")
        
        if show:
            import tempfile
            import webbrowser
            
            # Create temp file
            with tempfile.NamedTemporaryFile(suffix='.html', delete=False) as f:
                temp_path = f.name
                m.save(temp_path)
            
            # Open in default browser
            webbrowser.open(f'file://{temp_path}')
            
        return True
    
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
            print("No assignments to visualize. Generate assignments first.")
            return False
        
        # Calculate load for each volunteer
        volunteer_loads = {}
        volunteer_capacities = {}
        
        for volunteer_idx, recipient_idx in self.assignments:
            volunteer = self.env.volunteers[volunteer_idx]
            recipient = self.env.recipients[recipient_idx]
            
            volunteer_id = volunteer.volunteer_id
            
            if volunteer_id not in volunteer_loads:
                volunteer_loads[volunteer_id] = 0
                volunteer_capacities[volunteer_id] = volunteer.car_size
            
            volunteer_loads[volunteer_id] += recipient.num_items
        
        # Calculate utilization
        volunteer_ids = list(volunteer_loads.keys())
        loads = [volunteer_loads[vid] for vid in volunteer_ids]
        capacities = [volunteer_capacities[vid] for vid in volunteer_ids]
        utilization = [loads[i] / capacities[i] * 100 for i in range(len(loads))]
        
        # Create figure
        plt.figure(figsize=(12, 8))
        
        # Plot capacities and loads
        x = range(len(volunteer_ids))
        width = 0.35
        
        plt.bar(x, capacities, width, label='Capacity', alpha=0.7)
        plt.bar([i + width for i in x], loads, width, label='Actual Load', alpha=0.7)
        
        # Add utilization as text
        for i, util in enumerate(utilization):
            plt.text(
                i + width/2,
                max(capacities[i], loads[i]) + 1,
                f"{util:.1f}%",
                ha='center'
            )
        
        # Add labels
        plt.title('Volunteer Load Distribution')
        plt.xlabel('Volunteer ID')
        plt.ylabel('Number of Boxes')
        plt.xticks([i + width/2 for i in x], volunteer_ids)
        plt.legend()
        plt.grid(True, axis='y')
        
        # Save or show
        if save_path:
            plt.savefig(save_path)
            print(f"Visualization saved to {save_path}")
        
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
            print("No assignments to report. Generate assignments first.")
            return "No assignments available."
        
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        if output_format == 'markdown':
            report = f"# Volunteer Assignment Report\n\n"
            report += f"Generated on: {timestamp}\n\n"
            
            # Summary statistics
            volunteer_count = len(self.assignment_map)
            recipient_count = len(self.assignments)
            total_recipients = len(self.env.recipients)
            assignment_rate = recipient_count / total_recipients * 100
            
            report += f"## Summary\n\n"
            report += f"- **Volunteers Used:** {volunteer_count}\n"
            report += f"- **Recipients Assigned:** {recipient_count} / {total_recipients} ({assignment_rate:.1f}%)\n\n"
            
            # Volunteer assignments
            report += f"## Assignments by Volunteer\n\n"
            
            for volunteer_id, recipient_ids in self.assignment_map.items():
                # Find volunteer object
                volunteer_idx = next(i for i, v in enumerate(self.env.volunteers) 
                                    if v.volunteer_id == volunteer_id)
                volunteer = self.env.volunteers[volunteer_idx]
                
                # Calculate load
                total_boxes = sum(self.env.recipients[next(i for i, r in enumerate(self.env.recipients) 
                                                        if r.recipient_id == rid)].num_items 
                                for rid in recipient_ids)
                
                utilization = total_boxes / volunteer.car_size * 100
                
                report += f"### Volunteer {volunteer_id}\n\n"
                # report += f"- **Zip Code:** {volunteer.zip_code}\n"
                report += f"- **Car Capacity:** {volunteer.car_size} boxes\n"
                report += f"- **Assigned Load:** {total_boxes} boxes ({utilization:.1f}% utilization)\n"
                report += f"- **Assigned Recipients:** {len(recipient_ids)}\n\n"
                
                # Table of recipients
                report += f"| Recipient ID | Boxes | Distance (km) |\n"
                report += f"|-------------|-------|---------------|\n"
                
                for recipient_id in recipient_ids:
                    # Find recipient object
                    recipient_idx = next(i for i, r in enumerate(self.env.recipients) 
                                        if r.recipient_id == recipient_id)
                    recipient = self.env.recipients[recipient_idx]
                    
                    # Calculate distance
                    distance = self.env.distance_matrix[volunteer_idx, recipient_idx]
                    
                    report += f"| {recipient_id} | {recipient.num_items} | {distance:.2f} |\n"
                
                report += "\n"
            
            # Add analysis section
            report += f"## Performance Analysis\n\n"
            
            # Calculate average distance
            total_distance = sum(self.env.distance_matrix[vol_idx, rec_idx] 
                                for vol_idx, rec_idx in self.assignments)
            avg_distance = total_distance / len(self.assignments)
            
            # Calculate average utilization
            total_utilization = 0
            volunteer_used_count = 0
            
            for volunteer_id in self.assignment_map:
                volunteer_idx = next(i for i, v in enumerate(self.env.volunteers) 
                                    if v.volunteer_id == volunteer_id)
                volunteer = self.env.volunteers[volunteer_idx]
                
                recipient_ids = self.assignment_map[volunteer_id]
                total_boxes = sum(self.env.recipients[next(i for i, r in enumerate(self.env.recipients) 
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
    
    def run_complete_pipeline(self, agent_path='./checkpoints/checkpoint_final', export_csv=True, save_visualizations=True, save_report=True):
        """
        Run the complete assignment pipeline.
        
        Args:
            agent_path (str): Path to the trained agent checkpoint
            export_csv (bool): Whether to export assignments to CSV
            save_visualizations (bool): Whether to save visualizations
            save_report (bool): Whether to save the report
            
        Returns:
            bool: Whether the pipeline was successful
        """
        # 1. Load agent
        success = self.load_agent(agent_path)
        if not success:
            return False
        
        # 2. Generate assignments
        success = self.generate_assignments(deterministic=True)
        if not success:
            return False
        
        # 3. Save assignments to database
        self.save_assignments_to_db()
        
        # 4. Export to CSV if requested
        if export_csv:
            self.export_assignments_to_csv()
        
        # 5. Create and save visualizations if requested
        if save_visualizations:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            
            # Assignment map
            map_path = os.path.join(self.output_dir, f"assignment_map_{timestamp}.html")
            self.visualize_assignments(save_path=map_path, show=False)
            
            # Load distribution
            load_path = os.path.join(self.output_dir, f"load_distribution_{timestamp}.png")
            self.visualize_volunteer_load(save_path=load_path, show=False)
        
        # 6. Generate and save report if requested
        if save_report:
            report = self.generate_assignment_report()
            self.save_report(report)
        
        print("Assignment pipeline completed successfully!")
        return True


if __name__ == "__main__":
    # Test the assigner
    assigner = VolunteerAssigner(
        use_clustering=True,
        max_steps=40
    )
    
    # Check if a trained agent exists, otherwise create a simple agent for testing
    checkpoint_dir = "./checkpoints"
    checkpoint_path = os.path.join(checkpoint_dir, "checkpoint_final")
    
    if not os.path.exists(checkpoint_path):
        os.makedirs(checkpoint_dir, exist_ok=True)
        
        # Create a simple agent for testing
        state_dim = assigner.env.observation_space.shape[0]
        action_dim = assigner.env.action_space.n
        
        agent = ActorCriticAgent(state_dim, action_dim)
        agent.save_models(checkpoint_path)
    
    # Generate assignments
    assigner.load_agent(checkpoint_path)
    assigner.generate_assignments()
    
    # Visualize
    assigner.visualize_assignments()
    # assigner.visualize_volunteer_load()
    
    # Generate and print report
    report = assigner.generate_assignment_report()
    print("\nAssignment Report:")
    print(report)
    
    # Export to CSV
    assigner.export_assignments_to_csv()
