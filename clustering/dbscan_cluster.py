#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
DBSCAN-based recipient clustering for the AID-RL project.
Clusters recipients based on their geographic coordinates.
"""

import numpy as np
import pandas as pd
import hdbscan
from sklearn.preprocessing import StandardScaler
import folium
from geopy.distance import geodesic
import math
import webbrowser
import os
from pathlib import Path


class RecipientClusterer:
    """
    Class for clustering recipients based on their geographic coordinates
    using the HDBSCAN algorithm.
    """
    
    def __init__(self, min_cluster_size=5, min_samples=None, cluster_selection_epsilon=0.5):
        """
        Initialize the HDBSCAN clusterer.
        
        Args:
            min_cluster_size (int): Minimum size of clusters
            min_samples (int): Number of samples in a neighborhood
            cluster_selection_epsilon (float): Distance threshold for cluster selection
        """
        self.clusterer = hdbscan.HDBSCAN(
            min_cluster_size=min_cluster_size,
            min_samples=min_samples,
            cluster_selection_epsilon=cluster_selection_epsilon,
            metric='haversine'
        )
        self.fitted = False
        self.cluster_labels = None
        self.cluster_centers = None
        
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
        lat1, lon1, lat2, lon2 = map(math.radians, [lat1, lon1, lat2, lon2])
        
        # Haversine formula
        dlat = lat2 - lat1
        dlon = lon2 - lon1
        a = math.sin(dlat/2)**2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon/2)**2
        c = 2 * math.asin(math.sqrt(a))
        r = 6371  # Radius of Earth in kilometers
        
        return c * r
    
    def get_distance_matrix(self, coordinates):
        """
        Calculate the distance matrix between all pairs of coordinates.
        
        Args:
            coordinates (numpy.ndarray): Array of (latitude, longitude) pairs
            
        Returns:
            distances (numpy.ndarray): Matrix of distances between all pairs
        """
        n = coordinates.shape[0]
        distances = np.zeros((n, n))
        
        for i in range(n):
            for j in range(i+1, n):
                lat1, lon1 = coordinates[i]
                lat2, lon2 = coordinates[j]
                
                # Calculate Haversine distance
                dist = self._haversine_distance(lat1, lon1, lat2, lon2)
                
                # Store the distance (symmetric matrix)
                distances[i, j] = dist
                distances[j, i] = dist
        
        return distances
    
    def fit(self, coordinates):
        """
        Fit the HDBSCAN clustering algorithm to the recipient coordinates.
        
        Args:
            coordinates (numpy.ndarray): Array of (latitude, longitude) pairs
            
        Returns:
            cluster_labels (numpy.ndarray): Cluster labels for each recipient
        """
        # Convert coordinates to radians for Haversine distance
        coordinates_rad = np.radians(coordinates)
        
        # Fit HDBSCAN
        self.cluster_labels = self.clusterer.fit_predict(coordinates_rad)
        self.fitted = True
        
        # Calculate cluster centers
        self._calculate_cluster_centers(coordinates)
        
        # Calculate cluster counts
        self.cluster_counts = {label: np.sum(self.cluster_labels == label) for label in np.unique(self.cluster_labels)}
        
        return self.cluster_labels
    
    def _calculate_cluster_centers(self, coordinates):
        """
        Calculate the center of each cluster.
        
        Args:
            coordinates (numpy.ndarray): Array of (latitude, longitude) pairs
        """
        if not hasattr(self, 'cluster_labels'):
            raise ValueError("HDBSCAN must be fitted before calculating cluster centers")
        
        # Get unique cluster labels (excluding noise points with label -1)
        unique_labels = np.unique(self.cluster_labels)
        unique_labels = unique_labels[unique_labels != -1]
        
        # Initialize dictionary to store cluster centers
        self.cluster_centers = {}
        
        # Calculate centroid for each cluster
        for label in unique_labels:
            # Get indices of points in the current cluster
            cluster_indices = np.where(self.cluster_labels == label)[0]
            
            # Get coordinates of these points
            cluster_coords = coordinates[cluster_indices]
            
            # Calculate mean center
            center = np.mean(cluster_coords, axis=0)
            
            # Store in dictionary
            self.cluster_centers[label] = center
    
    def get_clusters(self):
        """
        Get the clustering results.
        
        Returns:
            dict: Dictionary with cluster information
        """
        if not self.fitted:
            raise ValueError("HDBSCAN must be fitted before getting clusters")
        
        # Count number of elements in each cluster
        cluster_counts = {}
        for label in self.cluster_labels:
            if label in cluster_counts:
                cluster_counts[label] += 1
            else:
                cluster_counts[label] = 1
        
        # Create dictionary with cluster information
        clusters = {
            'labels': self.cluster_labels,
            'counts': cluster_counts,
            'centers': self.cluster_centers
        }
        
        return clusters
    
    def visualize_clusters(self, coordinates, recipient_ids=None, recipient_boxes=None, volunteer_coords=None, save_path=None):
        """
        Visualize the clustering results on an interactive Leaflet map.
        
        Args:
            coordinates (numpy.ndarray): Array of (latitude, longitude) pairs
            recipient_ids (list, optional): IDs of the recipients
            recipient_boxes (list, optional): Number of boxes for each recipient
            volunteer_coords (numpy.ndarray, optional): Volunteer coordinates to plot
            save_path (str, optional): Path to save the HTML map
        """
        if not self.fitted:
            raise ValueError("HDBSCAN must be fitted before visualization")
            
        # Create a base map centered on the mean of all points
        mean_lat = np.mean(coordinates[:, 0])
        mean_lon = np.mean(coordinates[:, 1])
        m = folium.Map(location=[mean_lat, mean_lon], zoom_start=12)
        
        # Get unique labels and colors for clusters
        unique_labels = set(self.cluster_labels)
        colors = ['red', 'blue', 'green', 'purple', 'orange', 'darkred',
                 'lightred', 'beige', 'darkblue', 'darkgreen', 'cadetblue',
                 'darkpurple', 'pink', 'lightblue', 'lightgreen', 'gray',
                 'black', 'lightgray']
        
        # Create feature groups for each cluster
        cluster_groups = {}
        
        # First, add all individual recipient points
        for i, (lat, lon) in enumerate(coordinates):
            # Get cluster label for this point
            label = self.cluster_labels[i]
            
            # Create popup content
            popup_content = f"""
            <div style='width:200px'>
                <b>Recipient:</b> {recipient_ids[i] if recipient_ids else i}<br>
                <b>Boxes:</b> {recipient_boxes[i] if recipient_boxes else 'N/A'}<br>
                <b>Cluster:</b> {'Noise' if label == -1 else f'Cluster {label}'}
            </div>
            """
            
            # Set color based on cluster
            if label == -1:
                # Noise points
                color = '#666666'
                fill_color = '#666666'
                fill_opacity = 0.7
                tooltip = None  # No tooltip for noise points as requested
            else:
                # Clustered points
                color = colors[label % len(colors)]
                fill_color = colors[label % len(colors)]
                fill_opacity = 0.7
                tooltip = f'Recipient {recipient_ids[i] if recipient_ids else i}: {recipient_boxes[i] if recipient_boxes else "N/A"} boxes'
                
                # Add to cluster group for later reference
                if label not in cluster_groups:
                    cluster_groups[label] = []
                cluster_groups[label].append({
                    'id': recipient_ids[i] if recipient_ids else i,
                    'boxes': recipient_boxes[i] if recipient_boxes else 'N/A',
                    'lat': lat,
                    'lon': lon
                })
            
            # Add circle marker
            folium.CircleMarker(
                location=[lat, lon],
                radius=6,
                tooltip=tooltip,
                popup=popup_content,
                color=color,
                fill=True,
                fill_color=fill_color,
                fill_opacity=fill_opacity
            ).add_to(m)
        
        # Now add cluster centers with detailed information
        for label in unique_labels:
            if label == -1:
                continue  # Skip noise cluster
                
            if label in self.cluster_centers:
                center = self.cluster_centers[label]
                
                # Create detailed popup with list of all recipients in this cluster
                if label in cluster_groups:
                    recipients_in_cluster = cluster_groups[label]
                    
                    # Create HTML table for recipients
                    popup_html = f"""
                    <div style='width:300px; max-height:300px; overflow-y:auto;'>
                        <h4>Cluster {label} - {len(recipients_in_cluster)} Recipients</h4>
                        <table style='width:100%; border-collapse:collapse;'>
                            <tr>
                                <th style='border:1px solid #ddd; padding:4px;'>ID</th>
                                <th style='border:1px solid #ddd; padding:4px;'>Boxes</th>
                            </tr>
                    """
                    
                    # Add rows for each recipient
                    for recipient in recipients_in_cluster:
                        popup_html += f"""
                            <tr>
                                <td style='border:1px solid #ddd; padding:4px;'>{recipient['id']}</td>
                                <td style='border:1px solid #ddd; padding:4px;'>{recipient['boxes']}</td>
                            </tr>
                        """
                    
                    popup_html += """
                        </table>
                    </div>
                    """
                    
                    # Create popup with HTML content
                    popup = folium.Popup(popup_html, max_width=300)
                else:
                    popup = f'Cluster {label} Center'
                
                # Calculate actual cluster radius (max distance from center)
                cluster_points = coordinates[self.cluster_labels == label]
                if len(cluster_points) > 0:
                    distances = [self._haversine_distance(center[0], center[1], pt[0], pt[1]) 
                                for pt in cluster_points]
                    cluster_radius = max(distances) * 1000  # Convert km to meters
                else:
                    cluster_radius = self.clusterer.cluster_selection_epsilon * 1000
                
                # Draw adaptive circle around cluster
                folium.Circle(
                    location=center,
                    radius=cluster_radius,
                    color=colors[label % len(colors)],
                    fill=True,
                    fill_opacity=0.15,
                    weight=1,
                    fill_color=colors[label % len(colors)],
                    tooltip=f'Cluster {label} radius: {cluster_radius/1000:.2f}km'
                ).add_to(m)
                
                # Add marker for cluster center with recipient count
                recipient_count = len(cluster_groups.get(label, [])) if label in cluster_groups else 0
                folium.Marker(
                    location=center,
                    icon=folium.DivIcon(
                        icon_size=(20, 20),
                        icon_anchor=(10, 10),
                        html=f"""
                        <div style="
                            background: {colors[label % len(colors)]};
                            border-radius: 50%;
                            width: 20px;
                            height: 20px;
                            display: flex;
                            align-items: center;
                            justify-content: center;
                            font-weight: bold;
                            color: white;
                            text-shadow: 1px 1px 1px rgba(0,0,0,0.5);
                            font-size: 12px;
                        ">
                            {recipient_count}
                        </div>
                        """
                    ),
                    tooltip=f'Cluster {label}: {recipient_count} recipients',
                    popup=popup
                ).add_to(m)
        
        # Add volunteers if provided
        if volunteer_coords is not None:
            for i, (lat, lon) in enumerate(volunteer_coords):
                folium.Marker(
                    location=[lat, lon],
                    icon=folium.Icon(color='red', icon='user', prefix='fa'),
                    tooltip=f'Volunteer {i+1}',
                    popup=f'Volunteer {i+1}'
                ).add_to(m)
        
        # Add layer control
        folium.LayerControl().add_to(m)
        
        # Save or display the map
        if save_path:
            # Ensure directory exists
            Path(save_path).parent.mkdir(parents=True, exist_ok=True)
            m.save(save_path)
            print(f"Map saved to {save_path}")
            
            # Open in browser
            webbrowser.open(f'file://{os.path.abspath(save_path)}')
        else:
            # Save to temp file and open
            temp_path = os.path.join(Path.home(), 'temp_cluster_map.html')
            m.save(temp_path)
            webbrowser.open(f'file://{temp_path}')
        
        return m
