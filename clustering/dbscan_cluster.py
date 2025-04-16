#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
DBSCAN-based recipient clustering for the AID-RL project.
Clusters recipients based on their geographic coordinates.
"""

import numpy as np
import pandas as pd
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from scipy.spatial.distance import cdist
import math


class RecipientClusterer:
    """
    Class for clustering recipients based on their geographic coordinates
    using the DBSCAN algorithm.
    """
    
    def __init__(self, eps=0.3, min_samples=3):
        """
        Initialize the clusterer with DBSCAN parameters.
        
        Args:
            eps (float): The maximum distance between two samples for one to be 
                        considered as in the neighborhood of the other.
            min_samples (int): The minimum number of samples in a neighborhood 
                              for a point to be considered as a core point.
        """
        self.eps = eps
        self.min_samples = min_samples
        self.dbscan = DBSCAN(eps=eps, min_samples=min_samples, metric='haversine')
        self.scaler = StandardScaler()
        self.fitted = False
        self.clusters = None
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
        Fit the DBSCAN clustering algorithm to the recipient coordinates.
        
        Args:
            coordinates (numpy.ndarray): Array of (latitude, longitude) pairs
            
        Returns:
            cluster_labels (numpy.ndarray): Cluster labels for each recipient
        """
        # Convert latitude and longitude to radians for Haversine distance
        coordinates_rad = np.radians(coordinates)
        
        # Fit DBSCAN
        self.dbscan.fit(coordinates_rad)
        
        # Store cluster labels
        self.cluster_labels = self.dbscan.labels_
        
        # Calculate cluster centers
        self._calculate_cluster_centers(coordinates)
        
        self.fitted = True
        
        return self.cluster_labels
    
    def _calculate_cluster_centers(self, coordinates):
        """
        Calculate the center of each cluster.
        
        Args:
            coordinates (numpy.ndarray): Array of (latitude, longitude) pairs
        """
        if not hasattr(self, 'cluster_labels'):
            raise ValueError("DBSCAN must be fitted before calculating cluster centers")
        
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
            raise ValueError("DBSCAN must be fitted before getting clusters")
        
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
    
    def visualize_clusters(self, coordinates, recipient_ids=None, volunteer_coords=None, save_path=None):
        """
        Visualize the clustering results.
        
        Args:
            coordinates (numpy.ndarray): Array of (latitude, longitude) pairs
            recipient_ids (list, optional): IDs of the recipients
            volunteer_coords (numpy.ndarray, optional): Volunteer coordinates to plot
            save_path (str, optional): Path to save the figure
        """
        if not self.fitted:
            raise ValueError("DBSCAN must be fitted before visualization")
        
        # Create a figure
        plt.figure(figsize=(10, 8))
        
        # Get unique labels and number of clusters
        unique_labels = set(self.cluster_labels)
        n_clusters = len(unique_labels) - (1 if -1 in unique_labels else 0)
        
        # Plot each cluster
        for label in unique_labels:
            # Get indices of points in the current cluster
            cluster_indices = np.where(self.cluster_labels == label)[0]
            
            # Get coordinates of these points
            cluster_coords = coordinates[cluster_indices]
            
            # Plot these points
            if label == -1:
                # Black for noise points
                plt.scatter(cluster_coords[:, 1], cluster_coords[:, 0], s=50, 
                           c='k', marker='o', label='Noise')
            else:
                # Colored for clusters
                plt.scatter(cluster_coords[:, 1], cluster_coords[:, 0], s=50, 
                           marker='o', label=f'Cluster {label}')
                
                # Plot cluster center
                center = self.cluster_centers[label]
                plt.scatter(center[1], center[0], s=200, marker='*', 
                           edgecolors='k', label=f'Center {label}' if label == list(unique_labels)[1] else "")
                        #    edgecolors='k', label=f'Center {label}' if label == list(unique_labels) else "")
        
        # Plot volunteer locations if provided
        if volunteer_coords is not None:
            plt.scatter(volunteer_coords[:, 1], volunteer_coords[:, 0], s=100, 
                       marker='^', c='red', label='Volunteers')
        
        # Add labels
        # if recipient_ids is not None:
        #     for i, txt in enumerate(recipient_ids):
        #         plt.annotate(txt, (coordinates[i, 1], coordinates[i, 0]), fontsize=8)
        
        plt.title(f'DBSCAN Clustering: {n_clusters} clusters found')
        plt.xlabel('Longitude')
        plt.ylabel('Latitude')
        plt.legend(loc='best')
        plt.grid(True)
        
        # Save or show the figure
        if save_path:
            plt.savefig(save_path)
            plt.close()
        else:
            plt.tight_layout()
            plt.show()


if __name__ == "__main__":
    # Test the clustering with synthetic data
    # from data.db_config import DatabaseHandler

    # Get recipient coordinates
    db = DatabaseHandler()
    recipients = db.get_all_recipients()

    # Combine all points
    all_coords = np.array([[r.latitude, r.longitude] for r in recipients])
    
    # Create recipient IDs
    all_ids = [r.recipient_id for r in recipients]
    
    # Get volunteer coordinates
    # volunteers = db.get_all_volunteers()
    #create random volunteer coordinates in dallas-fort worth texas
    volunteer_coords = np.random.uniform(32.7, 33.0, (100, 2))
    
    # Initialize and fit the clusterer
    clusterer = RecipientClusterer(eps=0.005, min_samples=2)
    labels = clusterer.fit(all_coords)
    
    # Visualize the clusters
    clusterer.visualize_clusters(all_coords, all_ids, volunteer_coords)
