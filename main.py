from data.db_config import DatabaseHandler, Volunteer
# from models.rl_agent import ActorCriticAgent
from clustering.dbscan_cluster import RecipientClusterer
import numpy as np


def main():

    # Test the clustering with synthetic data
    from data.db_config import DatabaseHandler

    # Get recipient coordinates
    db = DatabaseHandler()
    recipients = db.get_all_recipients()

    # Combine all points
    all_coords = np.array([[r.latitude, r.longitude] for r in recipients])
    
    # Create recipient IDs
    all_ids = [r.recipient_id for r in recipients]
    
    # Get volunteer coordinates
    # volunteers = db.get_all_volunteers()
    lat_min, lat_max = 32.7, 33.0
    lon_min, lon_max = -97.8, -96.2
    num_volunteers = 3
    latitudes = np.random.uniform(lat_min, lat_max, num_volunteers)
    longitudes = np.random.uniform(lon_min, lon_max, num_volunteers)

    # Combine into a list of tuples
    volunteer_coords = np.array(list(zip(latitudes, longitudes)))
    
    # Initialize and fit the clusterer
    clusterer = RecipientClusterer(eps=0.0005, min_samples=3)
    labels = clusterer.fit(all_coords)
    
    # Visualize the clusters
    clusterer.visualize_clusters(all_coords, all_ids, volunteer_coords)

    

if __name__ == "__main__":
    main()