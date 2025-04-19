from data.db_config import DatabaseHandler, Volunteer
# from models.rl_agent import ActorCriticAgent
from clustering.dbscan_cluster import RecipientClusterer
import numpy as np
import os


def main():

    # Test the clustering with synthetic data
    from data.db_config import DatabaseHandler

    # Get recipient coordinates
    db = DatabaseHandler()
    recipients = db.get_all_recipients()

    print(len(recipients))
    # Combine all points
    all_coords = np.array([[r.latitude, r.longitude] for r in recipients])
    
    # Create recipient IDs
    all_ids = [r.recipient_id for r in recipients]
    recipient_boxes = [r.num_items for r in recipients]
    
    # Get volunteer coordinates
    # volunteers = db.get_all_volunteers()
    lat_min, lat_max = 32.7, 33.0
    lon_min, lon_max = -97.8, -96.2
    num_volunteers = 3
    latitudes = np.random.uniform(lat_min, lat_max, num_volunteers)
    longitudes = np.random.uniform(lon_min, lon_max, num_volunteers)

    # Combine into a list of tuples
    volunteer_coords = np.array([[v.latitude, v.longitude] for v in db.get_all_volunteers()])
    # Initialize and fit the clusterer
    clusterer = RecipientClusterer(
        min_cluster_size=2,
        cluster_selection_epsilon=0.00005,
        min_samples=1
    )
    labels = clusterer.fit(all_coords)

    # Get pickups
    pickups = db.get_all_pickups()
    pickup_coords = np.array([[p.latitude, p.longitude] for p in pickups])
    
    # Visualize the clusters
    output_path = os.path.join("./output", "cluster_map.html")
    clusterer.visualize_clusters(
        all_coords, 
        all_ids, 
        recipient_boxes,
        volunteer_coords,
        save_path=output_path,
        pickup_coords=pickup_coords
    )

    

if __name__ == "__main__":
    main()