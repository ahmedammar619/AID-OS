#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Quick capacity check script for AID-OS.
This script generates a quick report of volunteer capacity and recipient needs
without running the full assignment algorithm.
"""

import sys
import os
import numpy as np
import math
from datetime import datetime

# Add parent directory to path to import from project modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from data.db_config import DatabaseHandler

def haversine_distance(lat1, lon1, lat2, lon2):
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

def quick_capacity_report():
    """
    Generate a quick report of volunteer capacity and recipient needs.
    
    Returns:
        dict: Summary statistics about capacity and need.
    """
    # Initialize database handler
    db = DatabaseHandler()
    
    # Load data
    volunteers = db.get_all_volunteers()
    recipients = db.get_all_recipients()
    
    if not volunteers or not recipients:
        print("No data loaded!")
        return None
        
    # Calculate total capacity
    total_capacity = sum(v.car_size for v in volunteers)
    
    # Calculate total need
    total_boxes = sum(r.num_items for r in recipients)
    
    # Calculate volunteer capacity information
    volunteer_capacities = [v.car_size for v in volunteers]
    min_capacity = min(volunteer_capacities)
    max_capacity = max(volunteer_capacities)
    avg_capacity = total_capacity / len(volunteers)
    
    # Calculate recipient need information
    recipient_needs = [r.num_items for r in recipients]
    min_need = min(recipient_needs)
    max_need = max(recipient_needs)
    avg_need = total_boxes / len(recipients)
    
    # Calculate overall utilization if all volunteers were used
    overall_utilization = (total_boxes / total_capacity) * 100
    
    # Theoretical minimum number of volunteers needed (just based on capacity)
    min_volunteers_needed = math.ceil(total_boxes / max_capacity)
    
    # Calculate average distance
    total_distance = 0
    count = 0
    for v in volunteers:
        for r in recipients:
            distance = haversine_distance(v.latitude, v.longitude, r.latitude, r.longitude)
            total_distance += distance
            count += 1
    
    avg_distance = total_distance / count if count > 0 else 0
    
    # Report results
    report = {
        'total_capacity': total_capacity,
        'total_boxes': total_boxes,
        'min_capacity': min_capacity,
        'max_capacity': max_capacity,
        'avg_capacity': avg_capacity,
        'min_need': min_need,
        'max_need': max_need,
        'avg_need': avg_need,
        'overall_utilization': overall_utilization,
        'min_volunteers_needed': min_volunteers_needed,
        'avg_distance': avg_distance
    }
    
    print("\n=== QUICK CAPACITY REPORT ===")
    print(f"Total volunteer capacity: {total_capacity} boxes")
    print(f"Total recipient boxes: {total_boxes} boxes")
    print(f"Overall utilization if all volunteers used: {overall_utilization:.1f}%")
    print(f"Theoretical minimum volunteers needed: {min_volunteers_needed}")
    print(f"Average distance between any volunteer and recipient: {avg_distance:.2f} km")
    print("\nVolunteer capacity:")
    print(f"  Min: {min_capacity} boxes")
    print(f"  Max: {max_capacity} boxes")
    print(f"  Avg: {avg_capacity:.1f} boxes")
    print("\nRecipient needs:")
    print(f"  Min: {min_need} boxes")
    print(f"  Max: {max_need} boxes")
    print(f"  Avg: {avg_need:.1f} boxes")
    print("===========================\n")
    
    return report

def main():
    """Run a quick capacity check."""
    print("Running quick capacity check...")
    
    # Generate and print the quick capacity report
    quick_capacity_report()
    
    print("Quick capacity check completed.")

if __name__ == "__main__":
    main() 