#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Compare admin assignments with optimized assignments.
This script:
1. Loads current admin assignments from the database
2. Runs the optimization algorithm on the same dataset
3. Generates interactive maps for both assignments for comparison
4. Does NOT modify the original admin assignments in the database
"""

import os
import sys
import time
import math
import json
from datetime import datetime
from collections import defaultdict, namedtuple
import folium
import webbrowser
import numpy as np
from ortools.linear_solver import pywraplp
from assets.map_visualization import visualize_assignments

# Add parent directory to path to import from project modules
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from data.db_config import DatabaseHandler
from assignment.assign_volunteers_opt import VolunteerAssignerOpt

def get_admin_assignments():
    """Get current admin assignments from the delivery table."""
    print("Fetching current assignments from the delivery table...")
    db = DatabaseHandler()
    
    # Get all assignments from the delivery table (regardless of status)
    query = """
    SELECT d.volunteer_id, d.recipient_id, v.pickup_location_id 
    FROM delivery d 
    JOIN volunteer v ON d.volunteer_id = v.volunteer_id 
    where d.volunteer_id is not null
    """

    assignment_rows = db.execute_raw_query(query)
    
    
    if not assignment_rows:
        print("No assignments found in the delivery table.")
        return None
    
    # Convert to the format we need: (volunteer_id, recipient_id, pickup_id)
    assignments = []
    volunteer_ids = set()
    recipient_ids = set()
    pickup_ids = set()
    
    for row in assignment_rows:
        vol_id = row.get('volunteer_id')
        rec_id = row.get('recipient_id')
        pickup_id = row.get('pickup_location_id')
        if vol_id and rec_id and pickup_id:
            assignments.append((vol_id, rec_id, pickup_id))
            volunteer_ids.add(vol_id)
            recipient_ids.add(rec_id)
            pickup_ids.add(pickup_id)
    
    # Get all volunteers and recipients from the database
    all_volunteers = db.get_all_volunteers()
    all_recipients = db.get_all_recipients()
    all_pickups = db.get_all_pickups()
    
    # Filter to only include those in the delivery table
    volunteers = [v for v in all_volunteers if v.volunteer_id in volunteer_ids]
    recipients = [r for r in all_recipients if r.recipient_id in recipient_ids]
    pickups = [p for p in all_pickups if p.location_id in pickup_ids]
    print(f"Using {len(volunteers)} volunteers to deliver to {len(recipients)} recipients")
    
    # Create assignment map: volunteer_id -> [recipient_ids]
    assignment_map = {}
    for volunteer in volunteers:
        assignment_map[volunteer.volunteer_id] = []
    
    # Fill in assignment map
    for vol_id, rec_id, pickup_id in assignments:
        if vol_id in assignment_map:
            assignment_map[vol_id].append(rec_id)
    
    # Calculate volunteer box counts
    volunteer_box_counts = {}
    for vol_id, rec_ids in assignment_map.items():
        total_boxes = 0
        for rec_id in rec_ids:
            # Find recipient
            recipient = next((r for r in recipients if r.recipient_id == rec_id), None)
            if recipient:
                try:
                    num_items = int(recipient.num_items) if isinstance(recipient.num_items, str) else recipient.num_items
                except (ValueError, TypeError):
                    num_items = 1
                total_boxes += num_items
        volunteer_box_counts[vol_id] = total_boxes
    
    print(f"Found {len(assignments)} admin assignments")
    print(f"Using {len(volunteers)} volunteers to deliver to {len(recipients)} recipients")
    
    # Create data dictionary
    data = {
        'assignments': assignments,
        'volunteers': volunteers,
        'recipients': recipients,
        'assignment_map': assignment_map,
        'pickups': pickups,
        'volunteer_box_counts': volunteer_box_counts
    }
    
    return data

def haversine_distance(lat1, lon1, lat2, lon2):
    """Calculate the distance between two points using the haversine formula."""
    R = 6371  # Earth radius in kilometers
    lat1_rad = math.radians(lat1)
    lon1_rad = math.radians(lon1)
    lat2_rad = math.radians(lat2)
    lon2_rad = math.radians(lon2)
    
    dlon = lon2_rad - lon1_rad
    dlat = lat2_rad - lat1_rad
    
    a = math.sin(dlat/2)**2 + math.cos(lat1_rad) * math.cos(lat2_rad) * math.sin(dlon/2)**2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1-a))
    distance = R * c
    
    return distance

def calculate_assignment_stats(data):
    """Calculate statistics for an assignment."""
    stats = {}
    
    # Count total volunteers and recipients
    stats['total_volunteers'] = len([v_id for v_id, r_ids in data['assignment_map'].items() if r_ids])
    stats['total_recipients'] = sum(len(r_ids) for r_ids in data['assignment_map'].values())
    
    # Calculate total distance
    total_distance = 0
    route_lengths = []
    utilizations = []
    
    # Create a dictionary to map volunteer_id to pickup_id
    volunteer_pickup = {}
    for vol_id, rec_id, pickup_id in data['assignments']:
        volunteer_pickup[vol_id] = pickup_id
    
    for volunteer_id, recipient_ids in data['assignment_map'].items():
        if not recipient_ids:
            continue
        
        # Find volunteer
        volunteer = next((v for v in data['volunteers'] if v.volunteer_id == volunteer_id), None)
        if not volunteer:
            continue
        
        # Find pickup location
        pickup_id = volunteer_pickup.get(volunteer_id)
        if not pickup_id:
            continue
            
        pickup = next((p for p in data['pickups'] if p.location_id == pickup_id), None)
        if not pickup:
            continue
        
        # Calculate route distance: volunteer -> pickup
        route_distance = haversine_distance(
            volunteer.latitude, volunteer.longitude,
            pickup.latitude, pickup.longitude
        )
        
        # Find recipients
        recipients = [r for r in data['recipients'] if r.recipient_id in recipient_ids]
        
        # For optimized routes, we'll use a simplified approach:
        # 1. Volunteer -> Pickup
        # 2. Pickup -> Each recipient (and back to pickup)
        # 3. Pickup -> Volunteer
        
        # Add distances for each recipient (out and back to pickup)
        recipient_distances = 0
        for recipient in recipients:
            recipient_distances += haversine_distance(
                pickup.latitude, pickup.longitude,
                recipient.latitude, recipient.longitude
            )
        
        # Add total recipient distances (we assume they go back to pickup after each delivery)
        route_distance += 2 * recipient_distances
        
        # Add pickup -> volunteer (return trip)
        route_distance += haversine_distance(
            pickup.latitude, pickup.longitude,
            volunteer.latitude, volunteer.longitude
        )
        
        total_distance += route_distance
        route_lengths.append(route_distance)
        
        # Calculate utilization
        total_boxes = 0
        for recipient in recipients:
            try:
                num_items = int(recipient.num_items) if isinstance(recipient.num_items, str) else recipient.num_items
            except (ValueError, TypeError):
                num_items = 1
            total_boxes += num_items
        
        try:
            car_size = int(volunteer.car_size) if isinstance(volunteer.car_size, str) else volunteer.car_size
        except (ValueError, TypeError):
            car_size = 6  # Default
            
        utilization = min(100, (total_boxes / car_size * 100)) if car_size > 0 else 0
        utilizations.append(utilization)
    
    stats['total_distance'] = total_distance
    stats['avg_route_length'] = sum(route_lengths) / len(route_lengths) if route_lengths else 0
    stats['avg_utilization'] = sum(utilizations) / len(utilizations) if utilizations else 0
    
    return stats

# def run_optimized_assignments(admin_data, show_maps=True, output_dir='./hist/output'):


def run_optimized_assignments(admin_data, show_maps=False, output_dir='./hist/output'):
    """Run an optimized assignment algorithm that uses the same volunteers and recipients as in admin_data."""
    
    # Create a new data structure for optimized assignments
    from collections import namedtuple, defaultdict
    
    Volunteer = namedtuple('Volunteer', ['volunteer_id', 'latitude', 'longitude', 'car_size'])
    Recipient = namedtuple('Recipient', ['recipient_id', 'latitude', 'longitude', 'num_items'])

    opt_agent = VolunteerAssignerOpt(data=admin_data, output_dir=output_dir, use_clustering=False)
    opt_agent.generate_assignments()    

    volunteers = opt_agent.volunteers
    recipients = opt_agent.recipients
    assigned_recipients = opt_agent.assigned_recipients
    # Final count of volunteers used
    final_used_volunteers = [v_id for v_id, r_ids in opt_agent.assignment_map.items() if r_ids]
    print(f"\nAfter redistribution: Using {len(final_used_volunteers)} of {len(volunteers)} volunteers")
    
    # Print assignment summary
    print(f"\nAssignment summary:")
    print(f"Total recipients assigned: {len(assigned_recipients)} of {len(recipients)}")
    print(f"Total volunteers used: {len([v for v, r in opt_agent.assignment_map.items() if r])} of {len(volunteers)}")
    
    # Check for empty assignments
    if not opt_agent.assignments:
        print("WARNING: No assignments were made! Using admin assignments as fallback.")
        return None
    
    # Calculate volunteer box counts
    volunteer_box_counts = {}
    for vol_id, rec_ids in opt_agent.assignment_map.items():
        total_boxes = sum(r.num_items for r in recipients if r.recipient_id in rec_ids)
        volunteer_box_counts[vol_id] = total_boxes
    
    # Create optimized data dictionary
    opt_data = {
        'assignments': opt_agent.assignments,
        'volunteers': volunteers,
        'recipients': recipients,
        'assignment_map': dict(opt_agent.assignment_map),  # Convert defaultdict to regular dict
        'pickups': opt_agent.pickups,
        'volunteer_box_counts': volunteer_box_counts
    }
    
    # Calculate statistics
    admin_stats = calculate_assignment_stats(admin_data)
    opt_stats = calculate_assignment_stats(opt_data)
    opt_data['stats'] = opt_stats
    
    # Create timestamp for output files
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    os.makedirs(output_dir, exist_ok=True)
    
    # Create admin assignment map
    admin_map_path = os.path.join(output_dir, f"admin_assignment_map_{timestamp}.html")
    admin_map = visualize_assignments(
        data=admin_data,
        title="Admin Assignments",
        config={
            'show_volunteers': True,
            'show_recipients': True,
            'show_pickups': True,
            'show_routes': True,
            'show_stats': True,
            'volunteer_color': 'blue',
            'recipient_color': 'red',
            'pickup_color': 'green',
            'route_color': 'purple'
        },
        save_path=admin_map_path,
        show=show_maps
    )
    
    # Create optimized assignment map
    opt_map_path = os.path.join(output_dir, f"optimized_assignment_map_{timestamp}.html")
    opt_map = visualize_assignments(
        data=opt_data,
        title="Optimized Assignments",
        config={
            'show_volunteers': True,
            'show_recipients': True,
            'show_pickups': True,
            'show_routes': True,
            'show_stats': True,
            'volunteer_color': 'purple',
            'recipient_color': 'orange',
            'pickup_color': 'green',
            'route_color': 'blue'
        },
        save_path=opt_map_path,
        show=show_maps
    )
    
    return admin_stats, opt_stats, admin_map_path, opt_map_path

def compare_assignments(admin_stats, opt_stats):
    """Compare admin and optimized assignments."""
    print("\n" + "="*50)
    print("ASSIGNMENT COMPARISON")
    print("="*50)
    
    print(f"{'Metric':<25} {'Admin':<15} {'Optimized':<15} {'Difference':<15} {'% Change':<15}")
    print("-"*75)
    
    metrics = [
        ('Total Volunteers', 'total_volunteers', 0),
        ('Total Recipients', 'total_recipients', 0),
        ('Total Distance (km)', 'total_distance', 2),
        ('Avg Route Length (km)', 'avg_route_length', 2),
        ('Avg Utilization (%)', 'avg_utilization', 1)
    ]
    
    for label, key, decimals in metrics:
        admin_val = admin_stats[key]
        opt_val = opt_stats[key]
        diff = opt_val - admin_val
        pct_change = (diff / admin_val * 100) if admin_val != 0 else 0
        
        # Format with appropriate decimals
        admin_str = f"{admin_val:.{decimals}f}" if decimals > 0 else f"{admin_val}"
        opt_str = f"{opt_val:.{decimals}f}" if decimals > 0 else f"{opt_val}"
        diff_str = f"{diff:.{decimals}f}" if decimals > 0 else f"{diff}"
        pct_str = f"{pct_change:.1f}%"
        
        print(f"{label:<25} {admin_str:<15} {opt_str:<15} {diff_str:<15} {pct_str:<15}")
    
    print("="*75)
    print("Note: Negative difference/change means the optimized assignment is better (lower).")
    print("      Positive difference/change for utilization means better (higher).")

def main():
    """Main function to run the comparison."""
    admin_data = get_admin_assignments()
    
    if not admin_data:
        print("No admin assignments found. Cannot run optimization without assignments.")
        return
    
    # Calculate admin stats
    admin_stats = calculate_assignment_stats(admin_data)
    admin_data['stats'] = admin_stats
    
    # Print admin assignment statistics
    print(f"\nAdmin Assignment Statistics:")
    print(f"Total Volunteers: {admin_stats['total_volunteers']}")
    print(f"Total Recipients: {admin_stats['total_recipients']}")
    print(f"Total Distance: {admin_stats['total_distance']:.2f} km")
    print(f"Average Route Length: {admin_stats['avg_route_length']:.2f} km")
    print(f"Average Utilization: {admin_stats['avg_utilization']:.1f}%")
    
    print(f"\nRunning optimization on the same dataset...")
    result = run_optimized_assignments(admin_data)
    
    if result:
        admin_stats, opt_stats, admin_map_path, opt_map_path = result
        
        # Display comparison between admin and optimized assignments
        compare_assignments(admin_stats, opt_stats)
        
        # Open maps in browser
        print("\nOpening maps in browser for comparison...")
        # webbrowser.open('file://' + os.path.abspath(admin_map_path))
        # webbrowser.open('file://' + os.path.abspath(opt_map_path))
    else:
        print("\nOptimization failed. Only showing admin assignments.")

if __name__ == "__main__":
    main()
