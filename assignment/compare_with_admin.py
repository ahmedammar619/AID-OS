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

def run_optimized_assignments(admin_data, show_maps=True, output_dir='./hist/output'):
    """Run an optimized assignment algorithm that uses the same volunteers and recipients as in admin_data."""
    print("Running optimized assignment algorithm...")
    
    # Create a new data structure for optimized assignments
    from collections import namedtuple, defaultdict
    
    Volunteer = namedtuple('Volunteer', ['volunteer_id', 'latitude', 'longitude', 'car_size'])
    Recipient = namedtuple('Recipient', ['recipient_id', 'latitude', 'longitude', 'num_items'])
    
    # Use the same volunteers and recipients as in admin_data
    # This ensures we're only working with volunteers and recipients that appear in the delivery table
    volunteers = []
    for v in admin_data['volunteers']:
        # Skip volunteers without coordinates or car size
        if not v.latitude or not v.longitude or not v.car_size:
            continue
            
        try:
            car_size = int(v.car_size) if isinstance(v.car_size, str) else v.car_size
        except (ValueError, TypeError):
            car_size = 6  # Default
            
        volunteers.append(Volunteer(
            volunteer_id=v.volunteer_id,
            latitude=v.latitude,
            longitude=v.longitude,
            car_size=car_size
        ))
    
    print(f"Total volunteers from delivery table: {len(volunteers)}")
    
    recipients = []
    for r in admin_data['recipients']:
        try:
            num_items = int(r.num_items) if isinstance(r.num_items, str) else r.num_items
        except (ValueError, TypeError):
            num_items = 1  # Default
            
        recipients.append(Recipient(
            recipient_id=r.recipient_id,
            latitude=r.latitude,
            longitude=r.longitude,
            num_items=num_items
        ))
    
    pickups = admin_data['pickups'][:]
    
    print(f"Total volunteers available: {len(volunteers)}")
    print(f"Total recipients to assign: {len(recipients)}")
    print(f"Total pickup locations: {len(pickups)}")
    
    # Assign each volunteer to their closest pickup
    volunteer_pickup = {}
    for v in volunteers:
        closest_pickup = min(pickups, key=lambda p: 
                          haversine_distance(v.latitude, v.longitude, p.latitude, p.longitude))
        volunteer_pickup[v.volunteer_id] = closest_pickup.location_id
    
    # Calculate distance matrix between volunteers and recipients
    # This will help us make more distance-efficient assignments
    distance_matrix = {}
    for v in volunteers:
        pickup_id = volunteer_pickup[v.volunteer_id]
        pickup = next(p for p in pickups if p.location_id == pickup_id)
        
        # Calculate volunteer -> pickup distance
        vol_to_pickup = haversine_distance(
            v.latitude, v.longitude,
            pickup.latitude, pickup.longitude
        )
        
        for r in recipients:
            # Calculate pickup -> recipient distance
            pickup_to_recipient = haversine_distance(
                pickup.latitude, pickup.longitude,
                r.latitude, r.longitude
            )
            
            # Total route distance: volunteer -> pickup -> recipient -> pickup -> volunteer
            # We're assuming the volunteer returns to the pickup after each delivery
            total_distance = vol_to_pickup + (2 * pickup_to_recipient) + vol_to_pickup
            
            distance_matrix[(v.volunteer_id, r.recipient_id)] = total_distance
    
    # Track remaining capacity for each volunteer
    remaining_capacity = {v.volunteer_id: v.car_size for v in volunteers}
    
    # Initialize assignments
    assignments = []
    assignment_map = defaultdict(list)
    
    # Track assigned recipients to avoid duplicates
    assigned_recipients = set()
    
    # Sort recipients by number of boxes (descending) to assign larger deliveries first
    sorted_recipients = sorted(recipients, key=lambda r: r.num_items, reverse=True)
    
    # First pass: assign recipients to volunteers based on distance and capacity
    for recipient in sorted_recipients:
        # Skip if already assigned
        if recipient.recipient_id in assigned_recipients:
            continue
            
        # Find all valid volunteer-recipient pairs sorted by distance
        valid_assignments = []
        for volunteer in volunteers:
            # Check if volunteer has enough capacity
            if remaining_capacity[volunteer.volunteer_id] >= recipient.num_items:
                # Get the distance for this volunteer-recipient pair
                distance = distance_matrix.get((volunteer.volunteer_id, recipient.recipient_id))
                if distance is not None:
                    pickup_id = volunteer_pickup[volunteer.volunteer_id]
                    valid_assignments.append((volunteer, distance, pickup_id))
        
        # Sort by distance (ascending)
        valid_assignments.sort(key=lambda x: x[1])
        
        # Assign to closest volunteer with capacity
        if valid_assignments:
            best_volunteer, _, pickup_id = valid_assignments[0]
            
            # Assign recipient to best volunteer
            assignments.append((best_volunteer.volunteer_id, recipient.recipient_id, pickup_id))
            assignment_map[best_volunteer.volunteer_id].append(recipient.recipient_id)
            assigned_recipients.add(recipient.recipient_id)
            
            # Update remaining capacity
            remaining_capacity[best_volunteer.volunteer_id] -= recipient.num_items
            
            # Debug output
            print(f"Assigned recipient {recipient.recipient_id} ({recipient.num_items} boxes) to volunteer {best_volunteer.volunteer_id} (remaining capacity: {remaining_capacity[best_volunteer.volunteer_id]})")
        else:
            print(f"Could not assign recipient {recipient.recipient_id} ({recipient.num_items} boxes) - no volunteer with enough capacity")
    
    # Second pass: try to assign any remaining recipients to any volunteer with capacity
    unassigned_recipients = [r for r in recipients if r.recipient_id not in assigned_recipients]
    if unassigned_recipients:
        print(f"\nAttempting to assign {len(unassigned_recipients)} remaining recipients...")
        
        # Sort by number of boxes (ascending) to fit smaller ones first
        unassigned_recipients.sort(key=lambda r: r.num_items)
        
        for recipient in unassigned_recipients:
            # Sort volunteers by remaining capacity (descending)
            sorted_volunteers = sorted(volunteers, key=lambda v: remaining_capacity[v.volunteer_id], reverse=True)
            
            # Find first volunteer with enough capacity
            best_volunteer = None
            for volunteer in sorted_volunteers:
                if remaining_capacity[volunteer.volunteer_id] >= recipient.num_items:
                    best_volunteer = volunteer
                    pickup_id = volunteer_pickup[volunteer.volunteer_id]
                    break
            
            # Only assign if we found a volunteer with capacity
            if best_volunteer is not None:
                # Assign recipient to best volunteer
                assignments.append((best_volunteer.volunteer_id, recipient.recipient_id, pickup_id))
                assignment_map[best_volunteer.volunteer_id].append(recipient.recipient_id)
                assigned_recipients.add(recipient.recipient_id)
                
                # Update remaining capacity
                remaining_capacity[best_volunteer.volunteer_id] -= recipient.num_items
                
                print(f"Assigned recipient {recipient.recipient_id} ({recipient.num_items} boxes) to volunteer {best_volunteer.volunteer_id} (remaining capacity: {remaining_capacity[best_volunteer.volunteer_id]})")
            else:
                print(f"Could not assign recipient {recipient.recipient_id} ({recipient.num_items} boxes) - no volunteer with enough capacity")
    
    # Third pass: try to distribute assignments to use more volunteers
    # This helps when we have volunteers with the same coordinates
    used_volunteers = [v_id for v_id, r_ids in assignment_map.items() if r_ids]
    unused_volunteers = [v.volunteer_id for v in volunteers if v.volunteer_id not in used_volunteers]
    
    if unused_volunteers and used_volunteers:
        print(f"\nRedistributing assignments to use more volunteers...")
        print(f"Currently using {len(used_volunteers)} of {len(volunteers)} volunteers")
        
        # For each unused volunteer, try to take some assignments from a used volunteer
        for unused_vol_id in unused_volunteers:
            # Find the unused volunteer
            unused_vol = next(v for v in volunteers if v.volunteer_id == unused_vol_id)
            unused_vol_pickup = volunteer_pickup[unused_vol_id]
            
            # Find volunteers with the most assignments
            used_vols_with_assignments = [(v_id, assignment_map[v_id]) for v_id in used_volunteers if assignment_map[v_id]]
            used_vols_with_assignments.sort(key=lambda x: len(x[1]), reverse=True)
            
            # Try to take assignments from the volunteer with the most assignments
            for used_vol_id, recipient_ids in used_vols_with_assignments:
                if not recipient_ids:  # Skip if no assignments left
                    continue
                    
                # Find the used volunteer
                used_vol = next(v for v in volunteers if v.volunteer_id == used_vol_id)
                
                # Check if they have the same pickup location (to minimize distance changes)
                if volunteer_pickup[used_vol_id] == unused_vol_pickup:
                    # Take one recipient from this volunteer
                    recipient_id = recipient_ids[0]  # Take the first recipient
                    recipient = next(r for r in recipients if r.recipient_id == recipient_id)
                    
                    # Check if the unused volunteer has capacity
                    if remaining_capacity[unused_vol_id] >= recipient.num_items:
                        # Remove from current volunteer
                        assignment_map[used_vol_id].remove(recipient_id)
                        remaining_capacity[used_vol_id] += recipient.num_items
                        
                        # Assign to unused volunteer
                        assignment_map[unused_vol_id].append(recipient_id)
                        remaining_capacity[unused_vol_id] -= recipient.num_items
                        
                        # Update assignments list
                        # First remove the old assignment
                        assignments = [(v, r, p) for v, r, p in assignments if not (v == used_vol_id and r == recipient_id)]
                        # Then add the new assignment
                        assignments.append((unused_vol_id, recipient_id, unused_vol_pickup))
                        
                        print(f"Moved recipient {recipient_id} from volunteer {used_vol_id} to volunteer {unused_vol_id}")
                        
                        # Mark this volunteer as used now
                        used_volunteers.append(unused_vol_id)
                        break  # Move to next unused volunteer
    
    # Final count of volunteers used
    final_used_volunteers = [v_id for v_id, r_ids in assignment_map.items() if r_ids]
    print(f"\nAfter redistribution: Using {len(final_used_volunteers)} of {len(volunteers)} volunteers")
    
    # Print assignment summary
    print(f"\nAssignment summary:")
    print(f"Total recipients assigned: {len(assigned_recipients)} of {len(recipients)}")
    print(f"Total volunteers used: {len([v for v, r in assignment_map.items() if r])} of {len(volunteers)}")
    
    # Check for empty assignments
    if not assignments:
        print("WARNING: No assignments were made! Using admin assignments as fallback.")
        return None
    
    # Calculate volunteer box counts
    volunteer_box_counts = {}
    for vol_id, rec_ids in assignment_map.items():
        total_boxes = sum(r.num_items for r in recipients if r.recipient_id in rec_ids)
        volunteer_box_counts[vol_id] = total_boxes
    
    # Create optimized data dictionary
    opt_data = {
        'assignments': assignments,
        'volunteers': volunteers,
        'recipients': recipients,
        'assignment_map': dict(assignment_map),  # Convert defaultdict to regular dict
        'pickups': pickups,
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
