import folium
import math
import webbrowser
import numpy as np
import os

def visualize_assignments(data, title="Assignment Map", config=None, save_path=None, show=True, save_report=True):
    """
    Visualize volunteer-recipient assignments using a Folium map, including pickup locations.
    Shows the complete route: volunteer -> pickup -> recipients -> volunteer.
    
    Args:
        data (dict or object): Data containing volunteers, recipients, assignments, and pickups.
            Expected keys for dict (from first function):
                - 'volunteers': List of volunteer objects with volunteer_id, latitude, longitude, car_size
                - 'recipients': List of recipient objects with recipient_ids, latitude, longitude, num_items
                - 'pickups': List of pickup objects with location_id, latitude, longitude, num_items
                - 'assignments': List of (volunteer_id, recipient_id, pickup_id) tuples
                - 'assignment_map': Dict mapping volunteer_id to list of recipient_ids
                - 'volunteer_box_counts': Dict mapping volunteer_id to number of boxes
                - 'stats': Optional dict with statistics
                - 'save_report': Optional bool to save report
            For object (from second/third functions):
                - Attributes: volunteers, recipients, assignments, pickup_assignments, assignment_map
                - volunteers: List of objects with latitude, longitude, car_size, volunteer_id (optional)
                - recipients: List of objects with latitude, longitude, num_items, recipient_id (optional)
                - pickups: List of objects with latitude, longitude, num_items, location_id (optional)
                - assignments: List of (volunteer_idx, recipient_idx) tuples or similar
        title (str): Title for the map
        config (dict, optional): Configuration options
            - show_volunteers (bool): Show volunteer markers
            - show_recipients (bool): Show recipient markers
            - show_pickups (bool): Show pickup markers
            - show_routes (bool): Show routes
            - show_stats (bool): Show statistics
            - volunteer_color (str): Color for volunteer markers
            - recipient_color (str): Color for recipient markers
            - pickup_color (str): Color for pickup markers
            - route_color (str): Color for routes
            - zoom_level (int): Initial zoom level
        save_path (str, optional): Path to save HTML file
        show (bool): Whether to display the map
    
    Returns:
        folium.Map or None: The map object if show=True, else None
    """
    # Default configuration
    default_config = {
        'show_volunteers': True,
        'show_recipients': True,
        'show_pickups': True,
        'show_routes': True,
        'show_stats': True,
        'volunteer_color': 'blue',
        'recipient_color': 'red',
        'pickup_color': 'green',
        'route_color': 'purple',
        'zoom_level': 12
    }
    
    # Merge provided config with defaults
    config = config or {}
    for key, value in default_config.items():
        if key not in config:
            config[key] = value
    
    # Normalize data input
    if isinstance(data, dict):
        # First function format
        volunteers = data.get('volunteers', [])
        recipients = data.get('recipients', [])
        pickups = data.get('pickups', [])
        assignments = data.get('assignments', [])
        assignment_map = data.get('assignment_map', {})
        volunteer_box_counts = data.get('volunteer_box_counts', {})
        stats = data.get('stats', {})
    else:
        # Second/Third function format
        volunteers = getattr(data, 'volunteers', []) or getattr(data, 'env', None).volunteers
        recipients = getattr(data, 'recipients', []) or getattr(data, 'env', None).recipients
        pickups = getattr(data, 'pickups', []) or []
        assignments = getattr(data, 'assignments', [])
        assignment_map = getattr(data, 'assignment_map', {})
        pickup_assignments = getattr(data, 'pickup_assignments', {}) or getattr(data, 'volunteer_pickup_assignments', {})
        volunteer_box_counts = {}
        stats = {}
        
        # Convert assignments to common format if needed
        if assignments and isinstance(assignments[0], tuple) and len(assignments[0]) == 2:
            # Second function format: (volunteer_idx, recipient_idx)
            new_assignments = []
            for v_idx, r_idx in assignments:
                v_id = getattr(volunteers[v_idx], 'volunteer_id', v_idx)
                r_id = getattr(recipients[r_idx], 'recipient_id', r_idx)
                p_id = pickup_assignments.get(v_id)
                new_assignments.append((v_id, r_id, p_id))
            assignments = new_assignments
    
    # Check if there's anything to visualize
    if not any(assignment_map.values()) and not assignments:
        print("No assignments to visualize!")
        return None
    
    # Collect all coordinates for centering
    all_coords = []
    for v in volunteers:
        lat = getattr(v, 'latitude', v[0] if isinstance(v, (list, tuple)) else 0)
        lon = getattr(v, 'longitude', v[1] if isinstance(v, (list, tuple)) else 0)
        if lat and lon:
            all_coords.append((lat, lon))
    for r in recipients:
        lat = getattr(r, 'latitude', r[0] if isinstance(r, (list, tuple)) else 0)
        lon = getattr(r, 'longitude', r[1] if isinstance(r, (list, tuple)) else 0)
        if lat and lon:
            all_coords.append((lat, lon))
    for p in pickups:
        lat = getattr(p, 'latitude', 0)
        lon = getattr(p, 'longitude', 0)
        if lat and lon:
            all_coords.append((lat, lon))
    
    # Calculate center point
    if all_coords:
        center_lat = sum(c[0] for c in all_coords) / len(all_coords)
        center_lon = sum(c[1] for c in all_coords) / len(all_coords)
    else:
        center_lat, center_lon = 39.8283, -98.5795  # Default to central US
    
    # Create map
    m = folium.Map(location=[center_lat, center_lon], zoom_start=config['zoom_level'])
    
    # Create feature groups
    pickup_group = folium.FeatureGroup(name="Pickup Locations")
    volunteer_groups = {}
    
    # Helper function to calculate Haversine distance
    def haversine_distance(lat1, lon1, lat2, lon2):
        R = 6371  # Earth's radius in km
        lat1, lon1, lat2, lon2 = map(math.radians, [lat1, lon1, lat2, lon2])
        dlat = lat2 - lat1
        dlon = lon2 - lon1
        a = math.sin(dlat/2)**2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon/2)**2
        c = 2 * math.asin(math.sqrt(a))
        return R * c
    
    # Helper function to simulate route calculation (since _calculate_route_travel_time is not provided)
    def calculate_route(volunteer_idx, recipient_indices, pickup_idx=None):
        # Mock route: volunteer -> pickup -> recipients -> volunteer
        coords = []
        volunteer = volunteers[volunteer_idx]
        v_lat = getattr(volunteer, 'latitude', volunteer[0] if isinstance(volunteer, (list, tuple)) else 0)
        v_lon = getattr(volunteer, 'longitude', volunteer[1] if isinstance(volunteer, (list, tuple)) else 0)
        coords.append([v_lat, v_lon])
        
        total_distance = 0
        if pickup_idx is not None:
            pickup = pickups[pickup_idx]
            p_lat = getattr(pickup, 'latitude', 0)
            p_lon = getattr(pickup, 'longitude', 0)
            coords.append([p_lat, p_lon])
            total_distance += haversine_distance(v_lat, v_lon, p_lat, p_lon)
        
        prev_lat, prev_lon = coords[-1]
        for r_idx in recipient_indices:
            recipient = recipients[r_idx]
            r_lat = getattr(recipient, 'latitude', recipient[0] if isinstance(recipient, (list, tuple)) else 0)
            r_lon = getattr(recipient, 'longitude', recipient[1] if isinstance(recipient, (list, tuple)) else 0)
            coords.append([r_lat, r_lon])
            total_distance += haversine_distance(prev_lat, prev_lon, r_lat, r_lon)
            prev_lat, prev_lon = r_lat, r_lon
        
        # Return to volunteer
        coords.append([v_lat, v_lon])
        total_distance += haversine_distance(prev_lat, prev_lon, v_lat, v_lon)
        
        travel_time = total_distance * 2  # Rough estimate: 2 minutes per km
        return travel_time, total_distance, coords
    
    # Add pickup locations
    if config['show_pickups']:
        for p_idx, pickup in enumerate(pickups):
            p_id = getattr(pickup, 'location_id', p_idx)
            p_lat = getattr(pickup, 'latitude', 0)
            p_lon = getattr(pickup, 'longitude', 0)
            p_items = getattr(pickup, 'num_items', 0)
            
            # Calculate boxes needed
            boxes_needed = 0
            for v_id, r_ids in assignment_map.items():
                if not r_ids:
                    continue
                v_idx = next((i for i, v in enumerate(volunteers) if getattr(v, 'volunteer_id', i) == v_id), None)
                if v_idx is None:
                    continue
                p_assigned = None
                for v, r, p in assignments:
                    if v == v_id:
                        p_assigned = p
                        break
                if p_assigned == p_id:
                    for r_id in r_ids:
                        r_idx = next(i for i, r in enumerate(recipients) if getattr(r, 'recipient_id', i) == r_id)
                        boxes_needed += getattr(recipients[r_idx], 'num_items', 0)
            
            popup_content = f"""
            <b>Pickup Location {p_id}</b><br>
            Available Items: {p_items}<br>
            <b>Boxes Needed: {boxes_needed}</b><br>
            Location: ({p_lat:.4f}, {p_lon:.4f})
            """
            
            folium.Marker(
                location=[p_lat, p_lon],
                popup=folium.Popup(popup_content, max_width=300),
                icon=folium.Icon(color=config['pickup_color'], icon='shopping-cart', prefix='fa'),
                tooltip=f"Pickup {p_id}"
            ).add_to(pickup_group)
        
        pickup_group.add_to(m)
    
    # Process assignments
    if config['show_volunteers'] or config['show_recipients'] or config['show_routes']:
        for v_id, r_ids in assignment_map.items():
            if not r_ids:
                continue
            
            # Get volunteer details
            v_idx = next((i for i, v in enumerate(volunteers) if getattr(v, 'volunteer_id', i) == v_id), None)
            if v_idx is None:
                continue
            volunteer = volunteers[v_idx]
            v_lat = getattr(volunteer, 'latitude', volunteer[0] if isinstance(volunteer, (list, tuple)) else 0)
            v_lon = getattr(volunteer, 'longitude', volunteer[1] if isinstance(volunteer, (list, tuple)) else 0)
            car_size_raw = getattr(volunteer, 'car_size', 6)
            
            # Convert car_size to integer, with fallback
            try:
                car_size = int(car_size_raw)
            except (ValueError, TypeError):
                car_size = 6  # Default if conversion fails
            
            # Get pickup assignment
            p_id = None
            for v, r, p in assignments:
                if v == v_id:
                    p_id = p
                    break
            if p_id is None:
                continue
            
            p_idx = next((i for i, p in enumerate(pickups) if getattr(p, 'location_id', i) == p_id), None)
            if p_idx is None:
                continue
            pickup = pickups[p_idx]
            p_lat = getattr(pickup, 'latitude', 0)
            p_lon = getattr(pickup, 'longitude', 0)
            
            # Get recipient indices
            r_indices = []
            for r_id in r_ids:
                r_idx = next((i for i, r in enumerate(recipients) if getattr(r, 'recipient_id', i) == r_id), None)
                if r_idx is not None:
                    r_indices.append(r_idx)
            
            # Calculate statistics
            total_boxes = sum(getattr(recipients[r_idx], 'num_items', 0) for r_idx in r_indices)
            utilization = (total_boxes / car_size * 100) if car_size > 0 else 0
            
            # Calculate route
            travel_time, total_distance, route_coords = calculate_route(v_idx, r_indices, p_idx)
            
            # Create feature group
            group = folium.FeatureGroup(name=f"Volunteer {v_id}")
            volunteer_groups[v_id] = group
            
            # Add volunteer marker
            if config['show_volunteers']:
                popup_content = f"""
                <b>Volunteer {v_id}</b><br>
                Car Capacity: {car_size} boxes<br>
                Assigned: {total_boxes} boxes ({utilization:.1f}%)<br>
                Pickup Location: {p_id}<br>
                Recipients: {len(r_ids)}<br>
                Est. Travel: {travel_time:.1f} min ({total_distance:.1f} km)
                """
                
                folium.Marker(
                    location=[v_lat, v_lon],
                    popup=folium.Popup(popup_content, max_width=300),
                    icon=folium.Icon(color=config['volunteer_color'], icon='user', prefix='fa'),
                    tooltip=f"Volunteer {v_id}"
                ).add_to(group)
            
            # Add recipient markers
            if config['show_recipients']:
                for r_idx in r_indices:
                    recipient = recipients[r_idx]
                    r_id = getattr(recipient, 'recipient_id', r_idx)
                    r_lat = getattr(recipient, 'latitude', recipient[0] if isinstance(recipient, (list, tuple)) else 0)
                    r_lon = getattr(recipient, 'longitude', recipient[1] if isinstance(recipient, (list, tuple)) else 0)
                    num_items = getattr(recipient, 'num_items', 0)
                    
                    popup_content = f"""
                    <b>Recipient {r_id}</b><br>
                    Boxes: {num_items}<br>
                    Assigned to: Volunteer {v_id}<br>
                    Pickup: {p_id}
                    """
                    
                    folium.Marker(
                        location=[r_lat, r_lon],
                        popup=folium.Popup(popup_content, max_width=300),
                        icon=folium.Icon(color=config['recipient_color'], icon='home', prefix='fa'),
                        tooltip=f"Recipient {r_id}"
                    ).add_to(group)
            
            # Draw routes
            if config['show_routes']:
                # Volunteer to Pickup
                folium.PolyLine(
                    locations=[[v_lat, v_lon], [p_lat, p_lon]],
                    color='blue',
                    weight=3,
                    opacity=0.8,
                    tooltip=f"To Pickup: {haversine_distance(v_lat, v_lon, p_lat, p_lon):.1f} km"
                ).add_to(group)
                
                # Delivery route (pickup -> recipients)
                recipient_route = route_coords[1:-1]  # Skip volunteer and return
                for i in range(len(recipient_route) - 1):
                    start_lat, start_lon = recipient_route[i]
                    end_lat, end_lon = recipient_route[i+1]
                    segment_distance = haversine_distance(start_lat, start_lon, end_lat, end_lon)
                    segment_type = "Pickup to Recipient" if i == 0 else "Recipient to Recipient"
                    
                    # Draw line
                    folium.PolyLine(
                        locations=[[start_lat, start_lon], [end_lat, end_lon]],
                        color='green',
                        weight=3,
                        opacity=0.8,
                        tooltip=f"{segment_type}: {segment_distance:.1f} km"
                    ).add_to(group)
                    
                    # Add arrow
                    midpoint_lat = (start_lat + end_lat) / 2
                    midpoint_lon = (start_lon + end_lon) / 2
                    y = math.sin(math.radians(end_lon - start_lon)) * math.cos(math.radians(end_lat))
                    x = math.cos(math.radians(start_lat)) * math.sin(math.radians(end_lat)) - \
                        math.sin(math.radians(start_lat)) * math.cos(math.radians(end_lat)) * math.cos(math.radians(end_lon - start_lon))
                    bearing = math.degrees(math.atan2(y, x))
                    bearing = (bearing + 360) % 360
                    
                    folium.Marker(
                        [midpoint_lat, midpoint_lon],
                        icon=folium.features.DivIcon(
                            icon_size=(20, 20),
                            icon_anchor=(10, 10),
                            html=f'<div style="font-size: 12pt; color: green; transform: rotate({bearing}deg);">➤</div>'
                        )
                    ).add_to(group)
                
                # Return to volunteer
                last_lat, last_lon = recipient_route[-1]
                folium.PolyLine(
                    locations=[[last_lat, last_lon], [v_lat, v_lon]],
                    color='purple',
                    weight=2,
                    opacity=0.6,
                    tooltip=f"Return Trip: {haversine_distance(last_lat, last_lon, v_lat, v_lon):.1f} km"
                ).add_to(group)
            
            group.add_to(m)
    
    # Calculate statistics
    total_volunteers = len([vid for vid, rids in assignment_map.items() if rids])
    total_recipients = sum(len(rids) for rids in assignment_map.values())
    route_lengths = []
    total_distance = 0
    utilizations = []
    for v_id, r_ids in assignment_map.items():
        if not r_ids:
            continue
        v_idx = next((i for i, v in enumerate(volunteers) if getattr(v, 'volunteer_id', i) == v_id), None)
        if v_idx is None:
            continue
        r_indices = [next(i for i, r in enumerate(recipients) if getattr(r, 'recipient_id', i) == r_id) for r_id in r_ids]
        p_id = next((p for v, r, p in assignments if v == v_id), None)
        p_idx = next((i for i, p in enumerate(pickups) if getattr(p, 'location_id', i) == p_id), None) if p_id else None
        if p_idx is None:
            continue
        _, route_dist, _ = calculate_route(v_idx, r_indices, p_idx)
        route_lengths.append(route_dist)
        total_distance += route_dist
        total_boxes = sum(getattr(recipients[r_idx], 'num_items', 0) for r_idx in r_indices)
        car_size_raw = getattr(volunteers[v_idx], 'car_size', 6)
        try:
            car_size = int(car_size_raw)
        except (ValueError, TypeError):
            car_size = 6  # Default if conversion fails
        utilizations.append(total_boxes / car_size * 100 if car_size > 0 else 0)
    
    avg_route_length = sum(route_lengths) / len(route_lengths) if route_lengths else 0
    avg_utilization = sum(utilizations) / len(utilizations) if utilizations else 0
    
    # Add statistics panel
    if config['show_stats']:
        stats_html = f'''
        <div id="stats-panel" style="position: fixed; bottom: 50px; right: 50px; z-index: 1001; background-color: white; padding: 14px; border: 2px solid #666; border-radius: 8px; display: none; min-width: 250px;">
            <h4 style="margin-top:0">Assignment Statistics</h4>
            <ul style="padding-left: 1.2em;">
                <li><b>Total Volunteers:</b> {total_volunteers}</li>
                <li><b>Total Recipients:</b> {total_recipients}</li>
                <li><b>Avg. Route Length:</b> {avg_route_length:.2f} km</li>
                <li><b>Total Distance:</b> {total_distance:.2f} km</li>
                <li><b>Avg. Utilization:</b> {avg_utilization:.1f}%</li>
            </ul>
            <button onclick="document.getElementById('stats-panel').style.display='none'" style="margin-top:8px;">Close</button>
        </div>
        <button id="toggle-stats-btn" style="position: fixed; bottom: 10px; right: 10px; z-index: 1002; background-color: #4CAF50; color: white; border: none; padding: 8px 16px; border-radius: 5px; cursor: pointer;">Show Statistics</button>
        '''
        
        js_code = '''
        <script>
        document.getElementById('toggle-stats-btn').onclick = function() {
            var panel = document.getElementById('stats-panel');
            panel.style.display = panel.style.display === 'none' ? 'block' : 'none';
        };
        function addToggleAllVolunteers() {
            var layerControl = document.getElementsByClassName('leaflet-control-layers-overlays')[0];
            if (!layerControl) { setTimeout(addToggleAllVolunteers, 500); return; }
            if (document.getElementById('toggle-all-volunteers-row')) return;
            var row = document.createElement('label');
            row.id = 'toggle-all-volunteers-row';
            row.style.display = 'block';
            row.style.cursor = 'pointer';
            row.style.marginTop = '8px';
            var cb = document.createElement('input');
            cb.type = 'checkbox';
            cb.checked = true;
            cb.style.marginRight = '6px';
            row.appendChild(cb);
            var text = document.createTextNode('Show All Volunteers');
            row.appendChild(text);
            layerControl.insertBefore(row, layerControl.firstChild);
            cb.onchange = function() {
                var allInputs = layerControl.querySelectorAll('input[type=checkbox]');
                for (var i = 0; i < allInputs.length; i++) {
                    var label = allInputs[i].parentElement;
                    if (label.textContent.trim().startsWith('Volunteer')) {
                        if (allInputs[i].checked !== cb.checked) {
                            allInputs[i].click();
                        }
                    }
                }
            };
        }
        setTimeout(addToggleAllVolunteers, 1000);
        </script>
        '''
        
        m.get_root().html.add_child(folium.Element(stats_html))
        m.get_root().html.add_child(folium.Element(js_code))
    
    # Add legend
    legend_html = f'''
    <div style="position: fixed; bottom: 50px; left: 50px; z-index: 1000; background-color: white; padding: 10px; border: 2px solid grey; border-radius: 5px;">
        <h4>Route Legend</h4>
        <div><i style="background: blue; width: 15px; height: 3px; display: inline-block;"></i> Volunteer to Pickup</div>
        <div><i style="background: green; width: 15px; height: 3px; display: inline-block;"></i> Optimized Delivery Route ➤</div>
        <div><i style="background: purple; width: 15px; height: 3px; display: inline-block;"></i> Return Trip</div>
        <div style="margin-top: 5px;">
            <i style="background: {config['volunteer_color']}; width: 10px; height: 10px; border-radius: 50%; display: inline-block;"></i> Volunteer
            <i style="background: {config['pickup_color']}; width: 10px; height: 10px; border-radius: 50%; display: inline-block; margin-left: 5px;"></i> Pickup
            <i style="background: {config['recipient_color']}; width: 10px; height: 10px; border-radius: 50%; display: inline-block; margin-left: 5px;"></i> Recipient
        </div>
    </div>
    '''
    m.get_root().html.add_child(folium.Element(legend_html))
    
    # Add title
    title_html = f'''
    <div style="position: fixed; top: 10px; left: 50%; transform: translateX(-50%); z-index: 1000; background-color: white; padding: 10px; border: 2px solid grey; border-radius: 5px;">
        <h3 style="margin: 0;">{title}</h3>
    </div>
    '''
    m.get_root().html.add_child(folium.Element(title_html))
    
    # Add layer control
    folium.LayerControl().add_to(m)
    
    # Save or show
    if save_path and save_report:
        m.save(save_path)
        print(f"Map saved to {save_path}")
    
    if show:
        webbrowser.open(f'file://{os.path.abspath(save_path)}')
    
    return m