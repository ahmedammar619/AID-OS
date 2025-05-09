#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Main script for the AID-OS project.
Provides a command-line interface to run the various components.
Supports both Reinforcement Learning and Optimization-based approaches.
"""

import argparse
import os
import sys
import torch
import numpy as np
from datetime import datetime
import random

import warnings
warnings.filterwarnings("ignore", category=FutureWarning, module="sklearn")

# Import project modules
from data.db_config import DatabaseHandler
from env.delivery_env import DeliveryEnv
from clustering.dbscan_cluster import RecipientClusterer
from training.train_agent import AgentTrainer
from feedback.feedback_handler import FeedbackHandler
from assignment.assign_volunteers import VolunteerAssigner
from assignment.assign_volunteers_opt import VolunteerAssignerOpt
from assignment.compare_with_admin import get_admin_assignments, run_optimized_assignments, calculate_assignment_stats, compare_assignments


def setup_parser():
    """Set up command-line argument parser."""
    parser = argparse.ArgumentParser(description='AID-OS: Volunteer Assignment Optimization System')
    
    # Add subparsers for different commands
    subparsers = parser.add_subparsers(dest='command', help='Command to run')
    
    # Initialize database command
    init_parser = subparsers.add_parser('init-db', help='Initialize database tables')
    
    # Train command (RL approach)
    train_parser = subparsers.add_parser('train', help='Train the RL agent')
    train_parser.add_argument('--episodes', type=int, default=500, help='Number of episodes to train for')
    train_parser.add_argument('--steps', type=int, default=200, help='Maximum steps per episode')
    train_parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
    train_parser.add_argument('--checkpoint-dir', type=str, default='./hist/checkpoints', help='Directory to save checkpoints')
    train_parser.add_argument('--device', type=str, default='auto', help='Device to run on (cpu, cuda, auto)')
    
    # Generate assignments command (RL approach)
    assign_parser = subparsers.add_parser('assign-rl', help='Generate assignments using trained RL agent')
    assign_parser.add_argument('--agent', type=str, required=True, help='Path to trained agent checkpoint')
    assign_parser.add_argument('--deterministic', action='store_true', help='Use deterministic policy')
    assign_parser.add_argument('--visualize', action='store_true', help='Visualize assignments')
    assign_parser.add_argument('--save-report', action='store_true', help='Save assignment report')
    assign_parser.add_argument('--export-csv', action='store_true', help='Export assignments to CSV')
    
    # Generate assignments command (Optimization approach)
    assign_opt_parser = subparsers.add_parser('assign-opt', help='Generate assignments using optimization')
    assign_opt_parser.add_argument('--visualize', action='store_true', help='Visualize assignments')
    assign_opt_parser.add_argument('--save-report', action='store_true', help='Save assignment report')
    assign_opt_parser.add_argument('--export-csv', action='store_true', help='Export assignments to CSV')
    assign_opt_parser.add_argument('--output-dir', type=str, default='./hist/output', help='Directory to save output files')
    
    # Run pipeline command (RL approach)
    pipeline_parser = subparsers.add_parser('pipeline-rl', help='Run the complete RL pipeline')
    pipeline_parser.add_argument('--agent', type=str, help='Path to trained agent checkpoint')
    pipeline_parser.add_argument('--output-dir', type=str, default='./hist/output', help='Directory to save output files')
    
    # Run pipeline command (Optimization approach)
    pipeline_opt_parser = subparsers.add_parser('pipeline-opt', help='Run the complete optimization pipeline')
    pipeline_opt_parser.add_argument('--output-dir', type=str, default='./hist/output', help='Directory to save output files')
    
    # Compare approaches command
    compare_parser = subparsers.add_parser('compare', help='Compare RL and optimization approaches')
    compare_parser.add_argument('--agent', type=str, help='Path to trained agent checkpoint')
    compare_parser.add_argument('--output-dir', type=str, default='./hist/output', help='Directory to save output files')
    
    # Compare with admin assignments command
    compare_admin_parser = subparsers.add_parser('compare-admin', help='Compare admin assignments with optimized assignments')
    compare_admin_parser.add_argument('--output-dir', type=str, default='./hist/output', help='Directory to save output files')
    compare_admin_parser.add_argument('--show-maps', action='store_true', help='Display interactive maps for comparison')
    compare_admin_parser.add_argument('--print-stats', action='store_true', help='Print statistics for comparison')
    
    # Clustering command
    cluster_parser = subparsers.add_parser('map', help='Cluster recipients')
    cluster_parser.add_argument('--eps', type=float, default=0.00005, help='DBSCAN epsilon parameter')
    cluster_parser.add_argument('--min-samples', type=int, default=3, help='DBSCAN min_samples parameter')
    cluster_parser.add_argument('--min-cluster-size', type=int, default=2, help='DBSCAN min_cluster_size parameter')
    
    # Feedback command
    feedback_parser = subparsers.add_parser('feedback', help='Handle admin feedback')
    feedback_parser.add_argument('--generate-report', action='store_true', help='Generate feedback report')
    feedback_parser.add_argument('--load', type=str, help='Load feedback from file')
    feedback_parser.add_argument('--save', type=str, help='Save feedback to file')
    
    return parser

def view_map(args):
    """View the recipient map."""
    
    print("Viewing recipient map...")

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
    # Combine into a list of tuples
    volunteer_coords = np.array([[v.latitude, v.longitude] for v in db.get_all_volunteers()])
    # Initialize and fit the clusterer
    clusterer = RecipientClusterer(
        min_cluster_size=args.min_cluster_size,
        cluster_selection_epsilon=args.eps,
        min_samples=args.min_samples
    )
    labels = clusterer.fit(all_coords)

    # Get pickups
    pickups = db.get_all_pickups()
    pickup_coords = np.array([[p.latitude, p.longitude] for p in pickups])
    
    # Visualize the clusters
    output_path = os.path.join("./hist/output", "cluster_map.html")
    clusterer.visualize_clusters(
        all_coords, 
        all_ids, 
        recipient_boxes,
        volunteer_coords,
        save_path=output_path,
        pickup_coords=pickup_coords
    )
    print("Map visualization complete!")

def train_agent(args):
    """Train the RL agent."""
    print("Training RL agent...")
    

    # Create database handler
    db_handler = DatabaseHandler()
    
    max_steps = 300

    # Create environment
    env = DeliveryEnv(db_handler=db_handler, max_steps=max_steps, use_clustering=True, cluster_eps=0.00005)

    # Create trainer
    trainer = AgentTrainer(
        state_dim=env.observation_space.shape[0],
        action_dim=env.action_space.n,
        db_handler=db_handler,
        device="cuda" if torch.cuda.is_available() else "cpu",
        actor_lr=0.0001,
        critic_lr=0.0002,
        gamma=0.95
    )

    
    # Training loops
    stats = trainer.train(
        env=env,
        num_episodes=4000,
        max_steps=max_steps,
        print_interval=10,
        checkpoint_interval=500,
        agent_num_updates=3
    )
    
    # # Create environment
    # env2 = DeliveryEnv(db_handler=db_handler, max_steps=max_steps, use_clustering=True, cluster_eps=0.00005)
    
    # # Create trainer
    # trainer2 = AgentTrainer(
    #     state_dim=env.observation_space.shape[0],
    #     action_dim=env.action_space.n,
    #     db_handler=db_handler,
    #     device="cuda" if torch.cuda.is_available() else "cpu",
    #     actor_lr=0.0001,
    #     critic_lr=0.0002,
    #     gamma=0.95
    # )
    # # Training loop
    # stats2 = trainer2.train(
    #     env=env2,
    #     num_episodes=502,
    #     max_steps=max_steps,
    #     print_interval=10,
    #     checkpoint_interval=1000,
    #     agent_num_updates=10
    # )
    
    print("Training complete!")

def run_pipeline_rl(args):
    """Run the complete RL assignment pipeline."""
    print("Running complete RL assignment pipeline...")
    import webbrowser
    import glob
    
    # Create assigner
    assigner = VolunteerAssigner(output_dir=args.output_dir)
    
    # Run pipeline
    success = assigner.run_complete_pipeline(
        export_csv=True,
        save_visualizations=True,
        save_report=True,
        agent_path='./hist/checkpoints/checkpoint_final' if not hasattr(args, 'agent') or not args.agent else args.agent
    )
    
    if success:
        print("RL pipeline completed successfully!")
        # Try to open the most recent assignment map or cluster map
        output_dir = args.output_dir if hasattr(args, 'output_dir') else './hist/output'
        # Try assignment_map first, then cluster_map
        html_files = sorted(glob.glob(os.path.join(output_dir, 'assignment_map_*.html')), reverse=True)
        if not html_files:
            html_files = sorted(glob.glob(os.path.join(output_dir, 'cluster_map.html')), reverse=True)
        if html_files:
            print(f"Opening map: {html_files[0]}")
            webbrowser.open(f'file://{os.path.abspath(html_files[0])}')
        else:
            print("No map HTML file found to open.")
    else:
        print("RL pipeline failed!")

def run_pipeline_opt(args):
    """Run the complete optimization assignment pipeline."""
    import webbrowser, os
    from datetime import datetime
    print("Displaying assignment map...")
    # Generate assignments only
    assigner = VolunteerAssignerOpt(output_dir=args.output_dir)
    success = assigner.generate_assignments()
    if success:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        map_path = os.path.join(args.output_dir, f"assignment_map_{timestamp}.html")
        assigner.visualize_assignments(save_path=map_path, show=False)
        print(f"Opening map: {map_path}")
        webbrowser.open(f'file://{os.path.abspath(map_path)}')
    else:
        print("Optimization pipeline failed!")

def handle_feedback(args):
    """Handle admin feedback."""
    print("Handling admin feedback...")
    
    # Create feedback handler
    handler = FeedbackHandler()
    
    # Load feedback if requested
    if args.load:
        success = handler.load_feedback(args.load)
        if success:
            print(f"Feedback loaded from {args.load}")
        else:
            print(f"Failed to load feedback from {args.load}")
    
    # Generate report if requested
    if args.generate_report:
        report = handler.generate_feedback_report()
        print("\nFeedback Report:")
        print(report)
        
        # Save report
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filepath = os.path.join("./hist/output", f"feedback_report_{timestamp}.md")
        
        with open(filepath, 'w') as f:
            f.write(report)
        
        print(f"Report saved to {filepath}")
    
    # Save feedback if requested
    if args.save:
        filepath = handler.save_feedback(args.save)
        print(f"Feedback saved to {filepath}")
    
    print("Feedback handling complete!")

def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def assign_rl(args):
    """Generate assignments using the RL approach."""
    print("Generating assignments using RL...")
    
    # Create assigner
    assigner = VolunteerAssigner(agent_path=args.agent)
    
    # Generate assignments
    success = assigner.generate_assignments(deterministic=args.deterministic)
    
    if success:
        print("Assignments generated successfully!")
        
        # Save to database
        assigner.save_assignments_to_db()
        
        # Export to CSV if requested
        if args.export_csv:
            assigner.export_assignments_to_csv()
        
        # Visualize if requested
        if args.visualize:
            assigner.visualize_assignments()
            assigner.visualize_volunteer_load()
        
        # Save report if requested
        if args.save_report:
            report = assigner.generate_assignment_report()
            assigner.save_report(report)
    else:
        print("Failed to generate assignments!")

def assign_opt(args):
    """Generate assignments using the optimization approach."""
    print("Generating assignments using optimization...")
    
    # Create assigner
    assigner = VolunteerAssignerOpt(output_dir=args.output_dir)
    
    # Generate assignments
    success = assigner.generate_assignments()
    
    if success:
        print("Assignments generated successfully!")
        
        # Save to database
        assigner.save_assignments_to_db()
        
        # Export to CSV if requested
        if args.export_csv:
            assigner.export_assignments_to_csv()
        
        # Visualize if requested
        if args.visualize:
            assigner.visualize_assignments()
            assigner.visualize_volunteer_load()
        
        # Save report if requested
        if args.save_report:
            report = assigner.generate_assignment_report()
            assigner.save_report(report)
    else:
        print("Failed to generate assignments!")

def compare_approaches(args):
    """Compare RL and optimization approaches."""
    print("Comparing RL and optimization approaches...")
    import pandas as pd
    import matplotlib.pyplot as plt
    from datetime import datetime
    
    # Create output directory if it doesn't exist
    output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)
    
    # Generate timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Run RL pipeline
    print("\n1. Running RL pipeline...")
    rl_assigner = VolunteerAssigner(
        agent_path=args.agent if hasattr(args, 'agent') and args.agent else './hist/checkpoints/checkpoint_final',
        output_dir=output_dir
    )
    rl_success = rl_assigner.generate_assignments(deterministic=True)
    
    if not rl_success:
        print("RL pipeline failed! Cannot continue comparison.")
        return
    
    # Export RL assignments
    rl_csv_path = rl_assigner.export_assignments_to_csv(f"assignments_rl_{timestamp}.csv")
    # Load and then remove RL CSV (no persistence)
    rl_df = pd.read_csv(rl_csv_path)
    os.remove(rl_csv_path)
    
    # Run optimization pipeline
    print("\n2. Running optimization pipeline...")
    opt_assigner = VolunteerAssignerOpt(output_dir=output_dir)
    opt_success = opt_assigner.generate_assignments()
    
    if not opt_success:
        print("Optimization pipeline failed! Cannot continue comparison.")
        return
    
    # Export optimization assignments
    opt_csv_path = opt_assigner.export_assignments_to_csv(f"assignments_opt_{timestamp}.csv")
    # Load and then remove Optimization CSV (no persistence)
    opt_df = pd.read_csv(opt_csv_path)
    os.remove(opt_csv_path)
    # DataFrames now ready for plotting/comparison
    
    # Calculate metrics for comparison
    rl_total_volunteers = len(rl_df['volunteer_id'].unique())
    opt_total_volunteers = len(opt_df['volunteer_id'].unique())
    
    # Handle different column names in RL and optimization CSVs
    rl_distance_col = 'distance' if 'distance' in rl_df.columns else 'distance_km'
    # Choose distance column for optimization (prefer total_route_km)
    if 'total_route_km' in opt_df.columns:
        opt_distance_col = 'total_route_km'
    else:
        opt_distance_col = 'distance_km' if 'distance_km' in opt_df.columns else 'distance'
    
    rl_total_distance = rl_df[rl_distance_col].sum()
    opt_total_distance = opt_df[opt_distance_col].sum()
    
    rl_avg_distance = rl_df[rl_distance_col].mean()
    opt_avg_distance = opt_df[opt_distance_col].mean()
    
    # Handle different column names for capacity and box counts
    rl_capacity_col = 'volunteer_capacity' if 'volunteer_capacity' in rl_df.columns else 'volunteer_car_size'
    rl_boxes_col = 'recipient_boxes' if 'recipient_boxes' in rl_df.columns else 'recipient_num_items'
    
    opt_capacity_col = 'volunteer_car_size' if 'volunteer_car_size' in opt_df.columns else 'volunteer_capacity'
    opt_boxes_col = 'recipient_num_items' if 'recipient_num_items' in opt_df.columns else 'recipient_boxes'
    
    # Calculate utilization
    rl_utilization = []
    for vol_id in rl_df['volunteer_id'].unique():
        vol_df = rl_df[rl_df['volunteer_id'] == vol_id]
        capacity = vol_df[rl_capacity_col].iloc[0]
        load = vol_df[rl_boxes_col].sum()
        rl_utilization.append(load / capacity * 100)
    
    opt_utilization = []
    for vol_id in opt_df['volunteer_id'].unique():
        vol_df = opt_df[opt_df['volunteer_id'] == vol_id]
        capacity = vol_df[opt_capacity_col].iloc[0]
        load = vol_df[opt_boxes_col].sum()
        opt_utilization.append(load / capacity * 100)
    
    rl_avg_utilization = sum(rl_utilization) / len(rl_utilization) if rl_utilization else 0
    opt_avg_utilization = sum(opt_utilization) / len(opt_utilization) if opt_utilization else 0
    
    # Generate comparison report
    report = "# RL vs. Optimization Comparison Report\n\n"
    report += f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n"
    
    report += "## Summary Metrics\n\n"
    report += "| Metric | RL Approach | Optimization Approach | Difference |\n"
    report += "|--------|------------|----------------------|------------|\n"
    report += f"| Total Volunteers | {rl_total_volunteers} | {opt_total_volunteers} | {opt_total_volunteers - rl_total_volunteers} |\n"
    report += f"| Total Distance (km) | {rl_total_distance:.2f} | {opt_total_distance:.2f} | {opt_total_distance - rl_total_distance:.2f} |\n"
    report += f"| Average Distance (km) | {rl_avg_distance:.2f} | {opt_avg_distance:.2f} | {opt_avg_distance - rl_avg_distance:.2f} |\n"
    report += f"| Average Utilization (%) | {rl_avg_utilization:.2f} | {opt_avg_utilization:.2f} | {opt_avg_utilization - rl_avg_utilization:.2f} |\n\n"
    
    # Create visualizations
    plt.figure(figsize=(12, 8))
    
    # 1. Volunteer count comparison
    plt.subplot(2, 2, 1)
    plt.bar(['RL', 'Optimization'], [rl_total_volunteers, opt_total_volunteers])
    plt.title('Number of Volunteers Used')
    plt.ylabel('Count')
    
    # 2. Total distance comparison
    plt.subplot(2, 2, 2)
    plt.bar(['RL', 'Optimization'], [rl_total_distance, opt_total_distance])
    plt.title('Total Distance (km)')
    plt.ylabel('Distance (km)')
    
    # 3. Average distance comparison
    plt.subplot(2, 2, 3)
    plt.bar(['RL', 'Optimization'], [rl_avg_distance, opt_avg_distance])
    plt.title('Average Distance per Assignment (km)')
    plt.ylabel('Distance (km)')
    
    # 4. Utilization comparison
    plt.subplot(2, 2, 4)
    plt.boxplot([rl_utilization, opt_utilization], labels=['RL', 'Optimization'])
    plt.title('Volunteer Utilization (%)')
    plt.ylabel('Utilization (%)')
    
    plt.tight_layout()
    
    # Save comparison plot
    plot_path = os.path.join(output_dir, f"comparison_{timestamp}.png")
    plt.savefig(plot_path)
    plt.close()
    
    report += f"![Comparison Plot]({plot_path})\n\n"
    
    # Save report
    # report_path = os.path.join(output_dir, f"comparison_report_{timestamp}.md")
    # with open(report_path, 'w') as f:
    #     f.write(report)
    
    # print(f"\nComparison completed! Report saved to {report_path}")
    print(f"Comparison plot saved to {plot_path}")

def compare_with_admin(args):
    """Compare admin assignments with optimized assignments.
    
    This function:
    1. Fetches current admin assignments from the delivery table
    2. Runs the optimization algorithm on the same dataset
    3. Compares the results (distance, utilization, etc.)
    4. Optionally displays interactive maps for visual comparison
    
    Args:
        args: Command line arguments
    """
    print("Comparing admin assignments with optimized assignments...")
    
    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Get admin assignments
    admin_data = get_admin_assignments()
    
    if not admin_data:
        print("No admin assignments found in the delivery table. Cannot proceed with comparison.")
        return
    
    # Calculate admin assignment statistics
    admin_stats = calculate_assignment_stats(admin_data)
    admin_data['stats'] = admin_stats
    
    # Print admin assignment statistics if requested
    if args.print_stats:
        print(f"\nAdmin Assignment Statistics:")
        print(f"Total Volunteers: {admin_stats['total_volunteers']}")
        print(f"Total Recipients: {admin_stats['total_recipients']}")
        print(f"Total Distance: {admin_stats['total_distance']:.2f} km")
        print(f"Average Route Length: {admin_stats['avg_route_length']:.2f} km")
        print(f"Average Utilization: {admin_stats['avg_utilization']:.1f}%")
    
    # Run optimization on the same dataset
    print(f"\nRunning optimization on the same dataset...")
    result = run_optimized_assignments(admin_data, show_maps=args.show_maps, output_dir=args.output_dir)
    
    if result:
        admin_stats, opt_stats, admin_map_path, opt_map_path = result
        
        # Display comparison between admin and optimized assignments
        if args.print_stats:
            compare_assignments(admin_stats, opt_stats)
    else:
        print("\nOptimization failed. Only showing admin assignments.")


def main():
    """Main function to parse arguments and run commands."""
    parser = setup_parser()
    args = parser.parse_args()
    
    # Set seed
    set_seed(41)
    
    # Check if a command was provided
    if not args.command:
        parser.print_help()
        return
    
    # Run the appropriate command
    if args.command == 'map':
        view_map(args)
    elif args.command == 'train':
        train_agent(args)
    elif args.command == 'assign-rl':
        assign_rl(args)
    elif args.command == 'assign-opt':
        assign_opt(args)
    elif args.command == 'pipeline-rl':
        run_pipeline_rl(args)
    elif args.command == 'pipeline-opt':
        run_pipeline_opt(args)
    elif args.command == 'compare':
        compare_approaches(args)
    elif args.command == 'compare-admin':
        compare_with_admin(args)
    elif args.command == 'feedback':
        handle_feedback(args)
    else:
        print(f"Unknown command: {args.command}")
        parser.print_help()

if __name__ == "__main__":
    main()
