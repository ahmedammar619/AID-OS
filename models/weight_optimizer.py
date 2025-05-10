#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Weight Optimizer for Volunteer Assignment System

This module uses a neural network approach to find the optimal weights for the
volunteer assignment optimization algorithm.
"""

import os
import sys
import time
import random
import numpy as np
import signal
from datetime import datetime
from collections import defaultdict
import tensorflow as tf
from tensorflow import keras
from keras import layers
import matplotlib.pyplot as plt

# Add parent directory to path to import from project modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from assignment.compare_with_admin import get_admin_assignments, run_optimized_assignments, compare_assignments

class TimeoutException(Exception):
    """Exception raised when a function execution times out."""
    pass

def timeout_handler(signum, frame):
    """Handler for SIGALRM signal."""
    raise TimeoutException("Function execution timed out")

class WeightOptimizer:
    """
    Neural network-based optimizer for finding the best weights for the volunteer assignment system.
    
    This class uses a combination of genetic algorithms and neural networks to explore
    the weight space and find the optimal weights that produce the best assignment results.
    """
    
    def __init__(self, output_dir="./hist/output"):
        """
        Initialize the weight optimizer.
        
        Args:
            output_dir (str): Directory to save output files.
        """
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        
        # Get admin assignments from the database
        self.admin_data = get_admin_assignments()
        if not self.admin_data:
            raise ValueError("No admin assignments found in the database.")
        
        # Calculate admin stats
        self.admin_stats = self.admin_data.get('stats', None)
        if not self.admin_stats:
            from assignment.compare_with_admin import calculate_assignment_stats
            self.admin_stats = calculate_assignment_stats(self.admin_data)
            self.admin_data['stats'] = self.admin_stats
        
        # Initialize weights and results storage
        self.best_weights = None
        self.best_score = float('inf')
        self.results_history = []
        
        # Weight parameter ranges
        self.weight_params = {
            'distance': (0.0, 20.0),
            'volunteer_count': (0.0, 20.0),
            'capacity_util': (0.0, 20.0),
            'history': (0.0, 10.0),
            'compact_routes': (0.0, 10.0),
            'clusters': (0.0, 10.0)
        }
        
        # Neural network for weight prediction
        self.model = self._build_model()
        
    def _build_model(self):
        """
        Build and compile the neural network model for weight optimization.
        
        Returns:
            keras.Model: Compiled neural network model.
        """
        model = keras.Sequential([
            layers.Dense(64, activation='relu', input_shape=(len(self.weight_params),)),
            layers.Dense(128, activation='relu'),
            layers.Dense(64, activation='relu'),
            layers.Dense(len(self.weight_params), activation='sigmoid')
        ])
        
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=0.001),
            loss='mse'
        )
        
        return model
    
    def _normalize_weights(self, weights_dict):
        """
        Normalize weights to the appropriate ranges.
        
        Args:
            weights_dict (dict): Dictionary of weight values.
            
        Returns:
            dict: Dictionary of normalized weight values.
        """
        normalized = {}
        for key, value in weights_dict.items():
            min_val, max_val = self.weight_params.get(key, (0.0, 1.0))
            normalized[key] = value * (max_val - min_val) + min_val
        
        return normalized
    
    def _denormalize_weights(self, normalized_weights):
        """
        Denormalize weights from [0,1] to their original ranges.
        
        Args:
            normalized_weights (np.array): Array of normalized weight values.
            
        Returns:
            dict: Dictionary of denormalized weight values.
        """
        weights_dict = {}
        for i, key in enumerate(self.weight_params.keys()):
            min_val, max_val = self.weight_params.get(key, (0.0, 1.0))
            weights_dict[key] = (normalized_weights[i] - min_val) / (max_val - min_val)
        
        return weights_dict
    
    def _evaluate_weights(self, weights, timeout=60):
        """
        Evaluate a set of weights by running the optimization and comparing with admin assignments.
        
        Args:
            weights (dict): Dictionary of weight values.
            timeout (int): Maximum time in seconds to allow for evaluation.
            
        Returns:
            float: Score representing how good the weights are (lower is better).
            list: Percentage changes for each metric.
        """
        # Skip evaluation if weights are too extreme (to avoid long-running optimizations)
        max_weight_value = 20.0
        for key, value in weights.items():
            if value > max_weight_value:
                print(f"  Skipping evaluation due to extreme weight value: {key}={value}")
                return float('inf'), [0, 0, 0, 0, 0]
        
        # Set up timeout
        signal.signal(signal.SIGALRM, timeout_handler)
        signal.alarm(timeout)
        
        try:
            # Run optimization with these weights
            result = run_optimized_assignments(
                self.admin_data, 
                show_maps=False, 
                output_dir=self.output_dir,
                save_report=False,
                custom_weights=weights
            )
            
            if not result:
                # If optimization failed, return a very high score
                return float('inf'), [0, 0, 0, 0, 0]
            
            # Get stats and compare
            admin_stats, opt_stats, _, _ = result
            pct_changes = compare_assignments(admin_stats, opt_stats)
            
            # Calculate score based on percentage changes
            # Negative values for total_volunteers and total_distance are good
            # Positive values for avg_utilization are good
            # For avg_route_length, it depends on the context, but generally negative is better
            
            # Define weights for each metric in the score calculation
            metric_weights = {
                'total_volunteers': 1.0,  # Weight for volunteer count change
                'total_distance': 2.0,    # Weight for total distance change
                'avg_route_length': 0.5,  # Weight for route length change
                'avg_utilization': 1.5    # Weight for utilization change
            }
            
            # Calculate weighted score
            score = (
                metric_weights['total_volunteers'] * pct_changes[0] +  # total_volunteers change
                metric_weights['total_distance'] * pct_changes[2] +    # total_distance change
                metric_weights['avg_route_length'] * pct_changes[3] -  # avg_route_length change
                metric_weights['avg_utilization'] * pct_changes[4]     # avg_utilization change (negative is better)
            )
            
            # Add penalty for extreme changes (to avoid unrealistic solutions)
            if abs(pct_changes[0]) > 50 or abs(pct_changes[2]) > 50 or abs(pct_changes[3]) > 50 or abs(pct_changes[4]) > 50:
                score += 100  # Add penalty for extreme changes
            
            return score, pct_changes
            
        except TimeoutException:
            print(f"  Evaluation timed out after {timeout} seconds")
            return float('inf'), [0, 0, 0, 0, 0]
        except Exception as e:
            print(f"  Error during evaluation: {str(e)}")
            return float('inf'), [0, 0, 0, 0, 0]
        finally:
            # Cancel the alarm
            signal.alarm(0)
    
    def _generate_random_weights(self, num_samples=10):
        """
        Generate random weight combinations.
        
        Args:
            num_samples (int): Number of random weight combinations to generate.
            
        Returns:
            list: List of weight dictionaries.
        """
        weights_list = []
        for _ in range(num_samples):
            weights = {}
            for key, (min_val, max_val) in self.weight_params.items():
                weights[key] = random.uniform(min_val, max_val)
            weights_list.append(weights)
        
        return weights_list
    
    def _train_model(self, weights_data, scores):
        """
        Train the neural network on weights and their scores.
        
        Args:
            weights_data (list): List of weight dictionaries.
            scores (list): List of scores for each weight combination.
        """
        # Convert weights to numpy array
        X = np.array([[w[key] for key in self.weight_params.keys()] for w in weights_data])
        
        # Normalize X to [0,1] range for each feature
        X_norm = np.zeros_like(X)
        for i, key in enumerate(self.weight_params.keys()):
            min_val, max_val = self.weight_params[key]
            X_norm[:, i] = (X[:, i] - min_val) / (max_val - min_val)
        
        # Convert scores to normalized target values (higher score = worse)
        # We invert and normalize scores so that better weights have higher values
        max_score = max(scores)
        min_score = min(scores)
        score_range = max_score - min_score if max_score > min_score else 1.0
        
        # For each weight, set its target value based on its score
        # Better weights (lower scores) get higher target values
        y = np.zeros_like(X_norm)
        for i, score in enumerate(scores):
            # Normalize score to [0,1] range and invert (1 = best, 0 = worst)
            normalized_score = 1.0 - ((score - min_score) / score_range)
            # Set all weights for this sample to the normalized score
            y[i] = X_norm[i] * normalized_score
        
        # Train the model
        self.model.fit(X_norm, y, epochs=50, batch_size=4, verbose=0)
    
    def _predict_better_weights(self, num_predictions=5):
        """
        Use the trained model to predict better weight combinations.
        
        Args:
            num_predictions (int): Number of weight combinations to predict.
            
        Returns:
            list: List of predicted weight dictionaries.
        """
        # Generate random starting points
        random_weights = np.random.random((num_predictions, len(self.weight_params)))
        
        # Predict better weights
        predicted_weights = self.model.predict(random_weights, verbose=0)
        
        # Convert to weight dictionaries
        weights_list = []
        for weights in predicted_weights:
            weights_dict = {}
            for i, key in enumerate(self.weight_params.keys()):
                min_val, max_val = self.weight_params[key]
                weights_dict[key] = weights[i] * (max_val - min_val) + min_val
            weights_list.append(weights_dict)
        
        return weights_list
    
    def optimize(self, num_iterations=10, population_size=5, timeout=60):
        """
        Run the optimization process to find the best weights.
        
        Args:
            num_iterations (int): Number of iterations to run.
            population_size (int): Number of weight combinations to evaluate in each iteration.
            timeout (int): Maximum time in seconds to allow for each evaluation.
            
        Returns:
            dict: Best weights found.
            float: Best score achieved.
            list: History of results.
        """
        print(f"Starting weight optimization with {num_iterations} iterations...")
        start_time = time.time()
        
        # Set default weights in case we don't find better ones
        default_weights = {
            'distance': 10.0,
            'volunteer_count': 10.0,
            'capacity_util': 0.0,
            'history': 0.0,
            'compact_routes': 0.0,
            'clusters': 0.0
        }
        
        # Evaluate default weights first
        print("\nEvaluating default weights first:")
        print(f"  Weights: {default_weights}")
        score, pct_changes = self._evaluate_weights(default_weights, timeout)
        
        # Initialize best weights with default weights
        if score < float('inf'):
            self.best_weights = default_weights.copy()
            self.best_score = score
            print(f"  Default weights score: {score:.2f}")
            print(f"  Percentage changes: {pct_changes}")
            
            # Store results
            self.results_history.append({
                'weights': default_weights.copy(),
                'score': score,
                'pct_changes': pct_changes.copy(),
                'iteration': 0
            })
        else:
            print("  Default weights evaluation failed or timed out.")
            # Still initialize with default weights but don't claim they're good
            self.best_weights = default_weights.copy()
            self.best_score = float('inf')
        
        # Initial population with random weights (more conservative values)
        weights_population = []
        for _ in range(population_size):
            weights = {}
            for key, (min_val, max_val) in self.weight_params.items():
                # Use more conservative ranges to avoid extreme values
                conservative_max = min(15.0, max_val)
                weights[key] = random.uniform(0.0, conservative_max)
            weights_population.append(weights)
        
        for iteration in range(num_iterations):
            print(f"\nIteration {iteration+1}/{num_iterations}")
            
            # Evaluate all weights in the population
            scores = []
            pct_changes_list = []
            
            for i, weights in enumerate(weights_population):
                print(f"  Evaluating weight combination {i+1}/{len(weights_population)}")
                print(f"  Weights: {weights}")
                
                # Skip evaluation if too similar to previously evaluated weights
                skip = False
                for prev_result in self.results_history:
                    similarity = sum(abs(weights.get(k, 0) - prev_result['weights'].get(k, 0)) 
                                  for k in self.weight_params.keys())
                    if similarity < 1.0:  # Very similar weights
                        print(f"  Skipping evaluation (too similar to previous weights)")
                        score = prev_result['score']
                        pct_changes = prev_result['pct_changes']
                        skip = True
                        break
                
                if not skip:
                    score, pct_changes = self._evaluate_weights(weights, timeout)
                
                scores.append(score)
                pct_changes_list.append(pct_changes)
                
                print(f"  Score: {score:.2f}")
                print(f"  Percentage changes: {pct_changes}")
                
                # Update best weights if better
                if score < self.best_score:
                    self.best_score = score
                    self.best_weights = weights.copy()
                    print(f"  New best weights found! Score: {score:.2f}")
            
            # Store results
            for weights, score, pct_changes in zip(weights_population, scores, pct_changes_list):
                self.results_history.append({
                    'weights': weights.copy(),
                    'score': score,
                    'pct_changes': pct_changes.copy(),
                    'iteration': iteration + 1
                })
            
            # Train the model on all data collected so far
            all_weights = [result['weights'] for result in self.results_history]
            all_scores = [result['score'] for result in self.results_history]
            self._train_model(all_weights, all_scores)
            
            # Generate new population with the model
            weights_population = self._predict_better_weights(population_size)
            
            # Add some random weights to maintain diversity
            if iteration < num_iterations - 1:  # Skip in the last iteration
                num_random = max(1, population_size // 5)
                random_weights = []
                for _ in range(num_random):
                    weights = {}
                    for key, (min_val, max_val) in self.weight_params.items():
                        # Use more conservative ranges to avoid extreme values
                        conservative_max = min(15.0, max_val)
                        weights[key] = random.uniform(0.0, conservative_max)
                    random_weights.append(weights)
                weights_population = weights_population[:-num_random] + random_weights
            
            # Print current best
            print(f"Current best weights: {self.best_weights}")
            print(f"Current best score: {self.best_score:.2f}")
            
            # Early stopping if we found a very good solution
            if self.best_score < -50:  # Very good score threshold
                print("Found an excellent solution, stopping early.")
                break
        
        # Final evaluation with the best weights
        print("\nFinal evaluation with best weights:")
        print(f"Best weights: {self.best_weights}")
        
        # Run optimization with best weights
        result = run_optimized_assignments(
            self.admin_data, 
            show_maps=True,  # Show maps for the final result
            output_dir=self.output_dir,
            save_report=True,
            custom_weights=self.best_weights
        )
        
        if result:
            admin_stats, opt_stats, admin_map_path, opt_map_path = result
            pct_changes = compare_assignments(admin_stats, opt_stats)
            
            print(f"Final percentage changes: {pct_changes}")
            print(f"Admin map: {admin_map_path}")
            print(f"Optimized map: {opt_map_path}")
        
        end_time = time.time()
        print(f"\nOptimization completed in {end_time - start_time:.2f} seconds")
        
        # Save results to file
        self._save_results()
        
        return self.best_weights, self.best_score, self.results_history
    
    def _save_results(self):
        """Save optimization results to file."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        results_file = os.path.join(self.output_dir, f"weight_optimization_results_{timestamp}.txt")
        
        # Use default weights if no good weights were found
        if self.best_weights is None:
            self.best_weights = {
                'distance': 10.0,
                'volunteer_count': 10.0,
                'capacity_util': 0.0,
                'history': 0.0,
                'compact_routes': 0.0,
                'clusters': 0.0
            }
            self.best_score = float('inf')
            print("No good weights found, using default weights.")
        
        with open(results_file, 'w') as f:
            f.write("Weight Optimization Results\n")
            f.write("=========================\n\n")
            
            f.write("Best Weights:\n")
            for key, value in self.best_weights.items():
                f.write(f"  {key}: {value:.4f}\n")
            
            f.write(f"\nBest Score: {self.best_score:.4f}\n\n")
            
            f.write("Optimization History:\n")
            f.write("--------------------\n")
            for i, result in enumerate(self.results_history):
                f.write(f"\nTrial {i+1} (Iteration {result['iteration']}):\n")
                f.write("  Weights:\n")
                for key, value in result['weights'].items():
                    f.write(f"    {key}: {value:.4f}\n")
                f.write(f"  Score: {result['score']:.4f}\n")
                f.write(f"  Percentage Changes: {result['pct_changes']}\n")
        
        print(f"Results saved to {results_file}")
        
        # Also create a plot of the optimization progress
        self._plot_optimization_progress()
    
    def _plot_optimization_progress(self):
        """Create a plot showing the optimization progress over iterations."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        plot_file = os.path.join(self.output_dir, f"weight_optimization_plot_{timestamp}.png")
        
        # Extract data for plotting
        iterations = [result['iteration'] for result in self.results_history]
        scores = [result['score'] for result in self.results_history]
        
        # Find best score at each iteration
        best_scores = []
        current_best = float('inf')
        
        for i in range(1, max(iterations) + 1):
            iteration_scores = [score for score, iter_num in zip(scores, iterations) if iter_num <= i]
            if iteration_scores:
                current_best = min(min(iteration_scores), current_best)
            best_scores.append((i, current_best))
        
        # Create plot
        plt.figure(figsize=(12, 8))
        
        # Plot all scores
        plt.scatter(iterations, scores, alpha=0.5, label='All trials')
        
        # Plot best score progression
        best_x, best_y = zip(*best_scores)
        plt.plot(best_x, best_y, 'r-', linewidth=2, label='Best score')
        
        plt.title('Weight Optimization Progress')
        plt.xlabel('Iteration')
        plt.ylabel('Score (lower is better)')
        plt.legend()
        plt.grid(True)
        
        # Save plot
        plt.savefig(plot_file)
        print(f"Optimization progress plot saved to {plot_file}")


def main(num_iterations=10, population_size=5, timeout=60):
    """
    Main function to run the weight optimizer.
    
    Args:
        num_iterations (int): Number of iterations to run.
        population_size (int): Number of weight combinations to evaluate in each iteration.
        timeout (int): Maximum time in seconds to allow for each evaluation.
    """
    optimizer = WeightOptimizer()
    best_weights, best_score, _ = optimizer.optimize(
        num_iterations=num_iterations,
        population_size=population_size,
        timeout=timeout
    )
    
    print("\nOptimization completed!")
    print(f"Best weights: {best_weights}")
    print(f"Best score: {best_score:.4f}")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Weight Optimizer for Volunteer Assignment System')
    parser.add_argument('--iterations', type=int, default=10, help='Number of iterations to run')
    parser.add_argument('--population', type=int, default=5, help='Population size for each iteration')
    parser.add_argument('--timeout', type=int, default=60, help='Timeout in seconds for each evaluation')
    
    args = parser.parse_args()
    
    main(
        num_iterations=args.iterations,
        population_size=args.population,
        timeout=args.timeout
    )
