#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Weight Optimizer for Volunteer Assignment System

This module optimizes weights for the volunteer assignment system using a neural network.
It evaluates weight combinations against admin assignments and learns to predict better weights.
Supports disabling specific weights to focus optimization on selected objectives.
"""

import os
import sys
import time
import random
import numpy as np
import signal
from datetime import datetime
import tensorflow as tf
from tensorflow import keras
from keras import layers
import matplotlib.pyplot as plt

# Add parent directory to path for project imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from assignment.compare_with_admin import get_admin_assignments, run_optimized_assignments, compare_assignments

class TimeoutException(Exception):
    """Exception raised when a function execution times out."""
    pass

def timeout_handler(signum, frame):
    """Handle SIGALRM signal for timeouts."""
    raise TimeoutException("Function execution timed out")

class WeightOptimizer:
    """
    Optimizes weights for volunteer assignment using a neural network and iterative evaluation.
    
    Features:
    - Evaluates weights by comparing OR-Tools assignments to admin assignments.
    - Trains a neural network to predict better weights.
    - Supports disabling specific weights (e.g., set 'history' to 0).
    - Saves results and plots optimization progress.
    """
    
    def __init__(self, admin_data, output_dir="./hist/output"):
        """
        Initialize the optimizer.
        
        Args:
            admin_data (dict): Admin assignments data from database.
            output_dir (str): Directory to save results and plots.
        """
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        
        # Validate and store admin data
        self.admin_data = admin_data
        if not admin_data:
            raise ValueError("No admin assignments provided.")
        
        # Compute admin stats if not present
        self.admin_stats = admin_data.get('stats')
        if not self.admin_stats:
            from assignment.compare_with_admin import calculate_assignment_stats
            self.admin_stats = calculate_assignment_stats(admin_data)
            self.admin_data['stats'] = self.admin_stats
        
        # Weight ranges (allow negative weights for flexibility)
        self.weight_params = {
            'distance': (-20.0, 20.0),
            'volunteer_count': (-20.0, 20.0),
            'capacity_util': (-20.0, 20.0),
            'history': (-20.0, 20.0),
            'compact_routes': (-20.0, 20.0),
            'clusters': (-20.0, 20.0)
        }
        
        # Track best weights and results
        self.best_weights = None
        self.best_score = float('inf')
        self.results_history = []
        
        # Build neural network
        self.model = self._build_model()
    
    def _build_model(self):
        """
        Create a neural network to predict optimal weights.
        
        Returns:
            keras.Model: Compiled neural network.
        """
        model = keras.Sequential([
            layers.Dense(32, activation='relu', input_shape=(len(self.weight_params),)),
            layers.Dense(64, activation='relu'),
            layers.Dense(32, activation='relu'),
            layers.Dense(len(self.weight_params), activation='tanh')  # [-1,1] for weights
        ])
        model.compile(optimizer='adam', loss='mse')
        return model
    
    def _normalize_weights(self, weights_dict):
        """
        Normalize weights to [0,1] for neural network training.
        
        Args:
            weights_dict (dict): Weight values.
        
        Returns:
            dict: Normalized weights.
        """
        normalized = {}
        for key, value in weights_dict.items():
            min_val, max_val = self.weight_params[key]
            normalized[key] = (value - min_val) / (max_val - min_val)
        return normalized
    
    def _denormalize_weights(self, normalized_weights):
        """
        Convert normalized weights [0,1] back to original ranges.
        
        Args:
            normalized_weights (np.array): Normalized weight values.
        
        Returns:
            dict: Denormalized weights.
        """
        weights_dict = {}
        for i, key in enumerate(self.weight_params.keys()):
            min_val, max_val = self.weight_params[key]
            # Scale [-1,1] to original range
            scaled = (normalized_weights[i] + 1) / 2  # Convert [-1,1] to [0,1]
            weights_dict[key] = min_val + scaled * (max_val - min_val)
        return weights_dict
    
    def _evaluate_weights(self, weights, timeout=60):
        """
        Evaluate weights by running OR-Tools and comparing to admin assignments.
        
        Args:
            weights (dict): Weight values.
            timeout (int): Max seconds for evaluation.
        
        Returns:
            float: Score (lower is better).
            list: Percentage changes in metrics.
        """
        # Skip extreme weights
        for key, value in weights.items():
            if abs(value) > 20.0:
                print(f"Skipping evaluation: {key}={value} too extreme")
                return float('inf'), [0, 0, 0, 0, 0]
        
        # Set timeout
        signal.signal(signal.SIGALRM, timeout_handler)
        signal.alarm(timeout)
        
        try:
            # Run OR-Tools with weights
            result = run_optimized_assignments(
                self.admin_data,
                show_maps=False,
                output_dir=self.output_dir,
                save_report=False,
                custom_weights=weights
            )
            if not result:
                return float('inf'), [0, 0, 0, 0, 0]
            
            # Compare metrics
            admin_stats, opt_stats, _, _ = result
            pct_changes = compare_assignments(admin_stats, opt_stats)
            
            # Calculate score: prioritize distance and utilization
            metric_weights = {
                'total_volunteers': 0.3,  # Less focus on volunteer count
                'total_distance': 2.0,    # Minimize distance
                'avg_route_length': 5.0,  # Minimize route length
                'avg_utilization': 1.5    # Maximize utilization
            }
            score = (
                metric_weights['total_volunteers'] * max(pct_changes[0], 0) +  # Penalize positive, ignore negative
                metric_weights['total_distance'] * max(pct_changes[2], 0) +    # Penalize positive, ignore negative
                metric_weights['avg_route_length'] * max(pct_changes[3], 0) +  # Penalize positive, ignore negative
                metric_weights['avg_utilization'] * max(-pct_changes[4], 0)    # Penalize negative, reward positive
            )
            
            # Penalize extreme changes
            if any(abs(change) > 50 for change in pct_changes):
                score += 100
            return score, pct_changes
        
        except TimeoutException:
            print(f"Evaluation timed out after {timeout}s")
            return float('inf'), [0, 0, 0, 0, 0]
        except Exception as e:
            print(f"Evaluation error: {e}")
            return float('inf'), [0, 0, 0, 0, 0]
        finally:
            signal.alarm(0)
    
    def _generate_random_weights(self, num_samples, disabled_weights=None):
        """
        Generate random weight combinations, respecting disabled weights.
        
        Args:
            num_samples (int): Number of weight sets to generate.
            disabled_weights (set): Weights to set to 0 (e.g., {'history'}).
        
        Returns:
            list: List of weight dictionaries.
        """
        disabled = disabled_weights or set()
        weights_list = []
        for _ in range(num_samples):
            weights = {}
            for key, (min_val, max_val) in self.weight_params.items():
                weights[key] = 0.0 if key in disabled else random.uniform(min_val, max_val)
            weights_list.append(weights)
        return weights_list
    
    def _train_model(self, weights_data, scores):
        """
        Train the neural network on evaluated weights and scores.
        
        Args:
            weights_data (list): List of weight dictionaries.
            scores (list): Corresponding scores.
        """
        # Prepare training data
        X = np.array([[w[key] for key in self.weight_params.keys()] for w in weights_data])
        X_norm = np.zeros_like(X)
        for i, key in enumerate(self.weight_params.keys()):
            min_val, max_val = self.weight_params[key]
            X_norm[:, i] = (X[:, i] - min_val) / (max_val - min_val)
        
        # Normalize scores to [0,1], invert so lower scores are better
        min_score, max_score = min(scores), max(scores)
        score_range = max_score - min_score if max_score > min_score else 1.0
        y = np.array([1.0 - (score - min_score) / score_range for score in scores])
        y = np.clip(y, 0.0, 1.0)[:, None] * X_norm  # Scale normalized weights
        
        # Train
        self.model.fit(X_norm, y, epochs=30, batch_size=8, verbose=0)
    
    def _predict_better_weights(self, num_predictions, disabled_weights=None):
        """
        Predict new weight combinations using the neural network.
        
        Args:
            num_predictions (int): Number of predictions.
            disabled_weights (set): Weights to set to 0.
        
        Returns:
            list: Predicted weight dictionaries.
        """
        disabled = disabled_weights or set()
        random_inputs = np.random.random((num_predictions, len(self.weight_params)))
        predicted = self.model.predict(random_inputs, verbose=0)
        
        weights_list = []
        for weights in predicted:
            weights_dict = self._denormalize_weights(weights)
            for key in disabled:
                weights_dict[key] = 0.0
            weights_list.append(weights_dict)
        return weights_list
    
    def optimize(self, num_iterations=10, population_size=5, timeout=60, disabled_weights=None):
        """
        Run optimization to find the best weights.
        
        Args:
            num_iterations (int): Number of iterations.
            population_size (int): Weight sets per iteration.
            timeout (int): Max seconds per evaluation.
            disabled_weights (set): Weights to disable (e.g., {'history', 'clusters'}).
        
        Returns:
            dict: Best weights.
            float: Best score.
            list: History of results.
        """
        print(f"Starting optimization ({num_iterations} iterations, {population_size} sets)")
        disabled = disabled_weights or set()
        start_time = time.time()
        
        # Default weights
        default_weights = {
            'distance': 10.0,
            'volunteer_count': 5.0,
            'capacity_util': 5.0,
            'history': 0.0,
            'compact_routes': 0.0,
            'clusters': 0.0
        }
        for key in disabled:
            default_weights[key] = 0.0
        
        # Evaluate default weights
        print("\nTesting default weights:")
        print(f"Weights: {default_weights}")
        score, pct_changes = self._evaluate_weights(default_weights, timeout)
        if score < float('inf'):
            self.best_weights = default_weights.copy()
            self.best_score = score
            self.results_history.append({
                'weights': default_weights.copy(),
                'score': score,
                'pct_changes': pct_changes,
                'iteration': 0
            })
            print(f"Score: {score:.2f}, Changes: {pct_changes}")
        else:
            print("Default weights failed")
            self.best_weights = default_weights.copy()
        
        # Main loop
        weights_data = []
        scores = []
        for iteration in range(1, num_iterations + 1):
            print(f"\nIteration {iteration}/{num_iterations}")
            
            # Generate population
            population = self._generate_random_weights(population_size, disabled)
            
            # Evaluate population
            for i, weights in enumerate(population, 1):
                print(f"Evaluating set {i}/{population_size}: {weights}")
                score, pct_changes = self._evaluate_weights(weights, timeout)
                
                if score < float('inf'):
                    print(f"Score: {score:.2f}, Changes: {pct_changes}")
                    weights_data.append(weights)
                    scores.append(score)
                    self.results_history.append({
                        'weights': weights.copy(),
                        'score': score,
                        'pct_changes': pct_changes,
                        'iteration': iteration
                    })
                    if score < self.best_score:
                        self.best_score = score
                        self.best_weights = weights.copy()
                        print("New best weights found!")
                else:
                    print("Evaluation failed")
            
            # Train model if enough data
            if len(weights_data) >= 4:
                print("Training neural network...")
                self._train_model(weights_data, scores)
                # Generate new population: mix best, random, and predicted
                population = [self.best_weights.copy()]
                population += self._generate_random_weights(population_size // 2, disabled)
                population += self._predict_better_weights(population_size - len(population), disabled)
            else:
                population = self._generate_random_weights(population_size, disabled)
        
        # Final evaluation
        print("\nFinal evaluation with best weights:")
        print(f"Weights: {self.best_weights}")
        result = run_optimized_assignments(
            self.admin_data,
            show_maps=True,
            output_dir=self.output_dir,
            save_report=True,
            custom_weights=self.best_weights
        )
        if result:
            _, _, admin_map, opt_map = result
            print(f"Admin map: {admin_map}\nOptimized map: {opt_map}")
        
        # Save results
        self._save_results()
        print(f"\nCompleted in {time.time() - start_time:.1f}s")
        return self.best_weights, self.best_score, self.results_history
    
    def _save_results(self):
        """Save results to a text file and plot progress."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        results_file = os.path.join(self.output_dir, f"weight_optimization_results_{timestamp}.txt")
        
        with open(results_file, 'w') as f:
            f.write("Weight Optimization Results\n========================\n")
            f.write("\nBest Weights:\n")
            for key, value in self.best_weights.items():
                f.write(f"  {key}: {value:.2f}\n")
            f.write(f"\nBest Score: {self.best_score:.2f}\n")
            f.write("\nHistory:\n--------\n")
            for i, result in enumerate(self.results_history, 1):
                f.write(f"Trial {i} (Iteration {result['iteration']}):\n")
                f.write("  Weights: " + ", ".join(f"{k}: {v:.2f}" for k, v in result['weights'].items()) + "\n")
                f.write(f"  Score: {result['score']:.2f}\n")
                f.write(f"  Changes: {result['pct_changes']}\n\n")
        print(f"Results saved to {results_file}")
        
        # Plot progress
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        plot_file = os.path.join(self.output_dir, f"weight_optimization_plot_{timestamp}.png")
        iterations = [r['iteration'] for r in self.results_history]
        scores = [r['score'] for r in self.results_history if r['score'] < float('inf')]
        if not scores:
            print("No valid scores to plot")
            return
        
        plt.figure(figsize=(10, 6))
        plt.scatter(iterations, scores, alpha=0.5, label='Trials')
        best_scores = []
        current_best = float('inf')
        for i in range(max(iterations) + 1):
            valid_scores = [s for s, it in zip(scores, iterations) if it <= i]
            if valid_scores:
                current_best = min(min(valid_scores), current_best)
            best_scores.append(current_best)
        plt.plot(range(max(iterations) + 1), best_scores, 'r-', label='Best Score')
        plt.title('Weight Optimization Progress')
        plt.xlabel('Iteration')
        plt.ylabel('Score (Lower is Better)')
        plt.legend()
        plt.grid(True)
        plt.savefig(plot_file)
        print(f"Plot saved to {plot_file}")

def main(num_iterations=10, population_size=5, timeout=60, disabled_weights=None):
    """
    Run the weight optimizer with command-line arguments.
    
    Args:
        num_iterations (int): Number of iterations.
        population_size (int): Weight sets per iteration.
        timeout (int): Max seconds per evaluation.
        disabled_weights (set): Weights to disable (e.g., {'history', 'clusters'}).
    """
    admin_data = get_admin_assignments()
    if not admin_data:
        print("No admin assignments found")
        return
    
    optimizer = WeightOptimizer(admin_data)
    best_weights, best_score, _ = optimizer.optimize(
        num_iterations=num_iterations,
        population_size=population_size,
        timeout=timeout,
        disabled_weights=disabled_weights
    )
    print(f"\nBest Weights: {best_weights}")
    print(f"Best Score: {best_score:.2f}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Optimize weights for volunteer assignment")
    parser.add_argument('--iterations', type=int, default=10, help='Number of iterations')
    parser.add_argument('--population', type=int, default=5, help='Population size')
    parser.add_argument('--timeout', type=int, default=60, help='Timeout per evaluation (seconds)')
    parser.add_argument('--disable', nargs='*', default=[], help='Weights to disable (e.g., history clusters)')
    args = parser.parse_args()
    
    main(
        num_iterations=args.iterations,
        population_size=args.population,
        timeout=args.timeout,
        disabled_weights=set(args.disable)
    )