# Volunteer Assignment Optimization with Reinforcement Learning

## Project Description

This project automates and optimizes the monthly assignment of volunteers to recipients for box deliveries. Each recipient may require multiple boxes, and each volunteer has a limited car capacity. The goal is to build an AI system that uses reinforcement learning to learn from past assignments and intelligently handle future ones based on:

- Location proximity and travel time optimization
- Vehicle capacity utilization
- Directional consistency for efficient routing
- Historical volunteer–recipient pair preferences
- Efficient grouping of nearby recipients using clustering (HDBSCAN)
- Volunteer efficiency evaluation
- Admin feedback on assignment quality

The system produces a final, optimized assignment of all recipients to volunteers once a month, with the option to run a second time after admin review.

---

## Problem Type

- **Reinforcement Learning (RL)** modeled as a **Markov Decision Process (MDP)**
- **Episodic MDP:** One episode represents a single monthly assignment cycle
- **Objective:** Maximize overall assignment efficiency and quality over the episode

---

## MDP Components

### State (S)
Each state is represented by a feature vector that includes:
- **Volunteer Information:**  
  - Volunteer ID 
  - Volunteer location (converted from zip code to coordinates)  
  - Car capacity (number of boxes the volunteer can carry)  
  - Current load (boxes already assigned)
  - Historical match score (past assignments)

- **Recipient Information:**  
  - Recipient ID  
  - Recipient coordinates (latitude, longitude)  
  - Number of boxes required

- **Contextual Information:**  
  - Current pool of unassigned recipients  
  - Cluster information for recipients (derived from HDBSCAN clustering)  
  - Counts of recipients within each cluster  
  - Any additional admin or preference signals

### Action (A)
An action is a decision to assign a volunteer to a recipient or to a recipient cluster:
- **Direct Pairing:** `assign(volunteer_id, recipient_id)`
- **Cluster-Based Assignment:** `assign(volunteer_id, recipient_cluster)`

*For this project, we adopt the action formulation that best suits the scale and geographic complexity. In our implementation, we use a direct pairing assignment strategy.*

### Reward (R)
The reward function is defined to encourage optimal assignments, with configurable weights for each component:

- **Volunteer Efficiency:** 
  - +1.0 for selecting the optimal volunteer for a recipient
  - -1.0 to -4.0 penalties for selecting a suboptimal volunteer when better options exist

- **Proximity Rewards:**
  - Exponential decay based on distance (max +2.0)
  - +5.0 for recipients less than 2km apart
  - +2.0 for moderate distances (2-5km)
  - -3.0 * (distance/5.0) penalty for recipients more than 5km apart

- **Directional Consistency:**
  - +4.0 for recipients in similar directions (<30° difference)
  - +2.0 for recipients in moderately similar directions (30-45°)
  - -2.0 penalty for significantly different directions (90-120°)
  - -5.0 severe penalty for opposite directions (>120°)

- **Capacity Utilization:**
  - +3.0 for optimal capacity usage (90-115%)
  - +1.0 for good utilization (80-90%)
  - +0.5 for moderate utilization (70-80%)
  - -4.0 penalty for overloading (>115%)

- **Cluster Membership:**
  - +1.0 for keeping recipients from the same cluster with the same volunteer
  - -2.0 penalty for splitting clusters across volunteers with low utilization

- **Travel Time Efficiency:**
  - -2.5 points per hour of estimated travel time
  - Includes route optimization and delivery time estimation

### Episode
An episode is defined as one complete monthly assignment cycle. The episode ends when every recipient has been assigned to a volunteer.

---

## Data Source

All data is stored in a **MySQL** database accessed via phpMyAdmin. The key tables are:
- `volunteer`: Contains `volunteer_id`, `zip_code`, `car_size`
- `recipient`: Contains `recipient_id`, `latitude`, `longitude`, `num_items`
- `delivery_archive`: Contains `volunteer_id`, `recipient_id`, `timestamp` (historical data for the past two months)

Data is extracted via SQL queries using a suitable Python connector, SQLAlchemy.

---

## Implementation Details

### 1. Data Ingestion and Feature Engineering
- MySQL database connection using SQLAlchemy
- Haversine distance calculation for accurate distance measurement
- DBSCAN clustering for geographic grouping of recipients
- Historical match score retrieval and decay based on recency

### 2. Environment Design
- Custom Gym-compatible environment (DeliveryEnv)
- State representation with volunteer and recipient features
- Action space as volunteer-recipient pairings
- Sophisticated reward function with weighted components

### 3. RL Algorithm: Proximal Policy Optimization (PPO)
- **Actor:** Policy network that selects the best volunteer-recipient pairing
- **Critic:** Value network that estimates expected returns
- Clipped surrogate objective to prevent policy collapse
- Entropy bonus for exploration
- Generalized Advantage Estimation (GAE) for stable learning
- Configurable hyperparameters:
  - `clip_epsilon=0.2`
  - `value_loss_coef=0.5`
  - `entropy_coef=0.01`

### 4. Route Optimization
- Greedy nearest-neighbor algorithm for route planning
- Travel time estimation based on distance and number of stops
- Directional consistency evaluation to minimize backtracking
- Round-trip calculation (volunteer home → recipients → home)

### 5. Visualization and Reporting
- Interactive map visualization with Folium
- Detailed popup information including:
  - Volunteer capacity and utilization
  - Assigned recipients and box counts
  - Estimated travel time and distance
- Load distribution charts
- Comprehensive assignment reports

---

## Technology Stack

- **Programming Language:** Python 3.8+
- **Database:** MySQL (accessed via SQLAlchemy)
- **RL Framework:** Custom implementation using PyTorch
- **Environment Simulation:** Custom Gym-compatible environment
- **Data Processing:** Pandas, NumPy
- **Clustering:** Scikit-learn (HDBSCAN)
- **Visualization:** Matplotlib, Folium, Seaborn
- **Geospatial Calculations:** Haversine formula for accurate distances

---

## Project Structure


```bash
AID-RL/
├── data/
│   └── db_config.py           # MySQL connection configuration using SQLAlchemy
│
├── models/
│   ├── actor.py               # Defines the policy network (Actor)
│   ├── critic.py              # Defines the value estimator (Critic)
│   └── rl_agent.py            # The Actor-Critic agent class
│
├── clustering/
│   └── dbscan_cluster.py      # DBSCAN-based recipient clustering
│
├── env/
│   └── delivery_env.py        # Custom RL environment for volunteer-recipient assignment
│
├── training/
│   └── train_agent.py         # Training loop for RL agent
│
├── feedback/
│   └── feedback_handler.py    # Collects and processes admin feedback
│
├── assignment/
│   └── assign_volunteers.py   # Outputs optimal volunteer-to-recipient assignments
│
└── README.md                  # Project overview
```

---

## ✅ Achievements

- **Automated Assignment System:** Complete end-to-end pipeline from data to assignments
- **Optimized Routes:** Minimized travel time and distance with directional consistency
- **Efficient Capacity Utilization:** Balanced assignments to maximize vehicle usage
- **Intelligent Clustering:** Grouped nearby recipients for efficient delivery
- **Travel Time Estimation:** Accurate prediction of route completion times
- **Interactive Visualization:** Detailed maps showing assignments and metrics
- **Comprehensive Reporting:** Detailed assignment statistics and performance metrics

---

## ✨ Future Features

- **Pickup Location Integration:** Include food pickup locations in route planning
- **Advanced Route Optimization:** Implement full Traveling Salesman Problem (TSP) solution
- **Recipient Time Windows:** Support for preferred delivery time slots
- **Fairness Metrics:** Balance volunteer usage over time
- **Multi-City Scaling:** Support for multiple geographic regions
- **Hyperparameter Optimization:** Automated tuning of reward weights

---

## 📬 Contact

For any questions or collaborations, reach out via this repo or project lead.

