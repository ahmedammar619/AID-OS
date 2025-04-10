# Volunteer Assignment Optimization with Reinforcement Learning

## Project Description

This project automates and optimizes the monthly assignment of volunteers to recipients for box deliveries. Each recipient may require multiple boxes, and each volunteer has a limited car capacity. The goal is to build an AI system that uses reinforcement learning to learn from past assignments and intelligently handle future ones based on:

- Location proximity
- Vehicle capacity matching
- Historical volunteer–recipient pair preferences
- Efficient grouping of nearby recipients using clustering (DBSCAN)
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
  - Historical match score (past assignments)

- **Recipient Information:**  
  - Recipient ID  
  - Recipient coordinates (latitude, longitude)  
  - Number of boxes required

- **Contextual Information:**  
  - Current pool of unassigned recipients  
  - Cluster information for recipients (derived from DBSCAN clustering)  
  - Counts of recipients within each cluster  
  - Any additional admin or preference signals

### Action (A)
An action is a decision to assign a volunteer to a recipient or to a recipient cluster:
- **Direct Pairing:** `assign(volunteer_id, recipient_id)`
- **Cluster-Based Assignment:** `assign(volunteer_id, recipient_cluster)`

*For this project, we adopt the action formulation that best suits the scale and geographic complexity. In our implementation, we use a direct pairing assignment strategy.*

### Reward (R)
The reward function is defined to encourage optimal assignments:
- **Historical Match Reward:** +3 (if the volunteer has successfully served this recipient in the past)
- **Proximity Reward:** +2 (if the volunteer’s location is close to the recipient’s coordinates)
- **Capacity Compatibility Reward:** +2 (if the volunteer’s car capacity closely matches the recipient’s box requirement)
- **Wasted Capacity Penalty:** -1 (if there is significant unused capacity)
- **Overload Penalty:** -3 (if the assignment exceeds the volunteer's capacity)
- **Poor Clustering Penalty:** -2 (if nearby recipients are not efficiently grouped)
- **Admin Override Penalty:** -5 (if the admin rejects the assignment)

### Episode
An episode is defined as one complete monthly assignment cycle. The episode ends when every recipient has been assigned to a volunteer.

---

## Data Source

All data is stored in a **MySQL** database accessed via phpMyAdmin. The key tables are:
- `volunteers`: Contains `volunteer_id`, `zip_code`, `car_capacity`
- `recipients`: Contains `recipient_id`, `latitude`, `longitude`, `box_count`
- `assignment_history`: Contains `volunteer_id`, `recipient_id`, `timestamp` (historical data for the past two months)

Data is extracted via SQL queries using a suitable Python connector, SQLAlchemy.

---

## Implementation Plan

1. **Data Ingestion and Feature Engineering**  
   - Connect to MySQL to fetch data
   - Convert volunteer zip codes to coordinates
   - Compute travel distances between volunteers and recipients (e.g., using the Haversine formula)
   - Cluster recipients using **DBSCAN** (to capture natural geographic groupings)
   - Generate additional features from historical assignment data

2. **MDP and Environment Design**  
   - Define the state as a composite feature vector
   - Formulate the action space as discrete pairing decisions
   - Implement a custom Gym environment representing a monthly assignment cycle

3. **RL Algorithm: Actor-Critic**  
   - **Actor:** Parameterize the policy π(a|s, θ) using a neural network
   - **Critic:** Estimate the state-value function V(s, w) with another neural network or as part of a shared network  
   - Use the policy gradient theorem to update the actor and a TD error signal to update the critic
   - Optimize the average reward objective through stochastic gradient descent

4. **Training and Evaluation**  
   - Train the RL agent offline using historical data and simulated monthly episodes
   - Incorporate admin feedback as additional reward signals to adjust learning
   - Evaluate performance by comparing automated assignments with known effective assignments

5. **Deployment**  
   - Integrate the trained model into an assignment system
   - Provide an admin interface for review and override of assignments
   - Allow re-running the system after admin review to further optimize assignments

---

## Technology Stack

- **Programming Language:** Python
- **Database:** MySQL (accessed via phpMyAdmin)
- **RL Framework:** Custom implementation using PyTorch
- **Environment Simulation:** Custom Gym environment
- **Data Processing:** Pandas, NumPy
- **Clustering:** Scikit-learn (using DBSCAN)
- **Visualization:** Matplotlib, Plotly

---

## Project Structure



AID-RL/
├── data/
│   └── db_connection.py - Module for connecting to MySQL and fetching data
├── env/
│   └── volunteer_env.py - Custom Gym environment for monthly assignment cycles
├── models/
│   └── actor_critic.py - Actor-Critic model architecture
├── trainers/
│   └── train_agent.py - RL training loop for the Actor-Critic algorithm
├── evaluators/
│   └── evaluate_agent.py - Evaluation and performance tracking scripts
├── utils/
│   ├── feature_engineering.py - Clustering (DBSCAN), distance computation, data preprocessing
│   └── reward_utils.py - Implementation of the reward function
├── interface/
│   └── admin_review.py - Admin interface module for feedback and corrections
├── main.py - Entry-point to run the complete system
├── requirements.txt - List of Python dependencies
└── README.md - This file

---

## ✅ Goals

- Automate volunteer-to-recipient assignments
- Learn from historical data and admin feedback
- Minimize inefficiency in distance and car usage
- Handle changing volunteer and recipient pools
- Provide admin oversight and control

---

## ✨ Future Features

- Admin interface with real-time assignment override
- Traffic-based route optimization
- Recipient delivery window preferences
- Fairness tracking (how often each volunteer is used)
- Multi-city scaling

---

## 📬 Contact

For any questions or collaborations, reach out via this repo or project lead.

