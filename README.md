# 📦 Volunteer Assignment Optimization with Reinforcement Learning

## Project Description

This project automates and optimizes the monthly assignment of **volunteers** to **recipients** for box deliveries. Each recipient may require multiple boxes, and each volunteer has a limited car capacity. The goal is to build an AI system that learns from past assignments and intelligently handles future ones based on:

- Location proximity
- Vehicle capacity
- Historical volunteer-recipient relationships
- Efficient area grouping
- Admin feedback on assignment quality

The goal is to reduce the time and complexity involved in manual assignments, while maintaining high-quality, human-level decision-making.

---

## 🔍 Problem Type

- **Type:** Reinforcement Learning (RL)
- **Model:** Markov Decision Process (MDP)
- **Episode:** One month of assignments
- **Goal:** Maximize efficiency of assignments across each episode

---

## 🧩 MDP Components

### ✅ State (S)

Each state represents a snapshot of the delivery task:

- Remaining recipients (IDs, coords, box count)
- Available volunteers (ID, zip code, car capacity)
- Past assignment preference (volunteer-recipient history)
- Recipient clusters (grouped geographically)
- Unassigned recipient count per cluster
- Volunteer capacity used so far

### 🎯 Action (A)

Assign a volunteer to one or more recipients from a cluster:

- `assign(volunteer_id, recipient_id or recipient_group)`

### 💰 Reward (R)

| Scenario | Reward |
|---------|--------|
| Volunteer previously served the recipient | +3 |
| Short travel distance (within cluster) | +2 |
| Car capacity matches box count well | +2 |
| Wasted capacity (unused box space) | -1 |
| Assignment exceeds car size | -3 |
| Recipients in same area poorly split | -2 |
| Admin override/rejection | -5 |

### 🔁 Episode

One monthly assignment cycle = one full episode. Agent must complete all assignments.

---

## 📚 Data Source

- All data is stored in a **MySQL database** via **phpMyAdmin**
- Tables include:
  - `volunteers`: volunteer_id, zip_code, car_capacity
  - `recipients`: recipient_id, latitude, longitude, box_count
  - `assignments`: volunteer_id, recipient_id, timestamp
- Data is accessed and preprocessed using SQL queries

---

## 🧪 Implementation Plan

1. **Data Ingestion**
   - Connect to MySQL using SQLAlchemy or PyMySQL
   - Extract, clean, and convert data into appropriate format

2. **Feature Engineering**
   - Convert zip codes to lat/lon coordinates
   - Calculate distance matrix
   - Cluster recipients (e.g., using DBSCAN or KMeans)
   - Extract past volunteer-recipient preferences

3. **Environment Modeling**
   - Implement custom Gym-like environment
   - Define state transitions and reward structure

4. **Model Selection**
   - Use **Actor-Critic** or **DQN** with neural networks
   - Use function approximation to handle large state space

5. **Training**
   - Train using historical assignment data and simulated environment
   - Include admin feedback as part of reward shaping

6. **Deployment**
   - Expose model via API
   - Admin can trigger, inspect, and modify auto-assignments

---

## 🛠️ Technology Stack

- **Language:** Python
- **Database:** MySQL (access via phpMyAdmin)
- **AI Frameworks:** PyTorch or TensorFlow
- **Visualization:** Matplotlib / Plotly
- **Environment Simulation:** OpenAI Gym (custom)
- **Data Science Tools:** Pandas, NumPy, scikit-learn
- **Web/API (optional):** FastAPI or Flask

---

## 📂 Project Structure

AID-RL/
├── data/
│   └── db_connection.py        # Connects and fetches data from MySQL
├── env/
│   └── volunteer_env.py        # Custom Gym environment definition
├── models/
│   └── actor_critic.py         # Actor-Critic model architecture
├── trainers/
│   └── train_agent.py          # RL training loop
├── evaluators/
│   └── evaluate_agent.py       # Evaluation and performance tracking
├── utils/
│   ├── feature_engineering.py  # Clustering, distance, preprocessing
│   └── reward_utils.py         # Reward shaping and scoring logic
├── interface/
│   └── admin_review.py         # Admin feedback loop for corrections
├── main.py                     # Run complete system
├── requirements.txt            # Python dependencies
└── README.md                   # This file

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

