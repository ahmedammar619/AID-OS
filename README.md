# AID-RL
Assigning volunteers to recipients for delivery automatically using Reinforcement Learning algorithms.

# Volunteer Assignment Optimization with Reinforcement Learning

## 📦 Project Description

This project aims to automate and optimize the assignment of **volunteers** to **recipients** for monthly box deliveries. The main goal is to build an intelligent agent that learns how to assign the right volunteers to the right recipients while considering a variety of practical constraints such as location, vehicle capacity, and historical preferences.

The AI system should:
- Learn from historical assignment data
- Minimize travel distance
- Maximize car capacity usage
- Prioritize known volunteer-recipient pairings
- Adapt to changing volunteer availability and recipient demands
- Allow admin review and correction (used as feedback to learn better)

---

## 🧠 Problem Type

This is a **Reinforcement Learning (RL)** problem, modeled as a **Markov Decision Process (MDP)**.

- **Type:** Episodic (each monthly assignment cycle is an episode)
- **Goal:** Maximize efficiency and quality of volunteer-recipient assignments over each episode

---

## 🧩 MDP Components

### ✅ States

A state encodes the current decision context, including:

- Volunteer info:
  - Volunteer ID
  - Zip code (convertible to coordinates)
  - Car capacity (in box count)
  - History of previous recipient assignments
- Recipient info:
  - Recipient ID
  - Coordinates (lat/lon)
  - Number of boxes required
- Clustering info:
  - Cluster/group ID (if recipients are geographically clustered)
  - Number of unassigned recipients in this cluster
- Dynamic episode context:
  - Number of remaining recipients
  - Remaining volunteers

### 🎯 Actions

An action represents assigning a **volunteer to a recipient or group of recipients** (depending on the clustering approach).

- Option 1: Assign Volunteer X to Recipient Y
- Option 2: Assign Volunteer X to Recipient Group (cluster)

### 💰 Rewards

The reward function should guide the AI towards efficient, realistic assignments. Suggested reward components:

| Condition | Reward |
|----------|--------|
| Volunteer previously delivered to recipient | +3 |
| Volunteer and recipient are close (short travel distance) | +2 |
| Car capacity closely matches number of boxes | +2 |
| Assignment wastes capacity (e.g., car size 16 for 5 boxes) | -1 |
| Assignment exceeds car capacity | -3 |
| Recipients in same area not grouped well | -2 |
| Admin rejects assignment | -5 (simulated feedback) |

Rewards can be scaled and tuned.

### 🔁 Episodes

Each **monthly assignment cycle** is one episode. The agent should assign all recipients to available volunteers by the end of the episode.

---

## 📚 Available Data

The following data is available for training and evaluation:

- `volunteers.csv`:
  - `volunteer_id`
  - `zip_code`
  - `car_capacity`
- `recipients.csv`:
  - `recipient_id`
  - `latitude`
  - `longitude`
  - `box_count`
- `assignment_history.csv`:
  - `volunteer_id`
  - `recipient_id`
  - `timestamp` (past 2 months)
- Additional data can be generated using clustering algorithms (e.g., KMeans, DBSCAN) for recipient grouping

---

## 🧪 Implementation Plan

1. **Feature Engineering**
   - Convert zip codes to coordinates
   - Compute distances between volunteers and recipients
   - Calculate cluster membership
   - Create match history features

2. **Modeling as RL**
   - Define states, actions, rewards, episodes
   - Optionally simulate the environment to train the agent

3. **Algorithm**
   - Start with **Actor-Critic** or **DQN** (Deep Q-Network)
   - Use function approximation (neural networks) for state-action value or policy
   - Optionally add a **model-based component** (Dyna) for faster learning

4. **Training & Evaluation**
   - Use historical data to train (offline)
   - Run simulated episodes for testing
   - Evaluate reward scores, efficiency of assignments
   - Incorporate admin feedback for fine-tuning

---

## 🛠️ Technologies

- Python
- PyTorch or TensorFlow
- Pandas, NumPy
- Scikit-learn (for clustering)
- OpenAI Gym (for environment simulation)
- Matplotlib / Plotly (for visualizing assignments and rewards)

---

## ✅ Goals

- Build a system that performs or assists with volunteer-recipient assignment
- Minimize admin manual work
- Handle scale, changes in volunteer pool, and real-world constraints
- Learn from both data and feedback

---

## 🤝 Admin Feedback

Admin review is crucial. The system should:
- Allow manual override of assignments
- Use rejected assignments as **negative examples**
- Learn from corrections (e.g., through reward shaping or imitation learning)

---

## 📂 File Structure (suggested)

project/ │ ├── data/ │ ├── volunteers.csv │ ├── recipients.csv │ └── assignment_history.csv │ ├── env/ │ └── volunteer_env.py # Custom Gym environment │ ├── models/ │ └── actor_critic.py # RL model │ ├── utils/ │ └── feature_engineering.py # Zip code to coords, clustering, etc. │ ├── train.py # Training loop ├── evaluate.py # Evaluation scripts └── README.md

---

## ✨ Future Improvements

- Add preference learning (if volunteers prefer certain recipients or neighborhoods)
- Add traffic/time constraints for delivery windows
- Add transfer learning if the problem expands to other cities or months
- Explore federated learning if privacy is a concern

---

## 📬 Contact

For questions, improvements, or collaboration, please reach out!



