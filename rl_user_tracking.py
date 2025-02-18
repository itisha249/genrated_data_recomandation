import sqlite3
import json
import time
import pandas as pd
import numpy as np
from collections import defaultdict

# Connect to SQLite database
db_path = "user_interactions.db"
conn = sqlite3.connect(db_path)
cursor = conn.cursor()

# Create interactions table
cursor.execute('''
    CREATE TABLE IF NOT EXISTS interactions (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        user_id INTEGER,
        video_id INTEGER,
        action TEXT,
        timestamp REAL
    )
''')
conn.commit()

# Function to log user interactions
def log_user_interaction(user_id, video_id, action):
    """
    Logs user interactions (click, watch, skip) in SQLite.
    """
    timestamp = time.time()
    cursor.execute("INSERT INTO interactions (user_id, video_id, action, timestamp) VALUES (?, ?, ?, ?)",
                   (user_id, video_id, action, timestamp))
    conn.commit()
    print(f"✅ Interaction logged: User {user_id}, Video {video_id}, Action {action}")

# Function to retrieve interaction data
def get_interaction_data():
    """
    Retrieves and processes interaction data from SQLite.
    """
    df = pd.read_sql_query("SELECT * FROM interactions", conn)
    return df

# Multi-Armed Bandit (Thompson Sampling) for Adaptive Recommendations
def thompson_sampling_ranking(df):
    """
    Implements Thompson Sampling to rank videos dynamically based on user interactions.
    """
    video_rewards = defaultdict(lambda: [1, 1])  # Beta distribution parameters (success, failure)
    
    # Process user feedback
    for _, row in df.iterrows():
        video_id = row["video_id"]
        action = row["action"]
        
        if action == "click" or action == "watch":
            video_rewards[video_id][0] += 1  # Increase success count
        elif action == "skip":
            video_rewards[video_id][1] += 1  # Increase failure count
    
    # Compute Thompson Sampling scores
    video_scores = {vid: np.random.beta(success, failure) for vid, (success, failure) in video_rewards.items()}
    
    # Rank videos based on scores
    ranked_videos = sorted(video_scores.items(), key=lambda x: x[1], reverse=True)
    return ranked_videos

# Example Usage
log_user_interaction(user_id=101, video_id=915, action="click")
log_user_interaction(user_id=101, video_id=831, action="watch")
log_user_interaction(user_id=102, video_id=947, action="skip")

# Retrieve logged interactions
df = get_interaction_data()
print("🔹 User Interaction Data:")
print(df)

# Compute Thompson Sampling-Based Ranking
ranked_videos = thompson_sampling_ranking(df)
print("🔹 Ranked Videos:")
print(ranked_videos)
