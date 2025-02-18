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

# Create interactions table with engagement tracking
cursor.execute('''
    CREATE TABLE IF NOT EXISTS interactions (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        user_id INTEGER,
        video_id INTEGER,
        action TEXT,
        watch_time REAL DEFAULT 0,
        liked INTEGER DEFAULT 0,
        shared INTEGER DEFAULT 0,
        timestamp REAL
    )
''')
conn.commit()

# Function to log user interactions with engagement metrics
def log_user_interaction(user_id, video_id, action, watch_time=0, liked=0, shared=0):
    """
    Logs user interactions (click, watch, skip, like, share) in SQLite.
    """
    timestamp = time.time()
    cursor.execute("""
        INSERT INTO interactions (user_id, video_id, action, watch_time, liked, shared, timestamp)
        VALUES (?, ?, ?, ?, ?, ?, ?)
    """, (user_id, video_id, action, watch_time, liked, shared, timestamp))
    conn.commit()
    print(f"✅ Interaction logged: User {user_id}, Video {video_id}, Action {action}, Watch Time {watch_time}s, Liked {liked}, Shared {shared}")

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
        watch_time = row["watch_time"]
        liked = row["liked"]
        shared = row["shared"]
        
        if action in ["click", "watch"] or watch_time > 30 or liked or shared:
            video_rewards[video_id][0] += 1  # Increase success count
        elif action == "skip" or watch_time < 10:
            video_rewards[video_id][1] += 1  # Increase failure count
    
    # Compute Thompson Sampling scores
    video_scores = {vid: np.random.beta(success, failure) for vid, (success, failure) in video_rewards.items()}
    
    # Rank videos based on scores
    ranked_videos = sorted(video_scores.items(), key=lambda x: x[1], reverse=True)
    return ranked_videos

# Function to get ranked recommendations based on real-time data
def get_adaptive_recommendations():
    """
    Fetch and rank recommended videos dynamically using Thompson Sampling.
    """
    df = get_interaction_data()
    ranked_videos = thompson_sampling_ranking(df)
    return [video[0] for video in ranked_videos]

# Example Usage
log_user_interaction(user_id=101, video_id=915, action="click", watch_time=45, liked=1, shared=0)
log_user_interaction(user_id=101, video_id=831, action="watch", watch_time=120, liked=0, shared=1)
log_user_interaction(user_id=102, video_id=947, action="skip", watch_time=5, liked=0, shared=0)

# Retrieve logged interactions
df = get_interaction_data()
print("🔹 User Interaction Data:")
print(df)

# Compute and display ranked recommendations
adaptive_recommendations = get_adaptive_recommendations()
print("🔹 Adaptive Ranked Recommendations:")
print(adaptive_recommendations)
