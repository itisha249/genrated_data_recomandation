import numpy as np
import pandas as pd
from collections import defaultdict
from database import get_db_connection

# Function to retrieve interaction data
def get_interaction_data():
    """
    Retrieves and processes interaction data from SQLite.
    """
    conn = get_db_connection()
    df = pd.read_sql_query("SELECT * FROM interactions", conn)
    conn.close()
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
        
        if action in ["click", "watch"]:
            video_rewards[video_id][0] += 1  # Increase success count
        elif action == "skip":
            video_rewards[video_id][1] += 1  # Increase failure count
    
    # Compute Thompson Sampling scores
    video_scores = {vid: np.random.beta(success, failure) for vid, (success, failure) in video_rewards.items()}
    
    # Rank videos based on scores
    ranked_videos = sorted(video_scores.items(), key=lambda x: x[1], reverse=True)
    return ranked_videos
