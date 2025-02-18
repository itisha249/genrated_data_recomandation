import sqlite3
import json
import time
import pandas as pd
import numpy as np
import networkx as nx
import torch
from transformers import AutoTokenizer, AutoModel
from collections import defaultdict

# Connect to SQLite database
db_path = "user_interactions.db"
conn = sqlite3.connect(db_path)
cursor = conn.cursor()

# Create interactions table with session tracking
cursor.execute('''
    CREATE TABLE IF NOT EXISTS interactions (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        user_id INTEGER,
        session_id TEXT,
        video_id INTEGER,
        action TEXT,
        watch_time REAL DEFAULT 0,
        liked INTEGER DEFAULT 0,
        shared INTEGER DEFAULT 0,
        timestamp REAL
    )
''')
conn.commit()

# Load BERT model & tokenizer
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModel.from_pretrained(MODEL_NAME)

# Function to generate BERT embeddings
def get_embedding(text, model, tokenizer):
    """Generate sentence embeddings using BERT."""
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=128)
    with torch.no_grad():
        outputs = model(**inputs)
    return outputs.last_hidden_state[:, 0, :].squeeze().numpy()

# Function to log user interactions with session tracking
def log_user_interaction(user_id, video_id, action, watch_time=0, liked=0, shared=0, session_id=None):
    """
    Logs user interactions (click, watch, skip, like, share) in SQLite with session tracking.
    """
    if session_id is None:
        session_id = generate_session_id(user_id)
    timestamp = time.time()
    cursor.execute("""
        INSERT INTO interactions (user_id, session_id, video_id, action, watch_time, liked, shared, timestamp)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
    """, (user_id, session_id, video_id, action, watch_time, liked, shared, timestamp))
    conn.commit()
    print(f"✅ Interaction logged: User {user_id}, Session {session_id}, Video {video_id}, Action {action}, Watch Time {watch_time}s, Liked {liked}, Shared {shared}")

# Function to analyze session-based recommendations
def get_session_based_recommendations(user_id):
    """
    Retrieve the most interacted categories/videos within the user's session for short-term recommendations.
    """
    df = get_interaction_data()
    user_sessions = df[df["user_id"] == user_id]
    if user_sessions.empty:
        return []
    
    top_videos = user_sessions.groupby("video_id").agg({"watch_time": "sum", "liked": "sum", "shared": "sum"})
    top_videos["score"] = top_videos["watch_time"] + (top_videos["liked"] * 10) + (top_videos["shared"] * 5)
    ranked_videos = top_videos.sort_values(by="score", ascending=False).index.tolist()
    
    return ranked_videos

# Function to build a user-video graph for recommendations
def build_user_video_graph():
    """
    Creates a graph where users and videos are nodes, and interactions define edges.
    """
    df = get_interaction_data()
    G = nx.Graph()
    
    for _, row in df.iterrows():
        user = f"user_{row['user_id']}"
        video = f"video_{row['video_id']}"
        G.add_edge(user, video, weight=row["watch_time"] + row["liked"] * 10 + row["shared"] * 5)
    
    return G

# Function to recommend videos based on graph structure
def get_graph_based_recommendations(user_id):
    """
    Uses NetworkX to recommend videos based on graph relationships.
    """
    G = build_user_video_graph()
    user_node = f"user_{user_id}"
    
    if user_node not in G:
        return []
    
    recommendations = sorted(G[user_node], key=lambda x: G[user_node][x]['weight'], reverse=True)
    return [int(video.replace("video_", "")) for video in recommendations if video.startswith("video_")]

# Example Usage
session_id = generate_session_id(user_id=101)
log_user_interaction(user_id=101, video_id=915, action="click", watch_time=45, liked=1, shared=0, session_id=session_id)
log_user_interaction(user_id=101, video_id=831, action="watch", watch_time=120, liked=0, shared=1, session_id=session_id)
log_user_interaction(user_id=102, video_id=947, action="skip", watch_time=5, liked=0, shared=0)

# Retrieve logged interactions
df = get_interaction_data()
print("🔹 User Interaction Data:")
print(df)

# Compute and display session-based recommendations
session_recommendations = get_session_based_recommendations(user_id=101)
print("🔹 Session-Based Recommendations:")
print(session_recommendations)

# Compute and display graph-based recommendations
graph_recommendations = get_graph_based_recommendations(user_id=101)
print("🔹 Graph-Based Recommendations:")
print(graph_recommendations)
