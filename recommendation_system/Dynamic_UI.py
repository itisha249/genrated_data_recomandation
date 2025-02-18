import sqlite3
import json
import time
import pandas as pd
import numpy as np
import networkx as nx
import torch
import streamlit as st
import joblib
import os
from annoy import AnnoyIndex
from transformers import AutoTokenizer, AutoModel
from collections import defaultdict

# Try to import FAISS safely
try:
    import faiss
    FAISS_AVAILABLE = True
except ImportError:
    print("⚠️ FAISS is not installed. Running without FAISS support.")
    FAISS_AVAILABLE = False

# File paths to uploaded datasets
USER_DATA_PATH = r"D:\genrated_data_recomadation\genrated_data_recomandation\user_data_main.csv"
VIDEO_DATA_PATH = r"D:\genrated_data_recomadation\genrated_data_recomandation\video_data_main.csv"
FOLLOWER_DATA_PATH = r"D:\genrated_data_recomadation\genrated_data_recomandation\follower_data.csv"
CATEGORY_DATA_PATH = r"D:\genrated_data_recomadation\genrated_data_recomandation\df_cat.csv"
VIDEO_EMBEDDINGS_PATH = r"D:\genrated_data_recomadation\genrated_data_recomandation\video_embeddings.index"

# Load datasets
user_data = pd.read_csv(USER_DATA_PATH)
video_data = pd.read_csv(VIDEO_DATA_PATH)
follower_data = pd.read_csv(FOLLOWER_DATA_PATH)
df_cat = pd.read_csv(CATEGORY_DATA_PATH)

# Detect and load video embeddings
VECTOR_SIZE = 384  # Adjust based on embedding size
def load_embeddings(file_path):
    if file_path.endswith(".index"):  # Likely an Annoy or FAISS index
        try:
            print("🔍 Trying to load as Annoy index...")
            annoy_index = AnnoyIndex(VECTOR_SIZE, 'angular')
            annoy_index.load(file_path)
            return annoy_index
        except Exception:
            print("❌ Annoy index load failed, trying FAISS...")
        
        if FAISS_AVAILABLE:
            try:
                return faiss.read_index(file_path)
            except Exception as e:
                print("❌ FAISS index load failed!", e)
    elif file_path.endswith(".pkl"):
        try:
            return joblib.load(file_path)
        except Exception as e:
            print("❌ Joblib load failed!", e)
    print("⚠️ Could not load embeddings. Check file format!")
    return None

video_embeddings = load_embeddings(VIDEO_EMBEDDINGS_PATH)

# Connect to SQLite database
db_path = "user_interactions.db"
conn = sqlite3.connect(db_path)
cursor = conn.cursor()

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

# Function to retrieve interaction data
def get_interaction_data():
    """Retrieves and processes interaction data from SQLite."""
    df = pd.read_sql_query("SELECT * FROM interactions", conn)
    return df

# Function to get session-based recommendations
def get_session_based_recommendations(user_id):
    """Retrieve session-based recommendations by analyzing user interactions."""
    df = get_interaction_data()
    user_sessions = df[df["user_id"] == user_id]
    if user_sessions.empty:
        return []
    
    # Rank videos by watch time, likes, and shares
    top_videos = user_sessions.groupby("video_id").agg({"watch_time": "sum", "liked": "sum", "shared": "sum"})
    top_videos["score"] = top_videos["watch_time"] + (top_videos["liked"] * 10) + (top_videos["shared"] * 5)
    ranked_videos = top_videos.sort_values(by="score", ascending=False).index.tolist()
    return ranked_videos

# Function to get graph-based recommendations
def get_graph_based_recommendations(user_id):
    """Generate recommendations using a user-video interaction graph."""
    df = get_interaction_data()
    G = nx.Graph()
    
    for _, row in df.iterrows():
        user = f"user_{row['user_id']}"
        video = f"video_{row['video_id']}"
        G.add_edge(user, video, weight=row["watch_time"] + row["liked"] * 10 + row["shared"] * 5)
    
    user_node = f"user_{user_id}"
    if user_node not in G:
        return []
    
    recommendations = sorted(G[user_node], key=lambda x: G[user_node][x]['weight'], reverse=True)
    return [int(video.replace("video_", "")) for video in recommendations if video.startswith("video_")]

# Function to display recommendations in a dynamic UI
def display_recommendations(user_id):
    """Displays personalized video recommendations in a UI."""
    st.title("🎥 Personalized Video Recommendations")
    
    st.subheader(f"Recommended for User {user_id}")
    session_recommendations = get_session_based_recommendations(user_id)
    graph_recommendations = get_graph_based_recommendations(user_id)
    
    recommended_videos = list(set(session_recommendations + graph_recommendations))
    if not recommended_videos:
        st.write("No recommendations available. Try interacting with more videos!")
    else:
        for video_id in recommended_videos:
            st.write(f"📺 Video ID: {video_id}")
            st.button("Watch", key=f"watch_{video_id}")
            st.button("Like 👍", key=f"like_{video_id}")
            st.button("Share 🔄", key=f"share_{video_id}")
            st.markdown("---")

# Streamlit UI
def main():
    st.sidebar.title("User Selection")
    user_id = st.sidebar.number_input("Enter User ID:", min_value=1, step=1)
    if st.sidebar.button("Get Recommendations"):
        display_recommendations(user_id)

if __name__ == "__main__":
    main()
