import streamlit as st
import pandas as pd
import numpy as np
import datetime
import joblib
from sklearn.metrics.pairwise import cosine_similarity
from annoy import AnnoyIndex
from sentence_transformers import SentenceTransformer
import networkx as nx
import torch
from transformers import pipeline

# Load Models
text_model = SentenceTransformer('all-MiniLM-L6-v2')  # For text embeddings
sentiment_analyzer = pipeline('sentiment-analysis')  # For mood detection

# File paths
USER_DATA_PATH = r"D:\genrated_data_recomadation\genrated_data_recomandation\user_data_main.csv"
VIDEO_DATA_PATH = r"D:\genrated_data_recomadation\genrated_data_recomandation\video_data_main.csv"
FOLLOWER_DATA_PATH = r"D:\genrated_data_recomadation\genrated_data_recomandation\follower_data.csv"

@st.cache_data
def load_data():
    return pd.read_csv(USER_DATA_PATH), pd.read_csv(VIDEO_DATA_PATH), pd.read_csv(FOLLOWER_DATA_PATH)

user_data, video_data, follower_data = load_data()

# Identify category columns
category_columns = list(set(user_data.columns) & set(video_data.columns))

# Normalize data
@st.cache_data
def normalize_data(df, columns):
    for col in columns:
        min_val, max_val = df[col].min(), df[col].max()
        if max_val > min_val:
            df[col] = (df[col] - min_val) / (max_val - min_val)
    return df

user_data = normalize_data(user_data, category_columns)
video_data = normalize_data(video_data, category_columns)

# Mood-based recommendations
def get_user_mood(user_id):
    user_history = video_data.sample(5)['title'].tolist()
    mood_scores = sentiment_analyzer(user_history)
    mood = max(mood_scores, key=lambda x: x['score'])['label']
    return mood

# Build Graph-Based Recommendations
def build_graph():
    G = nx.Graph()
    for _, row in follower_data.iterrows():
        user = row['user_id']
        following = eval(row['following'])
        for f in following:
            G.add_edge(user, f)
    return G

graph = build_graph()

# AI-Based Chatbot for Recommendations
def chatbot_recommend(query):
    embedding = text_model.encode(query).reshape(1, -1)
    video_embeddings = np.array([text_model.encode(title) for title in video_data['title']])
    scores = cosine_similarity(embedding, video_embeddings).flatten()
    video_data['score'] = scores
    return video_data.nlargest(5, 'score')[['id', 'title', 'category']]

# AI-Generated Video Summaries
def summarize_video(video_title):
    summary_pipeline = pipeline('summarization')
    return summary_pipeline(video_title, max_length=20, min_length=5, do_sample=False)[0]['summary_text']

# Reinforcement Learning (Placeholder for Dynamic Updates)
def reinforcement_learning(user_id, watched_videos):
    pass  # Implement RL model for future updates
# Get trending videos for new users across all languages and categories
def get_trending_videos():
    global video_data  # Ensure we are working on the correct dataset

    #  Ensure 'recommendation_score' exists
    if 'recommendation_score' not in video_data.columns:
        video_data['recommendation_score'] = np.random.rand(len(video_data))  # Assign random scores to avoid errors

    #  Get a diverse selection of trending videos
    unique_languages = video_data['language'].dropna().unique()
    unique_categories = video_data['category'].dropna().unique()

    trending_videos = []
    for lang in unique_languages:
        for cat in unique_categories:
            lang_cat_videos = video_data[
                (video_data['language'] == lang) & (video_data['category'] == cat)
            ].nlargest(3, 'recommendation_score')
            
            if not lang_cat_videos.empty:
                trending_videos.append(lang_cat_videos)

    # Fill missing slots with top trending ones
    additional_videos_needed = 20 - sum(len(df) for df in trending_videos)
    if additional_videos_needed > 0:
        extra_videos = video_data.nlargest(additional_videos_needed, 'recommendation_score')
        extra_videos['reason'] = 'Additional Global Trending'
        trending_videos.append(extra_videos)

    #  Final dataframe cleanup
    final_trending_videos = pd.concat(trending_videos).drop_duplicates().nlargest(20, 'recommendation_score')
    final_trending_videos['reason'] = 'Trending'

    return final_trending_videos


# Hybrid Recommendation System
def get_recommendations(user_id):
    total_recommendations = 25
    
    user_row = user_data[user_data['user_id'] == user_id]
    if user_row.empty:
        return get_trending_videos()

    mood = get_user_mood(user_id)
    category_scores = user_row[category_columns].sum(axis=0)
    user_preferences = user_row[category_columns].values.flatten()
    video_vectors = video_data[category_columns].values
    scores = np.dot(video_vectors, user_preferences)
    video_data['recommendation_score'] = scores

    recommended_videos = video_data.nlargest(20, 'recommendation_score')
    recommended_videos['reason'] = 'Category Match'

    similar_users = list(graph.neighbors(user_id))[:3]
    similar_users_videos = video_data[video_data['id'].isin(similar_users)].nlargest(5, 'recommend_score')
    similar_users_videos['reason'] = 'Similar Users'

    exploration_videos = video_data.sample(5)
    exploration_videos['reason'] = 'Exploration'
    
    final_recommendations = pd.concat([recommended_videos, similar_users_videos, exploration_videos])
    return final_recommendations.nlargest(total_recommendations, 'recommendation_score')[['id', 'title', 'category', 'reason']]

# Streamlit UI
st.title("Enhanced AI Video Recommendation System")
user_id = st.number_input("Enter User ID:", min_value=1, step=1)
query = st.text_input("Search for videos (AI Chatbot):")

if query:
    chatbot_results = chatbot_recommend(query)
    st.write("### AI Chatbot Recommendations")
    st.dataframe(chatbot_results)

if st.button("Get Recommendations"):
    recommendations = get_recommendations(user_id)
    st.write("### Recommended Videos")
    st.dataframe(recommendations)

if st.button("Get Trending Videos"):
    trending_videos = video_data.nlargest(20, 'recommend_score')
    trending_videos['reason'] = 'Trending'
    st.write("### Trending Videos")
    st.dataframe(trending_videos)
