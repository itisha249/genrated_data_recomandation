import streamlit as st
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from annoy import AnnoyIndex
import datetime

# File paths
USER_DATA_PATH = r"D:\genrated_data_recomadation\genrated_data_recomandation\user_data_main.csv"
VIDEO_DATA_PATH = r"D:\genrated_data_recomadation\genrated_data_recomandation\video_data_main.csv"
FOLLOWER_DATA_PATH = r"D:\genrated_data_recomadation\genrated_data_recomandation\follower_data.csv"
CHUNK_SIZE = 5000


# Load user, video, and follower data
user_data = pd.read_csv(USER_DATA_PATH)
video_data = pd.read_csv(VIDEO_DATA_PATH)
follower_data = pd.read_csv(FOLLOWER_DATA_PATH)

# Identify category columns
category_columns = list(set(user_data.columns) & set(video_data.columns))

# Normalize data
def normalize_data(df, columns):
    for col in columns:
        min_val = df[col].min()
        max_val = df[col].max()
        if max_val > min_val:
            df[col] = (df[col] - min_val) / (max_val - min_val)
    return df

user_data = normalize_data(user_data, category_columns)
video_data = normalize_data(video_data, category_columns)

# Assign context-aware category preference based on time of day
def get_time_based_category():
    current_hour = datetime.datetime.now().hour
    if current_hour in range(6, 12):
        return ["news", "learning"]
    elif current_hour in range(18, 23):
        return ["entertainment", "movies"]
    else:
        return ["general", "random"]

# Build Annoy Index for fast user similarity lookup
feature_length = len(category_columns)
annoy_index = AnnoyIndex(feature_length, metric='angular')
for i, row in user_data.iterrows():
    annoy_index.add_item(i, row[category_columns].values)
annoy_index.build(10)

# Collaborative Filtering (Optimized using Annoy)
def get_similar_users(user_id, top_n=3):
    if user_id not in user_data['user_id'].values:
        return []
    user_idx = user_data.index[user_data['user_id'] == user_id].tolist()[0]
    similar_users_idx = annoy_index.get_nns_by_item(user_idx, top_n+1)[1:]
    return user_data.iloc[similar_users_idx]['user_id'].tolist()

# Social-Based Recommendations (Follower Influence)
def get_follower_recommendations(user_id):
    user_followers = follower_data[follower_data['user_id'] == user_id]
    if user_followers.empty:
        return pd.DataFrame(columns=['id', 'title', 'category'])
    followed_users = eval(user_followers.iloc[0]['following'])
    followed_videos = video_data[video_data['id'].isin(followed_users)].nlargest(5, 'recommend_score')
    return followed_videos[['id', 'title', 'category']]

# Assign time-based weight to recent uploads if 'upload_time' exists
if 'upload_time' in video_data.columns:
    video_data['upload_time'] = pd.to_datetime(video_data['upload_time'], errors='coerce')
    video_data['time_boost'] = np.exp(-0.01 * (datetime.datetime.now() - video_data['upload_time']).dt.days.fillna(0))
    video_data['recommend_score'] *= video_data['time_boost']

# Get user interest categories with scores
def get_user_interest(user_id):
    user_row = user_data[user_data['user_id'] == user_id]
    if user_row.empty:
        return "New user detected."
    user_interests = user_row[category_columns].iloc[0]
    interested_categories = user_interests[user_interests > 0]
    if interested_categories.empty:
        return "New user detected."
    return interested_categories.to_dict()

# Recommendation function
def get_recommendations(user_id):
    user_row = user_data[user_data['user_id'] == user_id]
    if user_row.empty:
        return video_data.nlargest(20, 'recommend_score')[['language', 'id', 'title', 'category', 'reason']]
    
    user_preferences = user_row[category_columns].values.flatten()
    video_vectors = video_data[category_columns].values
    scores = np.dot(video_vectors, user_preferences)
    video_data['recommendation_score'] = scores
    
    recommended_videos = video_data.nlargest(10, 'recommendation_score')
    recommended_videos['reason'] = 'Category Match'
    
    follower_recommendations = get_follower_recommendations(user_id)
    follower_recommendations['reason'] = 'Follower Watched'
    
    similar_users = get_similar_users(user_id)
    similar_users_videos = video_data[video_data['id'].isin(similar_users)].nlargest(5, 'recommend_score')
    similar_users_videos['reason'] = 'Similar Users'
    
    final_recommendations = pd.concat([recommended_videos, follower_recommendations, similar_users_videos])
    final_recommendations = final_recommendations.drop_duplicates().nlargest(20, 'recommendation_score')
    
    # If there are fewer than 20 recommendations, backfill with trending videos
    if len(final_recommendations) < 20:
        additional_videos = video_data[~video_data['id'].isin(final_recommendations['id'])].nlargest(20 - len(final_recommendations), 'recommend_score')
        additional_videos['reason'] = 'Trending'
        final_recommendations = pd.concat([final_recommendations, additional_videos])
    
    return final_recommendations[['language', 'id', 'title', 'category', 'reason']]

# Streamlit UI
st.title("Enhanced Video Recommendation System")
user_id = st.number_input("Enter User ID:", min_value=1, step=1)
if st.button("Get Recommendations"):
    user_interest = get_user_interest(user_id)
    st.write("### User Interest Areas:")
    if isinstance(user_interest, dict):
        st.dataframe(pd.DataFrame(user_interest.items(), columns=['Category', 'Score']))
    else:
        st.write(user_interest)
    
    recommendations = get_recommendations(user_id)
    if recommendations.empty:
        st.warning("No recommendations available.")
    else:
        st.dataframe(recommendations)