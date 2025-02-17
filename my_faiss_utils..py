import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA
from scipy.sparse import csr_matrix
from sklearn.neighbors import NearestNeighbors
from sklearn.linear_model import LogisticRegression
import faiss

# File paths
USER_DATA_PATH = "D:/genrated_data_recomadation/genrated_data_recomandation/user_data_main_updated.csv"
VIDEO_DATA_PATH = "D:/genrated_data_recomadation/genrated_data_recomandation/video_data_main.csv"

# Load user and video data with explicit dtype handling
user_data = pd.read_csv(USER_DATA_PATH, dtype={"user_id": int}, low_memory=False)
video_data = pd.read_csv(VIDEO_DATA_PATH, dtype={"id": int}, low_memory=False)

# Extract category columns
category_columns = list(set(user_data.columns) & set(video_data.columns))
required_user_columns = ["user_id"] + category_columns
required_video_columns = ["id", "title", "category"] + category_columns

# Normalize category interactions
def normalize_data(df, columns):
    scaler = MinMaxScaler()
    df[columns] = scaler.fit_transform(df[columns])
    return df

user_data = normalize_data(user_data, category_columns)
video_data = normalize_data(video_data, category_columns)

# Reduce dimensionality using PCA
pca = PCA(n_components=50)
user_reduced_matrix = pca.fit_transform(user_data[category_columns].values)

# Convert to sparse matrix
user_sparse_matrix = csr_matrix(user_reduced_matrix)

# FAISS for large-scale similarity search
dimension = user_reduced_matrix.shape[1]
#faiss_index = faiss_helper.IndexFlatL2(dimension)
faiss_index = faiss.IndexFlatL2(dimension)

faiss_index.add(user_reduced_matrix)

# Logistic Regression Model for Engagement Scoring
engagement_model = LogisticRegression()
X_train = np.random.rand(100, len(category_columns))
y_train = np.random.randint(0, 2, 100)
engagement_model.fit(X_train, y_train)

def predict_engagement(video_id, user_id):
    user_vector = user_data[user_data.user_id == user_id][category_columns].values
    video_vector = video_data[video_data.id == video_id][category_columns].values
    if len(user_vector) == 0 or len(video_vector) == 0:
        return 0  # No match found
    feature_vector = np.multiply(user_vector, video_vector)
    return engagement_model.predict_proba(feature_vector)[0][1]

def get_similar_users(user_id):
    if user_id not in user_data.user_id.values:
        return []  # Return empty for unknown users
    user_idx = user_data[user_data.user_id == user_id].index[0]
    _, indices = faiss_index.search(user_reduced_matrix[user_idx].reshape(1, -1), 10)
    return user_data.iloc[indices[0]]['user_id'].tolist()

# Recommendation function
def get_recommendations(user_id):
    if user_id not in user_data.user_id.values:
        st.warning("New user detected! Showing trending videos.")
        return video_data.nlargest(10, "recommend_score")[["id", "title", "category"]]
    
    similar_users = get_similar_users(user_id)
    recommended_videos = video_data[video_data[category_columns].sum(axis=1) > 0]
    recommended_videos["engagement_score"] = recommended_videos.apply(lambda row: predict_engagement(row["id"], user_id), axis=1)
    return recommended_videos.nlargest(10, "engagement_score")[["id", "title", "category", "engagement_score"]]

# Streamlit UI
st.title("Optimized Personalized Video Recommendation System")
user_id = st.number_input("Enter User ID:", min_value=1, step=1)

if st.button("Get Recommendations"):
    recommendations = get_recommendations(user_id)
    st.dataframe(recommendations)

st.write("### How to Run This App:")
st.code("streamlit run cold_strategy.py")
