import streamlit as st
import pandas as pd
import numpy as np

# Define file paths
USER_DATA_PATH = "D:/genrated_data_recomadation/genrated_data_recomandation/user_data_main_updated.csv"
VIDEO_DATA_PATH = "D:/genrated_data_recomadation/genrated_data_recomandation/video_data_main.csv"
CHUNK_SIZE = 5000

# Load sample data to identify category columns
sample_user_data = pd.read_csv(USER_DATA_PATH, nrows=5)
sample_video_data = pd.read_csv(VIDEO_DATA_PATH, nrows=5)
category_columns = list(set(sample_user_data.columns) & set(sample_video_data.columns))

# Required columns
required_user_columns = ["user_id"] + category_columns
required_video_columns = ["id", "title", "category"] + category_columns

# Normalization function
def normalize_data(df, columns):
    for col in columns:
        min_val = df[col].min()
        max_val = df[col].max()
        if max_val > min_val:
            df[col] = (df[col] - min_val) / (max_val - min_val)
    return df

# User preferences extraction
def get_user_preferences(user_row):
    top_categories = user_row[category_columns].T.nlargest(5, columns=user_row.index[0]).index.tolist()
    exploring_categories = user_row[category_columns].T.nlargest(15, columns=user_row.index[0]).index.tolist()
    
    main_preference = {cat: 1.0 for cat in top_categories}
    exploring_preference = {cat: 0.5 for cat in exploring_categories}
    
    return main_preference, exploring_preference

# Recommendation function
def get_user_recommendations(user_id):
    user_data = pd.read_csv(USER_DATA_PATH, usecols=required_user_columns)
    user_data = normalize_data(user_data, category_columns)
    user_row = user_data[user_data["user_id"] == user_id]
    
    if user_row.empty or (user_row[category_columns].sum().sum() == 0):
        st.warning("New user detected! Recommendations are based on top-rated videos in each category.")
        recommendations = []
        category_recommendations = []
        
        for chunk in pd.read_csv(VIDEO_DATA_PATH, usecols=required_video_columns, chunksize=CHUNK_SIZE):
            chunk = normalize_data(chunk, category_columns)
            if not category_columns:
                continue
            chunk["final_score"] = chunk[category_columns].mean(axis=1)
            for category in chunk["category"].unique():
                category_videos = chunk[chunk["category"] == category]
                category_recommendations.append(category_videos.nlargest(5, "final_score"))
        
        if category_recommendations:
            top_recommendations = pd.concat(category_recommendations)
            return top_recommendations[["id", "title", "category"]]
        else:
            return pd.DataFrame(columns=["id", "title", "category"])
    
    user_preferences = user_row[category_columns].values.flatten()
    main_preference, exploring_preference = get_user_preferences(user_row)
    recommendations = []
    
    for chunk in pd.read_csv(VIDEO_DATA_PATH, usecols=required_video_columns, chunksize=CHUNK_SIZE):
        chunk = normalize_data(chunk, category_columns)
        video_vectors = chunk[category_columns].values
        
        interest_scores = np.dot(video_vectors, user_preferences) * 0.4
        main_scores = np.array([
            sum(main_preference.get(cat, 0) * chunk.at[i, cat] for cat in category_columns) * 0.35
            for i in range(len(chunk))
        ])
        exploring_scores = np.array([
            sum(exploring_preference.get(cat, 0) * chunk.at[i, cat] for cat in category_columns) * 0.25
            for i in range(len(chunk))
        ])
        
        chunk["final_score"] = interest_scores + main_scores + exploring_scores
        recommendations.append(chunk.nlargest(15, "final_score")[["id", "title", "category"]])
    
    if recommendations:
        top_recommendations = pd.concat(recommendations)
        if "final_score" in top_recommendations.columns:
            top_recommendations = top_recommendations.nlargest(15, "final_score")
        return top_recommendations[["id", "title", "category"]]
    else:
        return pd.DataFrame(columns=["id", "title", "category"])

# Streamlit UI
st.title("Personalized Video Recommendation System")
user_id = st.number_input("Enter User ID:", min_value=1, step=1)

if st.button("Get Recommendations"):
    recommendations = get_user_recommendations(user_id)
    if recommendations.empty:
        st.warning("No recommendations available.")
    else:
        st.dataframe(recommendations)
