import streamlit as st
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from annoy import AnnoyIndex
import datetime
import joblib

# File paths
USER_DATA_PATH = r"D:\genrated_data_recomadation\genrated_data_recomandation\user_data_main.csv"
VIDEO_DATA_PATH = r"D:\genrated_data_recomadation\genrated_data_recomandation\video_data_main.csv"
FOLLOWER_DATA_PATH = r"D:\genrated_data_recomadation\genrated_data_recomandation\follower_data.csv"


# Load user, video, and follower data
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
        min_val = df[col].min()
        max_val = df[col].max()
        if max_val > min_val:
            df[col] = (df[col] - min_val) / (max_val - min_val)
    return df

user_data = normalize_data(user_data, category_columns)
video_data = normalize_data(video_data, category_columns)

# Assign time-based category preference
def get_time_based_category():
    current_hour = datetime.datetime.now().hour
    if current_hour in range(6, 12):
        return ["news", "learning"]
    elif current_hour in range(18, 23):
        return ["entertainment", "movies"]
    else:
        return ["general", "random"]

# Multi-Language Filtering with Equal Distribution
# def filter_language_recommendations(user_id, recommendations, total_recommendations=20):
#     user_row = user_data[user_data['user_id'] == user_id]
#     if user_row.empty:
#         return recommendations.nlargest(total_recommendations, 'recommend_score')
    
#     user_languages = str(user_row['languages'].values[0]).split(',') if not pd.isna(user_row['languages'].values[0]) else []
#     user_languages = [lang.strip() for lang in user_languages if lang.strip()]
    
#     if len(user_languages) == 0:
#         return recommendations.nlargest(total_recommendations, 'recommend_score')
    
#     num_languages = len(user_languages)
#     videos_per_language = total_recommendations // num_languages
#     extra_videos = total_recommendations % num_languages
    
#     filtered_videos = []
#     for lang in user_languages:
#         lang_videos = recommendations[recommendations['language'].str.strip() == lang].nlargest(videos_per_language, 'recommend_score')
#         filtered_videos.append(lang_videos)
    
#     # Fill extra slots with videos from any preferred language
#     if extra_videos > 0:
#         extra_lang_videos = recommendations[recommendations['language'].isin(user_languages)].nlargest(extra_videos, 'recommend_score')
#         filtered_videos.append(extra_lang_videos)
    
#     final_videos = pd.concat(filtered_videos).drop_duplicates()
    
#     # If we still don't have enough, get exploration videos
#     if len(final_videos) < total_recommendations:
#         exploration_videos = video_data[~video_data['id'].isin(final_videos['id'])].nlargest(total_recommendations - len(final_videos), 'recommend_score')
#         exploration_videos['reason'] = 'Exploration'
#         final_videos = pd.concat([final_videos, exploration_videos])
    
#         return final_videos.nlargest(total_recommendations, 'recommend_score')
#     user_row = user_data[user_data['user_id'] == user_id]
#     if user_row.empty:
#         return recommendations.nlargest(total_recommendations, 'recommend_score')
    
#     user_languages = list(set(str(user_row['languages'].values[0]).split(','))) if not pd.isna(user_row['languages'].values[0]) else []
    
#     if len(user_languages) == 0:
#         return recommendations.nlargest(total_recommendations, 'recommend_score')
    
#     num_languages = len(user_languages)
#     videos_per_language = total_recommendations // num_languages
#     extra_videos = total_recommendations % num_languages
    
#     filtered_videos = []
#     for lang in user_languages:
#         lang_videos = recommendations[recommendations['language'] == lang].nlargest(videos_per_language, 'recommend_score')
#         filtered_videos.append(lang_videos)
    
#     # Fill extra slots with random videos from preferred languages
#     if extra_videos > 0:
#         extra_lang_videos = recommendations[recommendations['language'].isin(user_languages)].sample(n=extra_videos, random_state=42, replace=True)
#         filtered_videos.append(extra_lang_videos)
    
#     final_videos = pd.concat(filtered_videos).drop_duplicates()
    
#     # If we still don't have enough, get exploration videos
#     if len(final_videos) < total_recommendations:
#         exploration_videos = video_data[~video_data['id'].isin(final_videos['id'])].nlargest(total_recommendations - len(final_videos), 'recommendation_score')
#         exploration_videos['reason'] = 'Exploration'
#         final_videos = pd.concat([final_videos, exploration_videos])
#     user_row = user_data[user_data['user_id'] == user_id]
#     if user_row.empty:
#         return recommendations.nlargest(total_recommendations, 'recommend_score')
    
#     user_languages = list(set(str(user_row['languages'].values[0]).split(','))) if not pd.isna(user_row['languages'].values[0]) else []
    
#     if len(user_languages) == 0:
#         return recommendations.nlargest(total_recommendations, 'recommend_score')
    
#     num_languages = len(user_languages)
#     videos_per_language = total_recommendations // num_languages
#     extra_videos = total_recommendations % num_languages
    
#     filtered_videos = []
#     for lang in user_languages:
#         lang_videos = recommendations[recommendations['language'] == lang].nlargest(videos_per_language, 'recommend_score')
#         filtered_videos.append(lang_videos)
    
#     # Fill extra slots with videos from any preferred language
#     if extra_videos > 0:
#         extra_lang_videos = recommendations[recommendations['language'].isin(user_languages)].nlargest(extra_videos, 'recommend_score')
#         filtered_videos.append(extra_lang_videos)
    
#     final_videos = pd.concat(filtered_videos).drop_duplicates()
    
#     # If we still don't have enough, get exploration videos
#     if len(final_videos) < total_recommendations:
#         exploration_videos = video_data[~video_data['id'].isin(final_videos['id'])].nlargest(total_recommendations - len(final_videos), 'recommendation_score')
#         exploration_videos['reason'] = 'Exploration'
#         final_videos = pd.concat([final_videos, exploration_videos])
    
#         return final_videos.nlargest(total_recommendations, 'recommendation_score')
#     user_row = user_data[user_data['user_id'] == user_id]
#     if user_row.empty:
#         return recommendations.nlargest(total_recommendations, 'recommend_score')
    
#     user_languages = list(set(str(user_row['languages'].values[0]).split(','))) if not pd.isna(user_row['languages'].values[0]) else []
    
#     if len(user_languages) == 0:
#         return recommendations.nlargest(total_recommendations, 'recommend_score')
    
#     num_languages = len(user_languages)
#     videos_per_language = total_recommendations // num_languages
#     extra_videos = total_recommendations % num_languages
    
#     filtered_videos = []
#     for lang in user_languages:
#         lang_videos = recommendations[recommendations['language'] == lang].nlargest(videos_per_language, 'recommend_score')
#         filtered_videos.append(lang_videos)
    
#     # Fill extra slots with random videos from preferred languages
#     if extra_videos > 0:
#         extra_lang_videos = recommendations[recommendations['language'].isin(user_languages)].nlargest(extra_videos, 'recommend_score')
#         filtered_videos.append(extra_lang_videos)
    
#         final_videos = pd.concat(filtered_videos).drop_duplicates().nlargest(total_recommendations, 'recommend_score')
#         return final_videos
#     user_row = user_data[user_data['user_id'] == user_id]
#     if user_row.empty:
#         return recommendations.nlargest(total_recommendations, 'recommend_score')
    
#     user_languages = list(set(str(user_row['languages'].values[0]).split(','))) if not pd.isna(user_row['languages'].values[0]) else []
    
#     if len(user_languages) == 0:
#         return recommendations.nlargest(total_recommendations, 'recommend_score')
    
#     num_languages = len(user_languages)
#     videos_per_language = total_recommendations // num_languages
#     extra_videos = total_recommendations % num_languages
    
#     filtered_videos = []
#     for lang in user_languages:
#         lang_videos = recommendations[recommendations['language'] == lang].nlargest(videos_per_language, 'recommend_score')
#         filtered_videos.append(lang_videos)
    
#     # If there are extra slots, fill them randomly from available preferred languages
#     if extra_videos > 0:
#         extra_lang_videos = recommendations[recommendations['language'].isin(user_languages)].sample(n=extra_videos, random_state=42, replace=True)
#         filtered_videos.append(extra_lang_videos)
    
#         final_videos = pd.concat(filtered_videos).drop_duplicates()
#         return final_videos.nlargest(total_recommendations, 'recommend_score')
#     user_row = user_data[user_data['user_id'] == user_id]
#     if user_row.empty:
#         return recommendations.nlargest(total_recommendations, 'recommend_score')
    
#     user_languages = list(set(str(user_row['languages'].values[0]).split(','))) if not pd.isna(user_row['languages'].values[0]) else []
    
#     if len(user_languages) == 0:
#         return recommendations.nlargest(total_recommendations, 'recommend_score')
    
#     num_languages = len(user_languages)
#     videos_per_language = total_recommendations // num_languages
#     extra_videos = total_recommendations % num_languages
    
#     filtered_videos = []
#     for lang in user_languages:
#         lang_videos = recommendations[recommendations['language'] == lang].nlargest(videos_per_language, 'recommend_score')
#         filtered_videos.append(lang_videos)
    
#     if extra_videos > 0:
#         extra_lang_videos = recommendations[recommendations['language'].isin(user_languages)].sample(n=extra_videos, random_state=42)
#         filtered_videos.append(extra_lang_videos)
    
#         return pd.concat(filtered_videos).drop_duplicates().nlargest(total_recommendations, 'recommend_score')
#         user_row = user_data[user_data['user_id'] == user_id]
#     if user_row.empty:
#         return recommendations.nlargest(total_recommendations, 'recommend_score')
    
#     user_languages = list(set(str(user_row['languages'].values[0]).split(','))) if not pd.isna(user_row['languages'].values[0]) else []
    
#     if len(user_languages) == 0:
#         return recommendations.nlargest(total_recommendations, 'recommend_score')
    
#     num_languages = len(user_languages)
#     videos_per_language = total_recommendations // num_languages
#     extra_videos = total_recommendations % num_languages
    
#     filtered_videos = []
#     for lang in user_languages:
#         lang_videos = recommendations[recommendations['language'] == lang].nlargest(videos_per_language, 'recommend_score')
#         filtered_videos.append(lang_videos)
    
#     if extra_videos > 0:
#         extra_lang_videos = recommendations[recommendations['language'].isin(user_languages)].nlargest(extra_videos, 'recommend_score')
#         filtered_videos.append(extra_lang_videos)
    
#     return pd.concat(filtered_videos).drop_duplicates().nlargest(total_recommendations, 'recommend_score')

def filter_language_recommendations(user_id, recommendations, total_recommendations=20):
    # Get user preferences
    user_row = user_data[user_data['user_id'] == user_id]
    if user_row.empty:
        return recommendations.nlargest(total_recommendations, 'recommend_score')
    
    # Extract preferred languages
    user_languages = str(user_row['languages'].values[0]).split(',') if not pd.isna(user_row['languages'].values[0]) else []
    user_languages = [lang.strip() for lang in user_languages if lang.strip()]
    
    # If no languages, return top recommendations
    if not user_languages:
        return recommendations.nlargest(total_recommendations, 'recommend_score')
    
    # Calculate video slots per language
    num_languages = len(user_languages)
    videos_per_language = total_recommendations // num_languages
    extra_videos = total_recommendations % num_languages

    # Collect filtered videos per language
    filtered_videos = []
    for lang in user_languages:
        lang_videos = recommendations[recommendations['language'].str.strip() == lang].nlargest(videos_per_language, 'recommend_score')
        filtered_videos.append(lang_videos)
    
    # Fill extra slots
    if extra_videos > 0:
        extra_lang_videos = recommendations[recommendations['language'].isin(user_languages)].nlargest(extra_videos, 'recommend_score')
        filtered_videos.append(extra_lang_videos)
    
    # Combine and remove duplicates
    final_videos = pd.concat(filtered_videos).drop_duplicates()

    # If not enough videos, add exploration videos
    if len(final_videos) < total_recommendations:
        exploration_videos = video_data[~video_data['id'].isin(final_videos['id'])].nlargest(total_recommendations - len(final_videos), 'recommend_score')
        exploration_videos['reason'] = 'Exploration'
        final_videos = pd.concat([final_videos, exploration_videos])

    # Return final recommendations
    return final_videos.nlargest(total_recommendations, 'recommend_score')


# Build Annoy Index for fast user similarity lookup
if "annoy_index" not in st.session_state:
    feature_length = len(category_columns)
    annoy_index = AnnoyIndex(feature_length, metric='angular')
    for i, row in user_data.iterrows():
        annoy_index.add_item(i, row[category_columns].values)
    annoy_index.build(10)
    st.session_state.annoy_index = annoy_index

annoy_index = st.session_state.annoy_index

# Social-Based Recommendations
def get_follower_recommendations(user_id):
    user_followers = follower_data[follower_data['user_id'] == user_id]
    if user_followers.empty:
        return pd.DataFrame(columns=['id', 'title', 'category'])
    followed_users = eval(user_followers.iloc[0]['following'])
    followed_videos = video_data[video_data['id'].isin(followed_users)].nlargest(5, 'recommend_score')
    return followed_videos[['id', 'title', 'category']]
# Ensure valid category extraction without indexing errors
def get_frequent_categories(user_row):
    if user_row.empty:
        return []
    try:
        frequent_categories = user_row[category_columns].T.loc[:, user_row[category_columns].sum(axis=1) > 0].index.tolist()
    except:
        frequent_categories = []
    return frequent_categories

# Similar Users Recommendation
def get_similar_users(user_id, top_n=3):
    if user_id not in user_data['user_id'].values:
        return []
    user_idx = user_data.index[user_data['user_id'] == user_id].tolist()[0]
    similar_users_idx = annoy_index.get_nns_by_item(user_idx, top_n+1)[1:]
    return user_data.iloc[similar_users_idx]['user_id'].tolist()

# Hybrid Recommendation Function

def get_recommendations(user_id):
    total_recommendations = 25  # Ensure a total of 25 recommendations

    #  Step 1: Check if the user exists
    user_row = user_data[user_data['user_id'] == user_id]
    if user_row.empty:
        print(f"User ID {user_id} not found! Returning trending videos.")
        return get_trending_videos()

    #  Step 2: Extract user preferences from category scores
    category_scores = user_row[category_columns].sum(axis=0)

    #  Step 3: Ensure valid categories for recommendations
    if (category_scores > 0).any():
        minimal_interest_categories = category_scores.nsmallest(2).index.tolist()
    else:
        minimal_interest_categories = []

    #  Step 4: Compute Content-Based Recommendation Scores
    user_preferences = user_row[category_columns].values.flatten()
    video_vectors = video_data[category_columns].values
    scores = np.dot(video_vectors, user_preferences)

    video_data['recommendation_score'] = scores
    recommended_videos = video_data.nlargest(20, 'recommendation_score')
    recommended_videos['reason'] = 'Category Match'

    #  Step 5: Get Social-Based Recommendations (Follower Watching)
    follower_recommendations = get_follower_recommendations(user_id)
    follower_recommendations['reason'] = 'Follower Watched'

    #  Step 6: Get Collaborative Filtering Recommendations (Similar Users)
    similar_users = get_similar_users(user_id)
    similar_users_videos = video_data[video_data['id'].isin(similar_users)].nlargest(5, 'recommend_score')
    similar_users_videos['reason'] = 'Similar Users'

    #  Step 7: Personal Interest-Based Recommendations
    interest_based_videos = video_data[video_data[category_columns].dot(user_preferences) > 0.5].nlargest(10, 'recommend_score')
    interest_based_videos['reason'] = 'Related to Interest'

    #  Step 8: Combine All Recommendations
    final_recommendations = pd.concat([
        recommended_videos, 
        follower_recommendations, 
        similar_users_videos, 
        interest_based_videos
    ])

    #  Step 9: Add Exploration Videos if needed
    if minimal_interest_categories:
        exploration_videos = video_data[video_data['category'].isin(minimal_interest_categories)].nlargest(5, 'recommend_score')
        exploration_videos['reason'] = 'Minimal Interest Exploration'
        final_recommendations = pd.concat([final_recommendations, exploration_videos])

    #  Step 10: Apply Language Filtering to Match User's Preferred Language
    final_recommendations = filter_language_recommendations(user_id, final_recommendations, total_recommendations)

    #  Step 11: Ensure 25 final recommendations and drop duplicates
    final_recommendations = final_recommendations.drop_duplicates().nlargest(total_recommendations, 'recommendation_score')

    return final_recommendations[['language', 'id', 'title', 'category', 'reason']]

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



# Streamlit UI
st.title("Enhanced Video Recommendation System")
user_id = st.number_input("Enter User ID:", min_value=1, step=1)
if st.button("Get Recommendations"):
    recommendations = get_recommendations(user_id)
    
    if recommendations.empty:
        st.write("New user detected.")
        user_row = user_data[user_data['user_id'] == user_id]
        def get_trending_videos():
            if video_data.empty:
                return pd.DataFrame(columns=['id', 'title', 'language', 'category', 'recommendation_score', 'reason'])

            trending_videos = video_data.nlargest(20, 'recommendation_score')
            trending_videos = trending_videos.copy()  # Ensure modifications apply
            trending_videos['reason'] = 'Trending'
            return trending_videos

    # if recommendations.empty:
    #     st.write("New user detected. Showing trending videos.")
        
    #     # Get trending videos across all languages and categories
    #     trending_videos = get_trending_videos()
        
    #     st.write("### Trending Videos for New Users (All Languages & Categories)")
    #     st.dataframe(trending_videos[['id', 'title', 'language', 'category', 'recommendation_score']])
        if not user_row.empty and 'languages' in user_row.columns and not pd.isna(user_row['languages'].values[0]):
            user_languages = str(user_row['languages'].values[0]).split(',')
            user_languages = [lang.strip() for lang in user_languages if lang.strip()]
            trending_videos = video_data[video_data['language'].isin(user_languages)].nlargest(20, 'recommend_score')
            
        else:
            trending_videos = video_data.nlargest(20, 'recommend_score')
            trending_videos['reason'] = 'Global Trending'
        st.write("### Trending Videos for New Users (All Languages)")
        st.dataframe(trending_videos[['id', 'title', 'language', 'category', 'recommend_score']])
        st.write("New user detected.")
        user_row = user_data[user_data['user_id'] == user_id]
        if not user_row.empty and 'languages' in user_row.columns and not pd.isna(user_row['languages'].values[0]):
            user_languages = str(user_row['languages'].values[0]).split(',')
            user_languages = [lang.strip() for lang in user_languages if lang.strip()]
            trending_videos = video_data[video_data['language'].isin(user_languages)].nlargest(20,'recommend_score')
            trending_videos['reason'] = 'Trending in Preferred Language'
        else:
            trending_videos = video_data.nlargest(20, 'recommend_score')
            trending_videos['reason'] = 'Global Trending'
        
        st.write("### Trending Videos for New Users")
        st.dataframe(trending_videos[['id', 'title', 'language', 'category', 'reason']])
        st.write("New user detected.")
        viral_videos = video_data.nlargest(20, 'recommend_score')[['language', 'id', 'title', 'category']]
        st.write("### Trending Viral Videos")
        st.dataframe(viral_videos)
    else:
        st.write("### Recommended Video Languages")
        st.write(recommendations['language'].unique())
        st.write("### Top 20 Recommended Videos")
        st.dataframe(recommendations[['id', 'title', 'language', 'category', 'reason']])
        st.write(recommendations['language'].unique())
        user_row = user_data[user_data['user_id'] == user_id][category_columns]
        if user_row.empty or user_row.sum().sum() == 0:
            st.write("New user detected.")
        else:
            st.write("### User Category Scores")
            st.dataframe(user_row.T[user_row.T.sum(axis=1) > 0])