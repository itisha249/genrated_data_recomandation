import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler

# Step 1: Load necessary data
user_data_path = r"D:\genrated_data_recomadation\genrated_data_recomandation\user_data_main_updated.csv"  # Path to user data
video_data_path = r"D:\genrated_data_recomadation\genrated_data_recomandation\video_data_main.csv"  # Path to video data



# Load data
user_data = pd.read_csv(user_data_path, low_memory=False)  # To handle mixed data types warning
video_data = pd.read_csv(video_data_path)

# Step 2: Normalize user data for category preferences
category_columns = user_data.columns[3:-2]  # Adjust to match actual category columns in your dataset
scaler = MinMaxScaler()
user_data[category_columns] = scaler.fit_transform(user_data[category_columns])
user_data_normalized = user_data.copy()

# Step 3: Add 'age_group' column if it's missing
def categorize_age(age):
    if 15 <= age <= 24:
        return "Teens & Young Adults"
    elif 25 <= age <= 39:
        return "Middle-Aged Adults"
    else:
        return "Older Adults"

if 'age_group' not in user_data_normalized.columns:
    user_data_normalized['age_group'] = user_data_normalized['age'].apply(categorize_age)

# Step 4: Prepare video data (if necessary, simulate freshness and popularity)
category_columns_video = video_data.columns[6:]  # Adjust based on video data

# Simulate 'upload_date' and 'popularity' for demonstration purposes
if 'upload_date' not in video_data.columns:
    video_data['upload_date'] = pd.date_range(start="2023-01-01", periods=len(video_data))

if 'popularity' not in video_data.columns:
    video_data['popularity'] = np.random.randint(1, 1000, size=len(video_data))

# Calculate freshness score
current_date = pd.Timestamp.now()
video_data['days_since_upload'] = (current_date - pd.to_datetime(video_data['upload_date'])).dt.days
max_days = video_data['days_since_upload'].max()
video_data['freshness_score'] = 1 - (video_data['days_since_upload'] / max_days)

# Normalize popularity score
video_data['popularity_score'] = video_data['popularity'] / video_data['popularity'].max()

# Ensure the category columns match between user data and video data
common_columns = list(set(category_columns) & set(category_columns_video))

# Step 5: Define age group preferences inside the function
def get_user_recommendations(user_id_to_test):
    # Define the age group preferences here
    age_group_preferences = {
        "Teens & Young Adults": ["gaming", "music", "funny", "anime-&-cartoons", "diy-&-home", "life-hacks", "travel", "learning", "news", "art"],
        "Middle-Aged Adults": ["diy-&-home", "life-hacks", "travel", "music", "news"],
        "Older Adults": ["learning", "news", "art", "music", "funny", "life-hacks"]
    }

    # Step 5.1: Retrieve user preferences for the test user
    user_row = user_data_normalized[user_data_normalized["user_id"] == user_id_to_test]
    
    if user_row.empty:
        print("User not found!")
        return

    # Normalize user preferences again (if needed)
    user_preferences = user_row[common_columns].values.flatten()

    # Step 5.2: Compute similarity between user preferences and videos
    video_recommendations = []

    for _, video in video_data.iterrows():
        video_id = video["id"]
        video_vector = video[common_columns].values

        # Compute similarity using dot product (cosine similarity)
        similarity_score = np.dot(user_preferences, video_vector)
        final_score = (
            0.6 * similarity_score +  # Weight for similarity
            0.3 * video['freshness_score'] +  # Weight for freshness
            0.1 * video['popularity_score']  # Weight for popularity
        )

        # Get the category for the video
        category = video['category'] if 'category' in video else 'No Category'  # Adjust based on your dataset
        video_recommendations.append((video_id, video["title"], category, final_score))

    # Step 5.3: Sort videos by final score and filter top 10 recommendations
    video_recommendations_sorted = sorted(video_recommendations, key=lambda x: x[3], reverse=True)
    top_10_recommendations = video_recommendations_sorted[:10]

    # Step 5.4: Apply Age-Based Filtering and Diversity
    final_recommendations = []
    age_group = user_row["age_group"].values[0]
    
    # Print top 10 recommendations before filtering
    print(f"\nTop 10 recommendations for User {user_id_to_test} before filtering:")
    for video in top_10_recommendations:
        print(f"Video ID: {video[0]}, Title: {video[1]}, Category: {video[2]}, Score: {video[3]}")

    valid_videos = [video for video in top_10_recommendations if video[2] in age_group_preferences.get(age_group, [])]

    # Print valid videos after age group filtering
    print(f"\nValid videos after applying age group filtering for User {user_id_to_test}:")
    for video in valid_videos:
        print(f"Video ID: {video[0]}, Title: {video[1]}, Category: {video[2]}")

    # Apply diversity filtering (limit 3 per category)
    category_counts = {}
    for video in valid_videos:
        category = video[2]
        if category not in category_counts:
            category_counts[category] = 0

        # Allow up to 3 videos per category, but also ensure we get at least some videos if fewer than 3 per category
        if category_counts[category] < 3:
            final_recommendations.append(video)
            category_counts[category] += 1

    # Print top 10 recommendations after diversity filtering
    print(f"\nTop 10 recommendations for User {user_id_to_test} after age and diversity filtering:")
    if final_recommendations:
        for idx, (video_id, title, category, score) in enumerate(final_recommendations, start=1):
            print(f"{idx}. Video ID: {video_id}, Title: {title}, Category: {category}, Similarity Score: {score:.4f}")
    else:
        print("No valid videos after filtering.")

    # Step 6: Option to save the recommendations as a CSV file
    recommendations_df = pd.DataFrame(final_recommendations, columns=["Video ID", "Title", "Category", "Similarity Score"])
    recommendations_file_path = f"user_{user_id_to_test}_recommendations.csv"
    recommendations_df.to_csv(recommendations_file_path, index=False)
    print(f"Recommendations saved to: {recommendations_file_path}")

# Step 7: Run the recommendation system for a test user ID
test_user_id = 32  # Replace with the user ID you'd like to test
get_user_recommendations(test_user_id)

