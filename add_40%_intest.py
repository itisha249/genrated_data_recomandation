import pandas as pd
import numpy as np

# Paths to data files
user_data_path = r"D:\genrated_data_recomadation\genrated_data_recomandation\user_data_main_updated.csv"
video_data_path = r"D:\genrated_data_recomadation\genrated_data_recomandation\video_data_main.csv"

# Define the chunk size to prevent memory overflow
chunk_size = 5000

# Load only necessary columns for efficient processing
required_user_columns = ["user_id"]  # Start with user_id
required_video_columns = ["id", "title", "category"]

# Load sample data first to determine relevant category columns
sample_user_data = pd.read_csv(user_data_path, nrows=5)
sample_video_data = pd.read_csv(video_data_path, nrows=5)

# Identify common category columns in both datasets
category_columns = list(set(sample_user_data.columns) & set(sample_video_data.columns))
required_user_columns.extend(category_columns)
required_video_columns.extend(category_columns)

# Normalize user preferences function
def normalize_data(df, columns):
    for col in columns:
        min_val = df[col].min()
        max_val = df[col].max()
        if max_val > min_val:  # Avoid division by zero
            df[col] = (df[col] - min_val) / (max_val - min_val)
    return df

# Function to get user preferences
def get_user_preferences(user_row):
    top_categories = user_row[category_columns].T.nlargest(5, columns=user_row.index[0]).index.tolist()
    exploring_categories = user_row[category_columns].T.nlargest(15, columns=user_row.index[0]).index.tolist()
    
    main_preference = {cat: 1.0 for cat in top_categories}
    exploring_preference = {cat: 0.5 for cat in exploring_categories}
    
    return main_preference, exploring_preference

# Function to process video data in chunks and generate recommendations
def get_user_recommendations(user_id_to_test):
    user_data = pd.read_csv(user_data_path, usecols=required_user_columns)
    
    # Normalize user data
    user_data = normalize_data(user_data, category_columns)

    # Get the user's row
    user_row = user_data[user_data["user_id"] == user_id_to_test]
    
    if user_row.empty:
        print("User not found!")
        return

    # Extract user preferences
    user_preferences = user_row[category_columns].values.flatten()
    main_preference, exploring_preference = get_user_preferences(user_row)

    # Process video data in chunks
    recommendations = []
    
    for chunk in pd.read_csv(video_data_path, usecols=required_video_columns, chunksize=chunk_size):
        chunk = normalize_data(chunk, category_columns)
        video_vectors = chunk[category_columns].values

        # Compute weighted scores
        interest_scores = np.dot(video_vectors, user_preferences) * 0.4
        
        main_scores = np.array([
            sum(main_preference.get(cat, 0) * chunk.at[i, cat] for cat in category_columns) * 0.35
            for i in range(len(chunk))
        ])
        
        exploring_scores = np.array([
            sum(exploring_preference.get(cat, 0) * chunk.at[i, cat] for cat in category_columns) * 0.25
            for i in range(len(chunk))
        ])

        # Final recommendation score
        final_scores = interest_scores + main_scores + exploring_scores
        chunk["final_score"] = final_scores

        # Get top 15 recommendations per chunk
        recommendations.append(chunk.nlargest(15, "final_score")[["id", "title", "category", "final_score"]])

    # Combine results from all chunks and get the final top 15 recommendations
    top_15_recommendations = pd.concat(recommendations).nlargest(15, "final_score")

    # Save the recommendations to a CSV file
    output_file = f"user_{test_user_id}_recommendations.csv"
    top_15_recommendations.to_csv(output_file, index=False)
    print(f"Top 15 recommendations saved to: {output_file}")

    # Print the recommendations
    print("\nTop 15 Video Recommendations:")
    print(top_15_recommendations)


# Example: Run the recommendation system for a test user ID
test_user_id = 124  # Replace with an actual user ID
get_user_recommendations(test_user_id)
